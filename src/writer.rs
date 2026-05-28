use crate::PipelineConfig;
/// Write a COPC 1.0 file.
///
/// Layout
/// ------
///  [LAS 1.4 header]           375 bytes
///  [copc info VLR]            54 + 160 = 214 bytes
///  [laszip VLR]               54 + variable (depends on point format)
///  [WKT CRS VLR]              optional
///  [i64 chunk-table offset]   8 bytes  (points to chunk table after all data)
///  [compressed chunk 0]       variable
///  [compressed chunk 1]       variable
///  ...
///  [LAZ chunk table]          variable (appended after data, referenced by the i64 above)
///  [copc hierarchy EVLR]      60 + n*32 bytes
///
/// Uses ParLasZipCompressor for parallel chunk compression via rayon.
/// Nodes are read from temp files and encoded in parallel batches, then
/// compressed in parallel via compress_chunks(). The chunk table is read
/// back from the file to recover per-chunk byte sizes for the hierarchy.
use crate::copc_types::{
    CopcInfo, EVLR_HEADER_SIZE, HierarchyEntry, TEMPORAL_HEADER_SIZE, TemporalIndexEntry,
    TemporalIndexHeader, TemporalPagePointer, VoxelKey, write_evlr, write_vlr,
};
use crate::octree::{GRID_CELLS_PER_AXIS, OctreeBuilder, RawPoint};
use anyhow::{Context, Result};
use byteorder::{LittleEndian, WriteBytesExt};
use laz::{LazVlrBuilder, ParLasZipCompressor};
use rayon::prelude::*;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;
use tracing::{debug, error, info};

// ---------------------------------------------------------------------------
// Point record sizes: format 6 = 30, format 7 = 36, format 8 = 38
// ---------------------------------------------------------------------------
fn point_record_length(fmt: u8, num_extra_bytes: u16) -> u16 {
    let base = match fmt {
        6 => 30,
        7 => 36,
        8 => 38,
        _ => 36,
    };
    base + num_extra_bytes
}

/// Encode the format-6 base fields (30 bytes) shared by all COPC formats.
fn encode_point_base(rp: &RawPoint, buf: &mut Vec<u8>) {
    buf.extend_from_slice(&rp.x.to_le_bytes());
    buf.extend_from_slice(&rp.y.to_le_bytes());
    buf.extend_from_slice(&rp.z.to_le_bytes());
    buf.extend_from_slice(&rp.intensity.to_le_bytes());
    let return_byte = (rp.return_number & 0x0F) | ((rp.number_of_returns & 0x0F) << 4);
    buf.push(return_byte);
    buf.push(0u8); // classification flags / scanner channel / scan dir / edge
    buf.push(rp.classification);
    buf.push(rp.user_data);
    buf.extend_from_slice(&rp.scan_angle.to_le_bytes());
    buf.extend_from_slice(&rp.point_source_id.to_le_bytes());
    buf.extend_from_slice(&rp.gps_time.to_le_bytes());
    // Total = 4+4+4+2+1+1+1+1+2+2+8 = 30 bytes
}

/// Encode one point according to the COPC output format (6, 7, or 8),
/// appending any per-point extra bytes after the format-specific fields.
fn encode_point(rp: &RawPoint, fmt: u8, buf: &mut Vec<u8>) {
    encode_point_base(rp, buf);
    if fmt >= 7 {
        buf.extend_from_slice(&rp.red.to_le_bytes());
        buf.extend_from_slice(&rp.green.to_le_bytes());
        buf.extend_from_slice(&rp.blue.to_le_bytes());
    }
    if fmt >= 8 {
        buf.extend_from_slice(&rp.nir.to_le_bytes());
    }
    buf.extend_from_slice(&rp.extras);
}

/// Write a complete COPC file to `output_path`.
///
/// Reads nodes from temp files and compresses them in parallel using
/// ParLasZipCompressor::compress_chunks(). Encoding and compression
/// happen across all available cores.
pub fn write_copc(
    output_path: &Path,
    builder: &OctreeBuilder,
    node_keys: &[(VoxelKey, usize)],
    config: &PipelineConfig,
) -> Result<()> {
    let memory_budget = config.memory_budget;
    let scale_x = builder.scale_x;
    let scale_y = builder.scale_y;
    let scale_z = builder.scale_z;
    let offset_x = builder.offset_x;
    let offset_y = builder.offset_y;
    let offset_z = builder.offset_z;

    let point_format = builder.point_format;
    let num_extra_bytes = builder.num_extra_bytes;
    let point_record_len = point_record_length(point_format, num_extra_bytes);

    // -----------------------------------------------------------------------
    // Build the LAZ VLR (variable-size chunks)
    // -----------------------------------------------------------------------
    let laz_vlr = LazVlrBuilder::default()
        .with_point_format(point_format, num_extra_bytes)
        .context("LazVlrBuilder for format")?
        .with_variable_chunk_size()
        .build();

    let mut laz_vlr_payload: Vec<u8> = Vec::new();
    laz_vlr.write_to(&mut laz_vlr_payload)?;

    // -----------------------------------------------------------------------
    // File layout constants
    // -----------------------------------------------------------------------
    let wkt_crs = &builder.wkt_crs;
    let extra_bytes_vlr = &builder.extra_bytes_vlr;
    let copc_info_vlr_size: u32 = 54 + 160; // 214
    let laz_vlr_size: u32 = 54 + laz_vlr_payload.len() as u32;
    let wkt_vlr_size: u32 = wkt_crs.as_ref().map(|d| 54 + d.len() as u32).unwrap_or(0);
    let extra_bytes_vlr_size: u32 = extra_bytes_vlr
        .as_ref()
        .map(|d| 54 + d.len() as u32)
        .unwrap_or(0);
    let mut num_vlrs: u32 = 2; // copc info + laszip
    if wkt_crs.is_some() {
        num_vlrs += 1;
    }
    if extra_bytes_vlr.is_some() {
        num_vlrs += 1;
    }
    let offset_to_point_data: u32 =
        375 + copc_info_vlr_size + laz_vlr_size + wkt_vlr_size + extra_bytes_vlr_size;

    let copc_info_payload_pos: u64 = 375 + 54;

    // Use the actual point count from node_keys (not builder.total_points which
    // is the original input count — the write-back sampling may have moved points).
    let actual_total_points: u64 = node_keys.iter().map(|(_, c)| *c as u64).sum();
    debug!(
        "Header total_points: {} (original: {})",
        actual_total_points, builder.total_points
    );

    // Snap the bounding box to the scale+offset grid so that the header
    // min/max values are exactly representable as (offset + n*scale).
    // Input files may use different offsets, so their reported bounds can
    // be non-integer multiples of our scale — lasinfo warns about this.
    let snap_floor = |v: f64, scale: f64, offset: f64| -> f64 {
        ((v - offset) / scale).floor() * scale + offset
    };
    let snap_ceil =
        |v: f64, scale: f64, offset: f64| -> f64 { ((v - offset) / scale).ceil() * scale + offset };
    let b = &builder.bounds;
    let (min_x, min_y, min_z) = (
        snap_floor(b.min_x, scale_x, offset_x),
        snap_floor(b.min_y, scale_y, offset_y),
        snap_floor(b.min_z, scale_z, offset_z),
    );
    let (max_x, max_y, max_z) = (
        snap_ceil(b.max_x, scale_x, offset_x),
        snap_ceil(b.max_y, scale_y, offset_y),
        snap_ceil(b.max_z, scale_z, offset_z),
    );

    // -----------------------------------------------------------------------
    // Build level-sorted key list from node_keys.
    // Sort by level (coarse LOD first for progressive loading), then by
    // x/y/z for deterministic order.  COPC hierarchy is a flat lookup table,
    // so strict BFS-reachability from root is not required.
    // -----------------------------------------------------------------------
    let point_counts: std::collections::HashMap<VoxelKey, usize> =
        node_keys.iter().copied().collect();

    let mut ordered_keys: Vec<VoxelKey> = node_keys.iter().map(|(k, _)| *k).collect();
    ordered_keys.sort_by(|a, b| {
        a.level
            .cmp(&b.level)
            .then(a.x.cmp(&b.x))
            .then(a.y.cmp(&b.y))
            .then(a.z.cmp(&b.z))
    });

    debug!(
        "Writing {} nodes, {} points",
        ordered_keys.len(),
        actual_total_points,
    );

    // -----------------------------------------------------------------------
    // Write LAS 1.4 header manually (375 bytes)
    // -----------------------------------------------------------------------
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(output_path)
        .with_context(|| format!("Cannot create {:?}", output_path))?;
    let mut w = BufWriter::new(file);

    w.write_all(b"LASF")?;
    w.write_u16::<LittleEndian>(0)?;
    w.write_u16::<LittleEndian>(0x0001 | 0x0010)?; // GPS standard + WKT
    w.write_all(&[0u8; 16])?; // project ID (GUID)
    w.write_u8(1)?; // version major
    w.write_u8(4)?; // version minor
    let mut sysid = [0u8; 32];
    b"copc_converter"
        .iter()
        .enumerate()
        .for_each(|(i, &c)| sysid[i] = c);
    w.write_all(&sysid)?;
    let mut gensoft = [0u8; 32];
    b"copc_converter 0.1"
        .iter()
        .enumerate()
        .for_each(|(i, &c)| gensoft[i] = c);
    w.write_all(&gensoft)?;
    w.write_u16::<LittleEndian>(1)?; // file creation day
    w.write_u16::<LittleEndian>(2024)?; // file creation year
    w.write_u16::<LittleEndian>(375)?; // header size
    w.write_u32::<LittleEndian>(offset_to_point_data)?;
    w.write_u32::<LittleEndian>(num_vlrs)?; // number of VLRs
    w.write_u8(128 | point_format)?; // LAZ compressed point format
    w.write_u16::<LittleEndian>(point_record_len)?;
    w.write_u32::<LittleEndian>(0)?; // legacy point count
    for _ in 0..5 {
        w.write_u32::<LittleEndian>(0)?;
    }
    w.write_f64::<LittleEndian>(scale_x)?;
    w.write_f64::<LittleEndian>(scale_y)?;
    w.write_f64::<LittleEndian>(scale_z)?;
    w.write_f64::<LittleEndian>(offset_x)?;
    w.write_f64::<LittleEndian>(offset_y)?;
    w.write_f64::<LittleEndian>(offset_z)?;
    w.write_f64::<LittleEndian>(max_x)?;
    w.write_f64::<LittleEndian>(min_x)?;
    w.write_f64::<LittleEndian>(max_y)?;
    w.write_f64::<LittleEndian>(min_y)?;
    w.write_f64::<LittleEndian>(max_z)?;
    w.write_f64::<LittleEndian>(min_z)?;
    w.write_u64::<LittleEndian>(0)?; // start of waveform data
    w.write_u64::<LittleEndian>(0)?; // start_of_first_EVLR – patched below
    let num_evlrs: u32 = if config.temporal_index.is_some() {
        2
    } else {
        1
    };
    w.write_u32::<LittleEndian>(num_evlrs)?; // number of EVLRs
    w.write_u64::<LittleEndian>(actual_total_points)?;
    for _ in 0..15 {
        w.write_u64::<LittleEndian>(0)?;
    }

    // -----------------------------------------------------------------------
    // VLR 1: copc info (placeholder – patched at the end)
    // -----------------------------------------------------------------------
    let copc_info_placeholder = CopcInfo {
        center_x: builder.cx,
        center_y: builder.cy,
        center_z: builder.cz,
        halfsize: builder.halfsize,
        spacing: 2.0 * builder.halfsize / GRID_CELLS_PER_AXIS as f64,
        root_hier_offset: 0,
        root_hier_size: 0,
        gpstime_minimum: 0.0,
        gpstime_maximum: 0.0,
    };
    let mut copc_info_buf = Vec::with_capacity(160);
    copc_info_placeholder.write(&mut copc_info_buf)?;
    write_vlr(&mut w, "copc", 1, "copc info", &copc_info_buf)?;

    // -----------------------------------------------------------------------
    // VLR 2: laszip VLR
    // -----------------------------------------------------------------------
    write_vlr(
        &mut w,
        "laszip encoded",
        22204,
        "laz variable chunks",
        &laz_vlr_payload,
    )?;

    // -----------------------------------------------------------------------
    // VLR 3 (optional): WKT CRS
    // -----------------------------------------------------------------------
    if let Some(wkt_data) = wkt_crs {
        write_vlr(&mut w, "LASF_Projection", 2112, "WKT", wkt_data)?;
    }

    // -----------------------------------------------------------------------
    // VLR 4 (optional): LAS Extra Bytes schema, passed through unchanged.
    // -----------------------------------------------------------------------
    if let Some(eb_data) = extra_bytes_vlr {
        write_vlr(&mut w, "LASF_Spec", 4, "Extra Bytes Record", eb_data)?;
    }

    w.flush()?;

    // -----------------------------------------------------------------------
    // Parallel compression via ParLasZipCompressor
    // -----------------------------------------------------------------------
    let laz_vlr_for_compressor = LazVlrBuilder::default()
        .with_point_format(point_format, num_extra_bytes)
        .context("LazVlrBuilder (compressor)")?
        .with_variable_chunk_size()
        .build();

    let mut compressor = ParLasZipCompressor::new(w, laz_vlr_for_compressor)
        .map_err(|e| anyhow::anyhow!("ParLasZipCompressor::new: {e}"))?;

    compressor
        .reserve_offset_to_chunk_table()
        .context("reserve_offset_to_chunk_table")?;

    // Only encode nodes that have actual points (empty ancestor nodes are
    // included in the hierarchy EVLR with offset=0/byte_size=0 but not compressed).
    let data_keys: Vec<VoxelKey> = ordered_keys
        .iter()
        .filter(|k| point_counts.get(k).copied().unwrap_or(0) > 0)
        .copied()
        .collect();

    // Writer memory model:
    //
    // The hot loop has two phases inside each outer batch:
    //
    //   Phase 1 — parallel read + encode. Each worker loads a node's
    //     Vec<RawPoint> (38 B/pt), sort-in-place, and produces a Vec<u8>
    //     of encoded bytes (point_record_len B/pt). The RawPoint buffer
    //     is dropped at end of closure. Peak: a few concurrent workers
    //     holding pts + raw_bytes, plus the *entire* rayon result vec
    //     accumulating raw_bytes from every finished node. Steady-state
    //     peak ≈ batch_points × point_record_len (since the RawPoint
    //     buffer's life is bounded by one worker's runtime, not the batch).
    //
    //   Phase 2 — compress. laz's ParLasZipCompressor::compress_chunks
    //     takes the encoded Vec<Vec<u8>>, parallel-compresses each into
    //     a new Vec<u8>, and collects the compressed outputs into a
    //     Vec<(usize, Vec<u8>)> before writing any to the output file
    //     (so chunks stay in input order for the chunk table).
    //     Peak during this call ≈ uncompressed input + compressed output
    //     ≈ batch_points × (point_record_len + ~6 B) ≈ batch_points × 42 B.
    //
    // Instead of trying to cram both phases into one budget sizing, we:
    //   1. Size the outer batch to fit Phase 1's peak (the larger of the two
    //      when everything is batched together). batch_points × point_record_len
    //      with a fragmentation safety factor.
    //   2. During Phase 2, break the encoded Vec into mini-batches of
    //      WRITER_COMPRESS_MINI_BATCH nodes and call compress_chunks on each.
    //      Freeing each mini-batch after compression means Phase 2 peak is
    //      bounded by a small constant, not by batch size.
    //
    // Net effect: wall time is dominated by Phase 1 which runs as big as the
    // budget allows (maximum CPU utilization); Phase 2 memory is capped at a
    // few hundred MB regardless of batch size.
    //
    // Fragmentation safety factor accounts for everything the
    // `batch_points × point_record_len` naive estimate misses: concurrent
    // RawPoint worker buffers held during parallel encoding, the NodeResult
    // tuple overhead in the rayon output Vec, per-node temporal sample Vecs
    // (when the temporal index is enabled), Phase 2 compressor scratch, and
    // allocator retention across many batches. 2× gives genuine headroom
    // without halving batch size so aggressively that wall-time suffers.
    const FRAGMENTATION_FACTOR: u64 = 20; // 2.0× as integer math
    let phase1_bytes_per_point = (point_record_len as u64 * FRAGMENTATION_FACTOR).div_ceil(10);

    // Mini-batch size for Phase 2. 32 nodes is enough to keep laz's internal
    // rayon parallelism saturated on typical 8-16 core machines (each core
    // gets at least 2-4 nodes per mini-batch, amortizing work-stealing
    // overhead) while keeping per-mini-batch memory at ~32 × avg_node_bytes.
    const WRITER_COMPRESS_MINI_BATCH: usize = 32;

    debug!(
        "Encoding {} data nodes ({} empty ancestors), budget {} MiB, \
         phase1 {} B/pt incl. fragmentation, compress mini-batch {}",
        data_keys.len(),
        ordered_keys.len() - data_keys.len(),
        memory_budget / 1_048_576,
        phase1_bytes_per_point,
        WRITER_COMPRESS_MINI_BATCH,
    );

    let mut return_counts = [0u64; 15];
    let mut gpstime_min = f64::MAX;
    let mut gpstime_max = f64::MIN;
    let mut temporal_entries: Vec<TemporalIndexEntry> = Vec::new();
    let temporal_index = config.temporal_index.map(|ts| ts as usize);

    let mut batch_start = 0;
    while batch_start < data_keys.len() {
        // Greedy-pack nodes into the batch up to phase 1's memory bound.
        let mut batch_bytes: u64 = 0;
        let mut batch_end = batch_start;
        while batch_end < data_keys.len() {
            let key = &data_keys[batch_end];
            let node_bytes =
                (point_counts.get(key).copied().unwrap_or(0) as u64) * phase1_bytes_per_point;
            // Always include at least one node per batch to avoid stalling.
            if batch_end > batch_start && batch_bytes + node_bytes > memory_budget {
                break;
            }
            batch_bytes += node_bytes;
            batch_end += 1;
        }

        let batch = &data_keys[batch_start..batch_end];
        let batch_points: u64 = batch
            .iter()
            .map(|k| point_counts.get(k).copied().unwrap_or(0) as u64)
            .sum();
        debug!(
            "Write batch {}: {} nodes, {} points, phase1 est {} MB (budget {} MB)",
            batch_start,
            batch.len(),
            batch_points,
            batch_bytes / 1_048_576,
            memory_budget / 1_048_576,
        );

        // Phase 1: encode every node in the batch in parallel. The RawPoint
        // buffers are dropped inside each closure, so the memory that
        // survives into `results` is only the encoded Vec<u8> per node.
        type NodeResult = (Vec<u8>, [u64; 15], f64, f64, Vec<f64>);
        let results: Vec<NodeResult> = batch
            .par_iter()
            .map(|key| -> Result<NodeResult> {
                let mut pts = builder.read_node(key)?;
                pts.sort_unstable_by(|a, b| {
                    a.gps_time
                        .partial_cmp(&b.gps_time)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let mut local_returns = [0u64; 15];
                let mut local_gps_min = f64::MAX;
                let mut local_gps_max = f64::MIN;
                let mut samples = Vec::new();
                let mut raw_bytes = Vec::with_capacity(point_record_len as usize * pts.len());
                for (i, rp) in pts.iter().enumerate() {
                    let rn = rp.return_number as usize;
                    if (1..=15).contains(&rn) {
                        local_returns[rn - 1] += 1;
                    }
                    if rp.gps_time < local_gps_min {
                        local_gps_min = rp.gps_time;
                    }
                    if rp.gps_time > local_gps_max {
                        local_gps_max = rp.gps_time;
                    }
                    if let Some(stride) = temporal_index
                        && (i % stride == 0 || i == pts.len() - 1)
                    {
                        samples.push(rp.gps_time);
                    }
                    encode_point(rp, point_format, &mut raw_bytes);
                }
                Ok((
                    raw_bytes,
                    local_returns,
                    local_gps_min,
                    local_gps_max,
                    samples,
                ))
            })
            .collect::<Result<Vec<_>>>()?;

        // Drain per-node stats into the running aggregates, and split the
        // encoded bytes out into a separate Vec so we can drop it in
        // mini-batch chunks without holding onto the NodeResult tuples.
        let mut encoded: Vec<Vec<u8>> = Vec::with_capacity(results.len());
        let mut actual_encoded_bytes: u64 = 0;
        for (i, (bytes, local_returns, local_min, local_max, samples)) in
            results.into_iter().enumerate()
        {
            for j in 0..15 {
                return_counts[j] += local_returns[j];
            }
            if local_min < gpstime_min {
                gpstime_min = local_min;
            }
            if local_max > gpstime_max {
                gpstime_max = local_max;
            }
            if temporal_index.is_some() {
                temporal_entries.push(TemporalIndexEntry {
                    key: batch[i],
                    samples,
                });
            }
            actual_encoded_bytes += bytes.len() as u64;
            encoded.push(bytes);
        }
        debug!(
            "Write batch {}: phase1 actual encoded {} MB (est {} MB, delta {:+})",
            batch_start,
            actual_encoded_bytes / 1_048_576,
            batch_bytes / 1_048_576,
            (actual_encoded_bytes as i64 - batch_bytes as i64) / 1_048_576,
        );

        // Phase 2: compress in mini-batches, freeing each mini-batch's
        // encoded bytes before the next is built. This caps Phase 2's
        // peak at `WRITER_COMPRESS_MINI_BATCH × avg_node_bytes ×
        // (1 + 1/compression_ratio)` ≈ a few hundred MB regardless of the
        // outer batch size.
        //
        // We process the mini-batches in order via drain(..end) so the
        // already-compressed prefix is freed as we go (drain leaves the
        // remaining suffix in `encoded` with its allocation intact).
        while !encoded.is_empty() {
            let take = WRITER_COMPRESS_MINI_BATCH.min(encoded.len());
            // drain(..take) yields the first `take` elements, shifting the
            // remainder to the front. The yielded Vec<Vec<u8>> is collected
            // (so its inner buffers move into the new vec) and consumed by
            // compress_chunks, then dropped — at which point all encoded
            // bytes in the mini-batch are freed.
            let mini: Vec<Vec<u8>> = encoded.drain(..take).collect();
            compressor
                .compress_chunks(mini)
                .context("compress_chunks")?;
        }

        config.report(crate::ProgressEvent::StageProgress {
            done: batch_end as u64,
        });
        batch_start = batch_end;
    }

    // If no points were processed, reset GPS time to 0.
    if gpstime_min > gpstime_max {
        gpstime_min = 0.0;
        gpstime_max = 0.0;
    }

    compressor.done().context("compressor done")?;

    let mut w = compressor.into_inner();
    w.flush()?;
    // After done(), the stream position may be at the patched offset location.
    // Get file size by seeking to the end.
    let end_pos = w.seek(SeekFrom::End(0))?;

    // Unwrap the BufWriter to get the underlying File for seek+read
    let mut file = w
        .into_inner()
        .map_err(|e| anyhow::anyhow!("BufWriter flush: {}", e.error()))?;

    // -----------------------------------------------------------------------
    // Read the chunk table back from the file to get per-chunk byte sizes
    // -----------------------------------------------------------------------
    let read_vlr = LazVlrBuilder::default()
        .with_point_format(point_format, num_extra_bytes)
        .context("LazVlrBuilder (read)")?
        .with_variable_chunk_size()
        .build();

    file.seek(SeekFrom::Start(offset_to_point_data as u64))?;
    let chunk_table = laz::laszip::ChunkTable::read_from(&mut file, &read_vlr)
        .map_err(|e| anyhow::anyhow!("Failed to read chunk table: {e}"))?;

    // -----------------------------------------------------------------------
    // Verify chunk table
    // -----------------------------------------------------------------------
    let evlr_start = end_pos;
    if chunk_table.len() != data_keys.len() {
        error!(
            "Chunk table has {} entries but we compressed {} chunks!",
            chunk_table.len(),
            data_keys.len()
        );
    }

    // -----------------------------------------------------------------------
    // Build chunk_info for the hierarchy EVLR
    // -----------------------------------------------------------------------
    let first_chunk_start = offset_to_point_data as u64 + 8;
    let mut current_offset = first_chunk_start;
    let mut chunk_info: Vec<(VoxelKey, u64, i32, i32)> = Vec::new();
    let mut chunk_index = 0usize;

    for key in &ordered_keys {
        let pc = point_counts.get(key).copied().unwrap_or(0);
        if pc == 0 {
            // Empty ancestor: present in hierarchy for tree traversal but has no chunk.
            chunk_info.push((*key, 0, 0, 0));
        } else {
            let byte_size = chunk_table[chunk_index].byte_count;
            chunk_info.push((*key, current_offset, byte_size as i32, pc as i32));
            current_offset += byte_size;
            chunk_index += 1;
        }
    }

    // -----------------------------------------------------------------------
    // EVLR: copc hierarchy (paged)
    //
    // Entries are split into a tree of pages so readers don't need to fetch
    // the whole hierarchy before rendering the root node. The root page's
    // offset and size are reported through the `CopcInfo` VLR so readers
    // know where to start; page pointers inside pages are real
    // `HierarchyEntry` records with `point_count = -1` (spec sentinel).
    // -----------------------------------------------------------------------

    let hier_evlr_data_start = evlr_start + EVLR_HEADER_SIZE as u64;
    let (hier_payload, hier_root_page_offset, hier_root_page_size) =
        build_hierarchy_payload(&chunk_info, hier_evlr_data_start)?;

    file.seek(SeekFrom::Start(evlr_start))?;
    let mut w = BufWriter::new(file);
    write_evlr(&mut w, "copc", 1000, "copc hierarchy", &hier_payload)?;

    // -----------------------------------------------------------------------
    // EVLR: temporal index (optional) — v2 paged layout
    // -----------------------------------------------------------------------
    if let Some(temporal_stride) = temporal_index {
        // Current file position is where the EVLR record header starts.
        // The EVLR data payload begins 60 bytes later.
        let temporal_evlr_start = w.stream_position()?;
        let evlr_data_start = temporal_evlr_start + EVLR_HEADER_SIZE as u64;

        let temporal_payload =
            build_temporal_payload(&temporal_entries, temporal_stride as u32, evlr_data_start)?;

        write_evlr(
            &mut w,
            "copc_temporal",
            1000,
            "temporal index",
            &temporal_payload,
        )?;
    }

    w.flush()?;
    let mut file = w
        .into_inner()
        .map_err(|e| anyhow::anyhow!("BufWriter flush: {}", e.error()))?;

    // -----------------------------------------------------------------------
    // Patch the file: copc info VLR + EVLR start offset
    // -----------------------------------------------------------------------
    let patched_info = CopcInfo {
        center_x: builder.cx,
        center_y: builder.cy,
        center_z: builder.cz,
        halfsize: builder.halfsize,
        spacing: 2.0 * builder.halfsize / GRID_CELLS_PER_AXIS as f64,
        root_hier_offset: hier_root_page_offset,
        root_hier_size: hier_root_page_size,
        gpstime_minimum: gpstime_min,
        gpstime_maximum: gpstime_max,
    };
    let mut pinfo_buf = Vec::with_capacity(160);
    patched_info.write(&mut pinfo_buf)?;
    file.seek(SeekFrom::Start(copc_info_payload_pos))?;
    file.write_all(&pinfo_buf)?;

    // Patch EVLR start offset
    file.seek(SeekFrom::Start(235))?;
    file.write_all(&evlr_start.to_le_bytes())?;

    // Patch number of points by return (15 × u64 starting at header offset 255)
    file.seek(SeekFrom::Start(255))?;
    for &count in &return_counts {
        file.write_all(&count.to_le_bytes())?;
    }

    info!("COPC file written: {:?}", output_path);
    Ok(())
}

// ---------------------------------------------------------------------------
// COPC hierarchy EVLR paged layout
// ---------------------------------------------------------------------------

/// One entry in the input to the hierarchy payload builder.
/// `(key, chunk_offset, byte_size, point_count)` — empty ancestors use
/// `(key, 0, 0, 0)` per COPC spec.
type HierarchyInputEntry = (VoxelKey, u64, i32, i32);

/// A single page in the COPC hierarchy EVLR, produced before absolute
/// child-page offsets are known. `pointer_patches` lists the byte
/// positions inside `data` where child page offset/size fields live,
/// paired with the index of the child `HierarchyPage` in the flat list.
struct HierarchyPage {
    data: Vec<u8>,
    pointer_patches: Vec<(usize, usize)>,
}

/// Recursively build hierarchy pages for a set of entries.
///
/// Entries with level **strictly less than** the current boundary stay in
/// this page as regular entries. Entries at the boundary level and below
/// are grouped by their ancestor at the boundary level — each group
/// becomes a child page. For each group, the parent page emits a single
/// `HierarchyEntry` with `point_count = -1` whose key matches the subtree
/// root; per COPC spec this tells readers "the entry for this node lives
/// in another hierarchy page" and they follow the pointer. The subtree
/// root's real entry, along with all its descendants, lives inside the
/// child page.
fn build_hierarchy_page_recursive(
    entries: &[&HierarchyInputEntry],
    boundaries: &[i32],
    boundary_idx: usize,
    pages: &mut Vec<HierarchyPage>,
) -> anyhow::Result<usize> {
    if boundary_idx >= boundaries.len() || entries.is_empty() {
        let mut data = Vec::with_capacity(entries.len() * 32);
        for (key, offset, byte_size, point_count) in entries {
            HierarchyEntry {
                key: *key,
                offset: *offset,
                byte_size: *byte_size,
                point_count: *point_count,
            }
            .write(&mut data)?;
        }
        let page_idx = pages.len();
        pages.push(HierarchyPage {
            data,
            pointer_patches: Vec::new(),
        });
        return Ok(page_idx);
    }

    let boundary_level = boundaries[boundary_idx];

    // Split: entries strictly above the boundary stay in this page; the
    // boundary level and everything below it goes into child pages, grouped
    // by their ancestor at the boundary level.
    let mut this_page_entries: Vec<&HierarchyInputEntry> = Vec::new();
    let mut child_groups: std::collections::BTreeMap<VoxelKey, Vec<&HierarchyInputEntry>> =
        std::collections::BTreeMap::new();
    for &entry in entries {
        let (key, _, _, _) = entry;
        if key.level < boundary_level {
            this_page_entries.push(entry);
        } else {
            let subtree_root = ancestor_at_level(*key, boundary_level);
            child_groups.entry(subtree_root).or_default().push(entry);
        }
    }

    // Reserve a slot for this page so children can know the flat-list index
    // to aim at with pointer patches.
    let this_page_idx = pages.len();
    pages.push(HierarchyPage {
        data: Vec::new(),
        pointer_patches: Vec::new(),
    });

    // Recurse into each child subtree first so we know all child page indices
    // before serialising this page's pointers.
    struct ChildInfo {
        subtree_root: VoxelKey,
        child_page_idx: usize,
    }
    let mut children: Vec<ChildInfo> = Vec::with_capacity(child_groups.len());
    for (subtree_root, child_entries) in &child_groups {
        let child_refs: Vec<&HierarchyInputEntry> = child_entries.to_vec();
        let child_page_idx =
            build_hierarchy_page_recursive(&child_refs, boundaries, boundary_idx + 1, pages)?;
        children.push(ChildInfo {
            subtree_root: *subtree_root,
            child_page_idx,
        });
    }

    // Serialise this page: regular entries first, then page pointers.
    let mut data = Vec::with_capacity((this_page_entries.len() + children.len()) * 32);
    for (key, offset, byte_size, point_count) in &this_page_entries {
        HierarchyEntry {
            key: *key,
            offset: *offset,
            byte_size: *byte_size,
            point_count: *point_count,
        }
        .write(&mut data)?;
    }

    // A hierarchy page pointer is a HierarchyEntry with point_count = -1.
    // `offset` carries the child page's absolute file offset (patched later)
    // and `byte_size` carries the child page's size. The two fields live at
    // bytes [16..24] (offset, u64) and [24..28] (byte_size, i32) inside each
    // 32-byte HierarchyEntry.
    let mut pointer_patches = Vec::with_capacity(children.len());
    for child in &children {
        let patch_offset = data.len() + 16;
        HierarchyEntry {
            key: child.subtree_root,
            offset: 0,    // placeholder — patched later
            byte_size: 0, // placeholder — patched later
            point_count: -1,
        }
        .write(&mut data)?;
        pointer_patches.push((patch_offset, child.child_page_idx));
    }

    pages[this_page_idx] = HierarchyPage {
        data,
        pointer_patches,
    };
    Ok(this_page_idx)
}

/// Build the COPC hierarchy EVLR payload with nested pages.
///
/// Returns `(payload_bytes, root_page_offset, root_page_size)`. The root
/// page may live anywhere inside the payload; the CopcInfo VLR carries
/// its absolute offset and size so readers can find it without scanning.
fn build_hierarchy_payload(
    entries: &[HierarchyInputEntry],
    evlr_data_start: u64,
) -> anyhow::Result<(Vec<u8>, u64, u64)> {
    if entries.is_empty() {
        return Ok((Vec::new(), evlr_data_start, 0));
    }

    let max_level = entries
        .iter()
        .map(|(k, _, _, _)| k.level)
        .max()
        .unwrap_or(0);
    let boundaries = choose_page_boundaries(max_level);

    let entry_refs: Vec<&HierarchyInputEntry> = entries.iter().collect();
    let mut pages: Vec<HierarchyPage> = Vec::new();
    let root_page_idx = build_hierarchy_page_recursive(&entry_refs, &boundaries, 0, &mut pages)?;

    // Lay pages out sequentially from the EVLR data start.
    let mut page_offsets: Vec<u64> = Vec::with_capacity(pages.len());
    let mut offset = evlr_data_start;
    for page in &pages {
        page_offsets.push(offset);
        offset += page.data.len() as u64;
    }

    // Patch child page offset/size fields in each page.
    for i in 0..pages.len() {
        let patches: Vec<(usize, u64, u32)> = pages[i]
            .pointer_patches
            .iter()
            .map(|&(patch_offset, child_idx)| {
                (
                    patch_offset,
                    page_offsets[child_idx],
                    pages[child_idx].data.len() as u32,
                )
            })
            .collect();
        for (patch_offset, abs_offset, size) in patches {
            pages[i].data[patch_offset..patch_offset + 8]
                .copy_from_slice(&abs_offset.to_le_bytes());
            pages[i].data[patch_offset + 8..patch_offset + 12].copy_from_slice(&size.to_le_bytes());
        }
    }

    let root_page_offset = page_offsets[root_page_idx];
    let root_page_size = pages[root_page_idx].data.len() as u64;

    let total: usize = pages.iter().map(|p| p.data.len()).sum();
    let mut payload = Vec::with_capacity(total);
    for page in &pages {
        payload.extend_from_slice(&page.data);
    }
    Ok((payload, root_page_offset, root_page_size))
}

// ---------------------------------------------------------------------------
// Temporal index v2 paged layout
// ---------------------------------------------------------------------------

/// Choose multiple page boundary levels for nested pages.
///
/// Places boundaries every 3 levels of the octree, starting at level 3.
/// For example:
///  - max_level=9:  [3]
///  - max_level=12: [3, 6, 9]
///  - max_level=15: [3, 6, 9, 12]
///
/// If the tree is very shallow (max_level <= 3), returns an empty vec (single
/// root page, no child pages needed).
fn choose_page_boundaries(max_level: i32) -> Vec<i32> {
    let mut boundaries = Vec::new();
    let mut l = 3;
    while l < max_level {
        boundaries.push(l);
        l += 3;
    }
    if boundaries.is_empty() && max_level > 3 {
        boundaries.push(max_level.min(3));
    }
    boundaries
}

/// Returns the ancestor VoxelKey at the given level.
fn ancestor_at_level(key: VoxelKey, level: i32) -> VoxelKey {
    let mut k = key;
    while k.level > level {
        k = k.parent().unwrap();
    }
    k
}

/// Compute the time range (min, max) across all entries in a slice.
///
/// Returns `(f64::MAX, f64::MIN)` if no entries have samples.
fn time_range_of(entries: &[&TemporalIndexEntry]) -> (f64, f64) {
    let mut tmin = f64::MAX;
    let mut tmax = f64::MIN;
    for e in entries {
        if let Some(&first) = e.samples.first()
            && first < tmin
        {
            tmin = first;
        }
        if let Some(&last) = e.samples.last()
            && last > tmax
        {
            tmax = last;
        }
    }
    (tmin, tmax)
}

/// A page produced by the recursive page builder. Contains its serialized node
/// entries and page pointers (with placeholder offsets), plus metadata needed to
/// patch in the correct absolute offsets in a second pass.
struct BuiltPage {
    /// Serialized bytes: node entries followed by page pointers.
    data: Vec<u8>,
    /// For each page pointer written into `data`, the byte offset within `data`
    /// where the `child_page_offset` u64 field starts, plus the index of the
    /// child `BuiltPage` in the flat page list.
    pointer_patches: Vec<(usize, usize)>,
}

/// Recursively build pages for a set of entries.
///
/// `entries` — all entries belonging to this page's subtree.
/// `boundaries` — the full list of page boundary levels.
/// `boundary_idx` — which boundary we are splitting at (index into `boundaries`).
/// `pages` — accumulator for all built pages (flat list, appended in order).
///
/// Returns the index of this page in `pages`.
fn build_page_recursive(
    entries: &[&TemporalIndexEntry],
    boundaries: &[i32],
    boundary_idx: usize,
    pages: &mut Vec<BuiltPage>,
) -> anyhow::Result<usize> {
    // If no more boundaries, or the subtree is empty, write all entries into one page.
    if boundary_idx >= boundaries.len() || entries.is_empty() {
        let mut data = Vec::new();
        for entry in entries {
            entry.write(&mut data)?;
        }
        let page_idx = pages.len();
        pages.push(BuiltPage {
            data,
            pointer_patches: Vec::new(),
        });
        return Ok(page_idx);
    }

    let boundary_level = boundaries[boundary_idx];

    // Split entries into those that belong in this page (level <= boundary)
    // and those that go into child pages (level > boundary).
    let mut this_page_entries: Vec<&TemporalIndexEntry> = Vec::new();
    let mut child_groups: std::collections::BTreeMap<VoxelKey, Vec<&TemporalIndexEntry>> =
        std::collections::BTreeMap::new();

    for &entry in entries {
        if entry.key.level <= boundary_level {
            this_page_entries.push(entry);
        } else {
            let subtree_root = ancestor_at_level(entry.key, boundary_level);
            child_groups.entry(subtree_root).or_default().push(entry);
        }
    }

    // If there are no child groups, just write everything into one page.
    if child_groups.is_empty() {
        let mut data = Vec::new();
        for entry in &this_page_entries {
            entry.write(&mut data)?;
        }
        let page_idx = pages.len();
        pages.push(BuiltPage {
            data,
            pointer_patches: Vec::new(),
        });
        return Ok(page_idx);
    }

    // Reserve a slot for this page in the flat list.
    let this_page_idx = pages.len();
    pages.push(BuiltPage {
        data: Vec::new(),
        pointer_patches: Vec::new(),
    });

    // Recursively build child pages. We need to collect their info before
    // writing this page, since we need child page indices for patching.
    struct ChildInfo {
        subtree_root: VoxelKey,
        child_page_idx: usize,
        /// Time range across ALL descendants in this subtree (including entries
        /// at the boundary level that are in the parent page).
        time_min: f64,
        time_max: f64,
    }

    let mut children: Vec<ChildInfo> = Vec::new();
    for (subtree_root, child_entries) in &child_groups {
        // Compute time range over ALL descendants: child_entries (deeper) plus
        // the subtree root node itself if it appears in this_page_entries.
        let (mut tmin, mut tmax) = time_range_of(child_entries);
        if let Some(root_entry) = this_page_entries.iter().find(|e| e.key == *subtree_root) {
            let (rmin, rmax) = time_range_of(&[root_entry]);
            tmin = tmin.min(rmin);
            tmax = tmax.max(rmax);
        }

        let child_refs: Vec<&TemporalIndexEntry> = child_entries.to_vec();
        let child_page_idx =
            build_page_recursive(&child_refs, boundaries, boundary_idx + 1, pages)?;

        children.push(ChildInfo {
            subtree_root: *subtree_root,
            child_page_idx,
            time_min: tmin,
            time_max: tmax,
        });
    }

    // Now serialize this page: node entries first, then page pointers.
    let mut data = Vec::new();
    for entry in &this_page_entries {
        entry.write(&mut data)?;
    }

    let mut pointer_patches = Vec::new();
    for child in &children {
        // Record where the child_page_offset field will be so we can patch it.
        // In the TemporalPagePointer layout:
        //   VoxelKey (16) + sample_count=0 (4) + child_page_offset (8) ...
        // So child_page_offset starts at current position + 20.
        let patch_offset = data.len() + 20;

        TemporalPagePointer {
            key: child.subtree_root,
            child_page_offset: 0, // placeholder — patched later
            child_page_size: 0,   // placeholder — patched later
            subtree_time_min: child.time_min,
            subtree_time_max: child.time_max,
        }
        .write(&mut data)?;

        pointer_patches.push((patch_offset, child.child_page_idx));
    }

    pages[this_page_idx] = BuiltPage {
        data,
        pointer_patches,
    };

    Ok(this_page_idx)
}

/// Build the complete temporal index EVLR payload with nested pages.
///
/// `evlr_data_start` is the absolute file offset where the EVLR data payload
/// begins (i.e., after the 60-byte EVLR record header).
fn build_temporal_payload(
    entries: &[TemporalIndexEntry],
    stride: u32,
    evlr_data_start: u64,
) -> anyhow::Result<Vec<u8>> {
    if entries.is_empty() {
        let mut payload = Vec::new();
        TemporalIndexHeader {
            version: 1,
            stride,
            node_count: 0,
            page_count: 1,
            root_page_offset: evlr_data_start + TEMPORAL_HEADER_SIZE as u64,
            root_page_size: 0,
        }
        .write(&mut payload)?;
        return Ok(payload);
    }

    let max_level = entries.iter().map(|e| e.key.level).max().unwrap_or(0);
    let boundaries = choose_page_boundaries(max_level);

    // Build all pages recursively into a flat list.
    let entry_refs: Vec<&TemporalIndexEntry> = entries.iter().collect();
    let mut pages: Vec<BuiltPage> = Vec::new();
    let root_page_idx = build_page_recursive(&entry_refs, &boundaries, 0, &mut pages)?;

    // Compute absolute offsets for each page. Pages are laid out sequentially
    // after the header.
    let pages_start = evlr_data_start + TEMPORAL_HEADER_SIZE as u64;
    let mut page_offsets: Vec<u64> = Vec::with_capacity(pages.len());
    let mut offset = pages_start;
    for page in &pages {
        page_offsets.push(offset);
        offset += page.data.len() as u64;
    }

    // Patch child_page_offset and child_page_size in each page's data.
    for i in 0..pages.len() {
        // Collect patches first to avoid borrow issues.
        let patches: Vec<(usize, u64, u32)> = pages[i]
            .pointer_patches
            .iter()
            .map(|&(patch_offset, child_idx)| {
                (
                    patch_offset,
                    page_offsets[child_idx],
                    pages[child_idx].data.len() as u32,
                )
            })
            .collect();

        for (patch_offset, abs_offset, size) in patches {
            // Patch child_page_offset (8 bytes at patch_offset).
            pages[i].data[patch_offset..patch_offset + 8]
                .copy_from_slice(&abs_offset.to_le_bytes());
            // Patch child_page_size (4 bytes immediately after).
            pages[i].data[patch_offset + 8..patch_offset + 12].copy_from_slice(&size.to_le_bytes());
        }
    }

    let root_page_offset = page_offsets[root_page_idx];
    let root_page_size = pages[root_page_idx].data.len() as u32;
    let page_count = pages.len() as u32;
    let node_count = entries.len() as u32;

    // Assemble the final payload: header + all pages in order.
    let mut payload = Vec::new();
    TemporalIndexHeader {
        version: 1,
        stride,
        node_count,
        page_count,
        root_page_offset,
        root_page_size,
    }
    .write(&mut payload)?;

    for page in &pages {
        payload.extend_from_slice(&page.data);
    }

    Ok(payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_point() -> RawPoint {
        RawPoint {
            x: -123456,
            y: 789012,
            z: -1,
            intensity: 65535,
            return_number: 3,
            number_of_returns: 5,
            classification: 6,
            scan_angle: -15000,
            user_data: 42,
            point_source_id: 1001,
            gps_time: 123456789.987654,
            red: 255,
            green: 0,
            blue: 65535,
            nir: 32768,
            extras: Box::<[u8]>::default(),
        }
    }

    fn sample_point_with_extras(extras: &[u8]) -> RawPoint {
        let mut p = sample_point();
        p.extras = extras.to_vec().into_boxed_slice();
        p
    }

    #[test]
    fn point_record_lengths() {
        assert_eq!(point_record_length(6, 0), 30);
        assert_eq!(point_record_length(7, 0), 36);
        assert_eq!(point_record_length(8, 0), 38);
        assert_eq!(point_record_length(6, 12), 42);
        assert_eq!(point_record_length(7, 12), 48);
        assert_eq!(point_record_length(8, 12), 50);
    }

    #[test]
    fn encode_point_appends_extras() {
        let p = sample_point_with_extras(&[0xAA, 0xBB, 0xCC, 0xDD]);
        let mut buf = Vec::new();
        encode_point(&p, 7, &mut buf);
        // format-7 base is 36 bytes; extras (4) should follow.
        assert_eq!(buf.len(), 40);
        assert_eq!(&buf[36..], &[0xAA, 0xBB, 0xCC, 0xDD]);
    }

    #[test]
    fn encode_point_format6_size() {
        let p = sample_point();
        let mut buf = Vec::new();
        encode_point(&p, 6, &mut buf);
        assert_eq!(buf.len(), 30);
    }

    #[test]
    fn encode_point_format7_size() {
        let p = sample_point();
        let mut buf = Vec::new();
        encode_point(&p, 7, &mut buf);
        assert_eq!(buf.len(), 36);
    }

    #[test]
    fn encode_point_format8_size() {
        let p = sample_point();
        let mut buf = Vec::new();
        encode_point(&p, 8, &mut buf);
        assert_eq!(buf.len(), 38);
    }

    #[test]
    fn encode_point_format7_includes_rgb() {
        let p = sample_point();
        let mut buf = Vec::new();
        encode_point(&p, 7, &mut buf);
        // RGB starts at offset 30 (after base fields)
        let red = u16::from_le_bytes([buf[30], buf[31]]);
        let green = u16::from_le_bytes([buf[32], buf[33]]);
        let blue = u16::from_le_bytes([buf[34], buf[35]]);
        assert_eq!(red, p.red);
        assert_eq!(green, p.green);
        assert_eq!(blue, p.blue);
    }

    #[test]
    fn encode_point_format8_includes_nir() {
        let p = sample_point();
        let mut buf = Vec::new();
        encode_point(&p, 8, &mut buf);
        // NIR starts at offset 36 (after RGB)
        let nir = u16::from_le_bytes([buf[36], buf[37]]);
        assert_eq!(nir, p.nir);
    }

    /// Helper: simulate the temporal sampling logic from the encoding loop.
    fn sample_gps_times(gps_times: &[f64], stride: usize) -> Vec<f64> {
        let mut samples = Vec::new();
        for (i, &t) in gps_times.iter().enumerate() {
            if i % stride == 0 || i == gps_times.len() - 1 {
                samples.push(t);
            }
        }
        samples
    }

    #[test]
    fn temporal_sampling_basic() {
        // 5000 points, stride 1000 → indices 0, 1000, 2000, 3000, 4000, 4999
        let times: Vec<f64> = (0..5000).map(|i| i as f64 * 0.1).collect();
        let samples = sample_gps_times(&times, 1000);
        assert_eq!(samples.len(), 6);
        assert_eq!(samples[0], 0.0);
        assert_eq!(samples[5], 4999.0 * 0.1);
    }

    #[test]
    fn temporal_sampling_fewer_than_stride() {
        // 50 points, stride 1000 → just first and last
        let times: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let samples = sample_gps_times(&times, 1000);
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0], 0.0);
        assert_eq!(samples[1], 49.0);
    }

    #[test]
    fn temporal_sampling_single_point() {
        let samples = sample_gps_times(&[42.0], 1000);
        assert_eq!(samples, vec![42.0]);
    }

    #[test]
    fn temporal_sampling_exact_stride() {
        // 1000 points, stride 1000 → indices 0 and 999
        let times: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let samples = sample_gps_times(&times, 1000);
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0], 0.0);
        assert_eq!(samples[1], 999.0);
    }

    #[test]
    fn encode_point_format6_matches_base_of_format7() {
        let p = sample_point();
        let mut buf6 = Vec::new();
        let mut buf7 = Vec::new();
        encode_point(&p, 6, &mut buf6);
        encode_point(&p, 7, &mut buf7);
        assert_eq!(
            buf6[..],
            buf7[..30],
            "format 6 must match the first 30 bytes of format 7"
        );
    }

    // -----------------------------------------------------------------------
    // Hierarchy paging
    // -----------------------------------------------------------------------

    /// Decode an in-memory hierarchy payload by following page pointers from
    /// the root. Returns every data entry reachable through the page tree,
    /// independent of serialisation order.
    fn collect_hierarchy(
        payload: &[u8],
        evlr_data_start: u64,
        root_offset: u64,
        root_size: u64,
    ) -> Vec<HierarchyInputEntry> {
        fn read_page(
            payload: &[u8],
            evlr_data_start: u64,
            offset: u64,
            size: u64,
            out: &mut Vec<HierarchyInputEntry>,
        ) {
            let start = (offset - evlr_data_start) as usize;
            let end = start + size as usize;
            let page = &payload[start..end];
            assert!(
                size.is_multiple_of(32),
                "hierarchy page size must be a multiple of 32"
            );
            for entry_bytes in page.chunks_exact(32) {
                let level = i32::from_le_bytes(entry_bytes[0..4].try_into().unwrap());
                let x = i32::from_le_bytes(entry_bytes[4..8].try_into().unwrap());
                let y = i32::from_le_bytes(entry_bytes[8..12].try_into().unwrap());
                let z = i32::from_le_bytes(entry_bytes[12..16].try_into().unwrap());
                let entry_offset = u64::from_le_bytes(entry_bytes[16..24].try_into().unwrap());
                let entry_byte_size = i32::from_le_bytes(entry_bytes[24..28].try_into().unwrap());
                let entry_point_count = i32::from_le_bytes(entry_bytes[28..32].try_into().unwrap());
                let key = VoxelKey { level, x, y, z };
                if entry_point_count == -1 {
                    read_page(
                        payload,
                        evlr_data_start,
                        entry_offset,
                        entry_byte_size as u64,
                        out,
                    );
                } else {
                    out.push((key, entry_offset, entry_byte_size, entry_point_count));
                }
            }
        }
        let mut out = Vec::new();
        read_page(payload, evlr_data_start, root_offset, root_size, &mut out);
        out
    }

    #[test]
    fn hierarchy_paging_small_tree_stays_flat() {
        // All entries within the shallow boundary → single page, no pointers.
        let entries: Vec<HierarchyInputEntry> = vec![
            (
                VoxelKey {
                    level: 0,
                    x: 0,
                    y: 0,
                    z: 0,
                },
                100,
                42,
                10,
            ),
            (
                VoxelKey {
                    level: 1,
                    x: 1,
                    y: 0,
                    z: 0,
                },
                200,
                84,
                20,
            ),
            (
                VoxelKey {
                    level: 2,
                    x: 3,
                    y: 1,
                    z: 1,
                },
                300,
                168,
                40,
            ),
        ];
        let (payload, root_off, root_size) = build_hierarchy_payload(&entries, 1_000).unwrap();

        // Every 32-byte record in the payload must be a regular entry;
        // none should be page pointers.
        assert_eq!(payload.len() as u64, root_size);
        assert_eq!(root_off, 1_000);

        let decoded = collect_hierarchy(&payload, 1_000, root_off, root_size);
        assert_eq!(decoded.len(), 3);
        // Order within a flat page matches insertion order.
        assert_eq!(decoded[0].0.level, 0);
        assert_eq!(decoded[2].0.level, 2);
    }

    #[test]
    fn hierarchy_paging_deep_tree_produces_multiple_pages() {
        // Build enough entries across levels 0–8 to trigger at least one
        // page split (boundaries kick in at levels 3, 6, ...).
        let mut entries: Vec<HierarchyInputEntry> = Vec::new();
        for level in 0..=8 {
            let span = 1 << level;
            for x in 0..span.min(3) {
                for y in 0..span.min(3) {
                    for z in 0..span.min(3) {
                        let key = VoxelKey { level, x, y, z };
                        entries.push((key, 100 + entries.len() as u64, 42, 10));
                    }
                }
            }
        }
        let n_entries = entries.len();

        let (payload, root_off, root_size) = build_hierarchy_payload(&entries, 10_000).unwrap();

        // Payload should contain more than just the root page when entries
        // span past the first boundary.
        assert!(
            payload.len() as u64 > root_size,
            "deep tree must produce a payload larger than the root page alone"
        );

        // Following page pointers from the root must recover exactly the
        // input set (ignoring order).
        let mut decoded = collect_hierarchy(&payload, 10_000, root_off, root_size);
        decoded.sort_by_key(|(k, _, _, _)| (k.level, k.x, k.y, k.z));
        let mut expected = entries.clone();
        expected.sort_by_key(|(k, _, _, _)| (k.level, k.x, k.y, k.z));
        assert_eq!(decoded.len(), n_entries);
        for (a, b) in decoded.iter().zip(expected.iter()) {
            assert_eq!(a.0, b.0, "key mismatch");
            assert_eq!(a.1, b.1, "offset mismatch for {:?}", a.0);
            assert_eq!(a.2, b.2, "byte_size mismatch for {:?}", a.0);
            assert_eq!(a.3, b.3, "point_count mismatch for {:?}", a.0);
        }
    }

    #[test]
    fn hierarchy_paging_empty_input_returns_empty_payload() {
        let (payload, root_off, root_size) = build_hierarchy_payload(&[], 5_000).unwrap();
        assert!(payload.is_empty());
        assert_eq!(root_off, 5_000);
        assert_eq!(root_size, 0);
    }

    /// Verify the spec-correct split: a boundary-level node is NOT a
    /// regular entry in the parent page — it must only appear there as a
    /// page pointer, with its real entry living inside the child page.
    #[test]
    fn hierarchy_paging_boundary_node_lives_in_child_page() {
        // First boundary is level 3 (see choose_page_boundaries).
        // Construct entries whose max level crosses the boundary.
        let boundary = 3i32;
        let boundary_key = VoxelKey {
            level: boundary,
            x: 1,
            y: 2,
            z: 3,
        };
        let descendant = VoxelKey {
            level: boundary + 1,
            x: 2,
            y: 5,
            z: 6, // child under (1,2,3) at level 3
        };
        let entries: Vec<HierarchyInputEntry> = vec![
            (
                VoxelKey {
                    level: 0,
                    x: 0,
                    y: 0,
                    z: 0,
                },
                100,
                42,
                10,
            ),
            (
                VoxelKey {
                    level: 1,
                    x: 0,
                    y: 0,
                    z: 0,
                },
                200,
                42,
                10,
            ),
            (
                VoxelKey {
                    level: 2,
                    x: 0,
                    y: 1,
                    z: 1,
                },
                300,
                42,
                10,
            ),
            (boundary_key, 400, 42, 10),
            (descendant, 500, 42, 10),
        ];

        let evlr_data_start = 7_000u64;
        let (payload, root_off, root_size) =
            build_hierarchy_payload(&entries, evlr_data_start).unwrap();

        // Parse the root page directly to see what's in it.
        let root_start = (root_off - evlr_data_start) as usize;
        let root_end = root_start + root_size as usize;
        let root_bytes = &payload[root_start..root_end];

        // Collect both regular entries and page pointers from the root.
        let mut root_regular: Vec<(VoxelKey, i32)> = Vec::new();
        let mut root_pointers: Vec<VoxelKey> = Vec::new();
        for chunk in root_bytes.chunks_exact(32) {
            let level = i32::from_le_bytes(chunk[0..4].try_into().unwrap());
            let x = i32::from_le_bytes(chunk[4..8].try_into().unwrap());
            let y = i32::from_le_bytes(chunk[8..12].try_into().unwrap());
            let z = i32::from_le_bytes(chunk[12..16].try_into().unwrap());
            let point_count = i32::from_le_bytes(chunk[28..32].try_into().unwrap());
            let key = VoxelKey { level, x, y, z };
            if point_count == -1 {
                root_pointers.push(key);
            } else {
                root_regular.push((key, point_count));
            }
        }

        // Root page must contain only levels 0..boundary as regular entries.
        for (key, _) in &root_regular {
            assert!(
                key.level < boundary,
                "root page contained a regular entry at boundary level: {key:?}"
            );
        }

        // The boundary node must appear as a page pointer, not a regular entry.
        assert!(
            root_pointers.contains(&boundary_key),
            "expected a page pointer for boundary node {boundary_key:?} in root page, got pointers {root_pointers:?}"
        );
        assert!(
            !root_regular.iter().any(|(k, _)| *k == boundary_key),
            "boundary node {boundary_key:?} must not appear as a regular entry in the root page"
        );

        // And the full traversal still recovers every entry.
        let mut decoded = collect_hierarchy(&payload, evlr_data_start, root_off, root_size);
        decoded.sort_by_key(|(k, _, _, _)| (k.level, k.x, k.y, k.z));
        let mut expected = entries.clone();
        expected.sort_by_key(|(k, _, _, _)| (k.level, k.x, k.y, k.z));
        assert_eq!(decoded, expected);
    }
}
