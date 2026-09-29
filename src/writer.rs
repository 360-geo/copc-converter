use crate::PipelineConfig;
use crate::TempCompression;
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
/// Nodes are read from temp files, then encoded and LAZ-compressed in
/// parallel windows sized to the memory budget; a node too large for a
/// window is sorted externally and streamed into its chunk.
use crate::copc_types::{
    CopcInfo, EVLR_HEADER_SIZE, HierarchyEntry, TEMPORAL_HEADER_SIZE, TemporalIndexEntry,
    TemporalIndexHeader, TemporalPagePointer, VoxelKey, write_evlr, write_vlr,
};
use crate::octree::{GRID_CELLS_PER_AXIS, OctreeBuilder, RawPoint, write_temp_batch};
use anyhow::{Context, Result};
use byteorder::{LittleEndian, WriteBytesExt};
use laz::laszip::{ChunkTable, ChunkTableEntry};
use laz::record::{LayeredPointRecordCompressor, RecordCompressor};
use laz::{LazVlr, LazVlrBuilder};
use rayon::prelude::*;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;
use tracing::{debug, info};

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
    buf.push(rp.flags); // classification flags / scanner channel / scan dir / edge
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
/// Reads nodes from temp files and compresses them in parallel windows
/// across all available cores.
pub fn write_copc(
    output_path: &Path,
    builder: &OctreeBuilder,
    node_keys: &[(VoxelKey, usize)],
    config: &PipelineConfig,
) -> Result<()> {
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

    // The header records the points' actual extents (LAS 1.4). Distribute
    // measured them in world coordinates; encode them exactly as the extreme
    // points are encoded, so the header matches the stored values.
    let to_grid = |v: f64, scale: f64, offset: f64| -> f64 {
        ((v - offset) / scale).round() * scale + offset
    };
    let b = &builder.bounds;
    let (min_x, min_y, min_z) = (
        to_grid(b.min_x, scale_x, offset_x),
        to_grid(b.min_y, scale_y, offset_y),
        to_grid(b.min_z, scale_z, offset_z),
    );
    let (max_x, max_y, max_z) = (
        to_grid(b.max_x, scale_x, offset_x),
        to_grid(b.max_y, scale_y, offset_y),
        to_grid(b.max_z, scale_z, offset_z),
    );

    // -----------------------------------------------------------------------
    // Build level-sorted key list from node_keys.
    // Sort by level (coarse LOD first for progressive loading), then by
    // x/y/z for deterministic order.  COPC hierarchy is a flat lookup table,
    // so strict BFS-reachability from root is not required.
    // -----------------------------------------------------------------------
    // Single sorted (key, count) list — sorted by level (coarse LOD first for
    // progressive loading) then x/y/z for determinism. Keeping counts paired
    // with keys avoids a separate `VoxelKey -> count` HashMap, which at tens of
    // millions of nodes would cost gigabytes on its own.
    let mut ordered: Vec<(VoxelKey, usize)> = node_keys.to_vec();
    ordered.sort_by(|a, b| {
        a.0.level
            .cmp(&b.0.level)
            .then(a.0.x.cmp(&b.0.x))
            .then(a.0.y.cmp(&b.0.y))
            .then(a.0.z.cmp(&b.0.z))
    });
    // Readers start traversal at the root entry, so an empty input still
    // needs one (as a zero-point node) for the file to be readable.
    if ordered.is_empty() {
        ordered.push((
            VoxelKey {
                level: 0,
                x: 0,
                y: 0,
                z: 0,
            },
            0,
        ));
    }

    debug!(
        "Writing {} nodes, {} points",
        ordered.len(),
        actual_total_points
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
    // Global encoding: bit 0 = GPS time type (carried over from the inputs),
    // bit 4 = CRS is WKT (required for point formats 6+).
    let gps_time_bit: u16 = if builder.gps_time_standard { 0x0001 } else { 0 };
    w.write_u16::<LittleEndian>(gps_time_bit | 0x0010)?;
    w.write_all(&[0u8; 16])?; // project ID (GUID)
    w.write_u8(1)?; // version major
    w.write_u8(4)?; // version minor
    let mut sysid = [0u8; 32];
    b"copc_converter"
        .iter()
        .enumerate()
        .for_each(|(i, &c)| sysid[i] = c);
    w.write_all(&sysid)?;
    w.write_all(&generating_software())?;
    let (creation_doy, creation_year) = file_creation_date();
    w.write_u16::<LittleEndian>(creation_doy)?;
    w.write_u16::<LittleEndian>(creation_year)?;
    w.write_u16::<LittleEndian>(375)?; // header size
    w.write_u32::<LittleEndian>(offset_to_point_data)?;
    w.write_u32::<LittleEndian>(num_vlrs)?; // number of VLRs
    w.write_u8(128 | point_format)?; // LAZ compressed point format
    w.write_u16::<LittleEndian>(point_record_len)?;
    // Legacy point count + 5 legacy by-return counts. LAS 1.4 R15 requires
    // these to be zero for point data record formats 6 and above, which is
    // all COPC can hold; the real counts go in the 1.4 fields below.
    w.write_u32::<LittleEndian>(0)?;
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
    // Point data: one LAZ chunk per node
    //
    // Chunks are compressed here rather than through laz's
    // ParLasZipCompressor so each node can take the path its size needs:
    // nodes that fit the window are encoded and compressed in parallel, and
    // a node larger than the window is GPS-sorted externally and streamed
    // into the compressor (see `write_oversized_node` for the one limit the
    // LAZ format imposes there).
    // -----------------------------------------------------------------------

    // LAZ convention: point data opens with the byte offset of the chunk
    // table, patched in once the table's position is known.
    w.write_i64::<LittleEndian>(-1)?;

    // Only encode nodes that have actual points (empty ancestor nodes are
    // included in the hierarchy EVLR with offset=0/byte_size=0 but not compressed).
    let data_keys: Vec<(VoxelKey, usize)> =
        ordered.iter().filter(|(_, c)| *c > 0).copied().collect();

    let window_points =
        writer_window_points(config.memory_budget, point_record_len, num_extra_bytes);
    debug!(
        "Encoding {} data nodes ({} empty ancestors), window {} points",
        data_keys.len(),
        ordered.len() - data_keys.len(),
        window_points,
    );

    let temporal_index = config.temporal_index.map(|ts| ts as usize);
    let mut totals = NodeStats::new();
    let mut temporal_entries: Vec<TemporalIndexEntry> = Vec::new();
    let mut chunk_table = ChunkTable::with_capacity(data_keys.len());

    let mut win_start = 0;
    while win_start < data_keys.len() {
        let (first_key, first_count) = data_keys[win_start];
        if first_count as u64 > window_points {
            // A node too large to hold: external sort, streamed compression.
            let start = w.stream_position()?;
            let stats = write_oversized_node(
                &mut w,
                builder,
                &first_key,
                first_count,
                window_points as usize,
                ((config.memory_budget as f64 * WRITER_BUDGET_FRACTION) as u64)
                    .max(MIN_OVERSIZED_CHUNK_BYTES),
                point_format,
                &laz_vlr,
                temporal_index,
            )?;
            let byte_count = w.stream_position()? - start;
            chunk_table.push(ChunkTableEntry {
                point_count: first_count as u64,
                byte_count,
            });
            totals.merge(&stats);
            if temporal_index.is_some() {
                temporal_entries.push(TemporalIndexEntry {
                    key: first_key,
                    samples: stats.samples,
                });
            }
            win_start += 1;
            config.report(crate::ProgressEvent::StageProgress {
                done: win_start as u64,
            });
            continue;
        }

        // Pack consecutive nodes that fit into one window.
        let mut win_points: u64 = 0;
        let mut win_end = win_start;
        while win_end < data_keys.len() {
            let node_points = data_keys[win_end].1 as u64;
            if node_points > window_points
                || (win_end > win_start && win_points + node_points > window_points)
            {
                break;
            }
            win_points += node_points;
            win_end += 1;
        }
        let window = &data_keys[win_start..win_end];

        // Encode and compress the window's nodes in parallel. Each closure's
        // point and raw-byte buffers are freed before it returns, so what
        // the window holds at once is its compressed chunks.
        let results: Vec<(Vec<u8>, NodeStats)> = window
            .par_iter()
            .map(|(key, count)| -> Result<(Vec<u8>, NodeStats)> {
                // Sized up front: a growing Vec can briefly hold twice this.
                let mut pts = Vec::with_capacity(*count);
                builder.stream_node(key, |p| {
                    pts.push(p);
                    Ok(())
                })?;
                // total_cmp: a total order even with NaN GPS times, matching
                // the oversized-node path.
                pts.sort_unstable_by(|a, b| a.gps_time.total_cmp(&b.gps_time));
                let mut stats = NodeStats::new();
                let mut raw_bytes = Vec::with_capacity(point_record_len as usize * pts.len());
                for (i, rp) in pts.iter().enumerate() {
                    stats.add(rp, i, pts.len(), temporal_index);
                    encode_point(rp, point_format, &mut raw_bytes);
                }
                drop(pts);
                Ok((compress_chunk(&raw_bytes, &laz_vlr)?, stats))
            })
            .collect::<Result<Vec<_>>>()?;

        for ((key, count), (bytes, stats)) in window.iter().zip(results) {
            w.write_all(&bytes)?;
            chunk_table.push(ChunkTableEntry {
                point_count: *count as u64,
                byte_count: bytes.len() as u64,
            });
            totals.merge(&stats);
            if temporal_index.is_some() {
                temporal_entries.push(TemporalIndexEntry {
                    key: *key,
                    samples: stats.samples,
                });
            }
        }
        config.report(crate::ProgressEvent::StageProgress {
            done: win_end as u64,
        });
        win_start = win_end;
    }

    let return_counts = totals.returns;
    // If no points were processed, report a zero GPS range.
    let (gpstime_min, gpstime_max) = if totals.gps_min > totals.gps_max {
        (0.0, 0.0)
    } else {
        (totals.gps_min, totals.gps_max)
    };

    // Chunk table after the last chunk; then point the offset at the start
    // of the point data to it.
    let chunk_table_pos = w.stream_position()?;
    chunk_table.write_to(&mut w, &laz_vlr)?;
    let evlr_start = w.stream_position()?;
    w.seek(SeekFrom::Start(offset_to_point_data as u64))?;
    w.write_i64::<LittleEndian>(chunk_table_pos as i64)?;
    w.flush()?;
    let mut file = w
        .into_inner()
        .map_err(|e| anyhow::anyhow!("BufWriter flush: {}", e.error()))?;

    // -----------------------------------------------------------------------
    // Build chunk_info for the hierarchy EVLR
    // -----------------------------------------------------------------------
    let first_chunk_start = offset_to_point_data as u64 + 8;
    let mut current_offset = first_chunk_start;
    let mut chunk_info: Vec<(VoxelKey, u64, i32, i32)> = Vec::new();
    let mut chunk_index = 0usize;

    for (key, pc) in &ordered {
        if *pc == 0 {
            // Empty ancestor: present in hierarchy for tree traversal but has no chunk.
            chunk_info.push((*key, 0, 0, 0));
        } else {
            // Hierarchy entries store byte size and point count as i32.
            let byte_size = chunk_table[chunk_index].byte_count;
            let entry_bytes = i32::try_from(byte_size).map_err(|_| {
                anyhow::anyhow!("node {key:?} is {byte_size} bytes, exceeding the COPC i32 limit")
            })?;
            let entry_points = i32::try_from(*pc).map_err(|_| {
                anyhow::anyhow!("node {key:?} has {pc} points, exceeding the COPC i32 limit")
            })?;
            chunk_info.push((*key, current_offset, entry_bytes, entry_points));
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

/// Fraction of the memory budget the writer's window may occupy. Nothing
/// else of size is resident while writing: the build's working set is gone.
const WRITER_BUDGET_FRACTION: f64 = 0.5;

/// Upper cap on the window, in points. Past this, larger windows only add
/// memory: a few million points already keep every core busy.
const MAX_WINDOW_POINTS: u64 = 4_000_000;

/// Lower bound on the window, in points, so tiny budgets still make
/// progress in reasonably sized steps.
const MIN_WINDOW_POINTS: u64 = 100_000;

/// Smallest compressed-chunk allowance for an oversized node, like the
/// window floor above: an artificially tiny `--memory-limit` shouldn't
/// reject nodes that compress to a few MB.
const MIN_OVERSIZED_CHUNK_BYTES: u64 = 64 * 1024 * 1024;

/// Points per writer window: the most whose transient state fits the
/// writer's share of the budget. Per point that is the decoded `RawPoint`
/// plus its extra bytes, the encoded record, and the compressed output
/// (bounded by the record size, even for incompressible data).
fn writer_window_points(memory_budget: u64, record_len: u16, num_extra_bytes: u16) -> u64 {
    let per_point =
        (std::mem::size_of::<RawPoint>() + num_extra_bytes as usize) as u64 + 2 * record_len as u64;
    ((memory_budget as f64 * WRITER_BUDGET_FRACTION) as u64 / per_point)
        .clamp(MIN_WINDOW_POINTS, MAX_WINDOW_POINTS)
}

/// LAZ-compress one node's encoded point records as a single chunk.
fn compress_chunk(raw: &[u8], vlr: &LazVlr) -> Result<Vec<u8>> {
    let mut compressor = LayeredPointRecordCompressor::new(Vec::new());
    compressor.set_fields_from(vlr.items())?;
    compressor.compress_many(raw)?;
    compressor.done()?;
    Ok(compressor.into_inner())
}

/// Per-node statistics folded into the header, COPC info and temporal index.
struct NodeStats {
    returns: [u64; 15],
    gps_min: f64,
    gps_max: f64,
    samples: Vec<f64>,
}

impl NodeStats {
    fn new() -> Self {
        Self {
            returns: [0; 15],
            gps_min: f64::MAX,
            gps_max: f64::MIN,
            samples: Vec::new(),
        }
    }

    /// Account for point `i` of a GPS-sorted node of `n` points.
    fn add(&mut self, rp: &RawPoint, i: usize, n: usize, temporal_stride: Option<usize>) {
        let rn = rp.return_number as usize;
        if (1..=15).contains(&rn) {
            self.returns[rn - 1] += 1;
        }
        self.gps_min = self.gps_min.min(rp.gps_time);
        self.gps_max = self.gps_max.max(rp.gps_time);
        if let Some(stride) = temporal_stride
            && (i.is_multiple_of(stride) || i == n - 1)
        {
            self.samples.push(rp.gps_time);
        }
    }

    /// Fold another node's totals in (samples stay per node).
    fn merge(&mut self, other: &NodeStats) {
        for (a, b) in self.returns.iter_mut().zip(&other.returns) {
            *a += b;
        }
        self.gps_min = self.gps_min.min(other.gps_min);
        self.gps_max = self.gps_max.max(other.gps_max);
    }
}

/// Write one node that is too large to hold in memory as a single LAZ
/// chunk, GPS-sorted like every other node: sort it in runs of
/// `run_points`, spill each run to a temp file, merge runs (at most
/// `MAX_MERGE_FANIN` open at once) and stream the final merge into the
/// compressor. Point memory is one run plus a read buffer per open run.
///
/// The LAZ format itself sets the floor: point formats 6–8 compress each
/// field into its own layer, and a chunk's layers are all buffered until the
/// chunk is finished. So the node's *compressed* size must fit
/// `max_chunk_bytes`. It is estimated from the first sorted run before any
/// merging, so a node that can't fit fails early with an actionable error
/// rather than getting the process OOM-killed.
#[allow(clippy::too_many_arguments)]
fn write_oversized_node<W: Write>(
    out: &mut W,
    builder: &OctreeBuilder,
    key: &VoxelKey,
    count: usize,
    run_points: usize,
    max_chunk_bytes: u64,
    point_format: u8,
    vlr: &LazVlr,
    temporal_stride: Option<usize>,
) -> Result<NodeStats> {
    let run_dir = builder
        .tmp_dir
        .join(format!("sort_{}_{}_{}_{}", key.level, key.x, key.y, key.z));
    std::fs::create_dir_all(&run_dir).with_context(|| format!("creating {run_dir:?}"))?;
    let result = sort_and_compress_node(
        out,
        builder,
        key,
        count,
        run_points,
        max_chunk_bytes,
        point_format,
        vlr,
        temporal_stride,
        &run_dir,
    );
    let _ = std::fs::remove_dir_all(&run_dir);
    result
}

#[allow(clippy::too_many_arguments)]
fn sort_and_compress_node<W: Write>(
    out: &mut W,
    builder: &OctreeBuilder,
    key: &VoxelKey,
    count: usize,
    run_points: usize,
    max_chunk_bytes: u64,
    point_format: u8,
    vlr: &LazVlr,
    temporal_stride: Option<usize>,
    run_dir: &Path,
) -> Result<NodeStats> {
    let nxb = builder.num_extra_bytes;
    let record_len = point_record_length(point_format, nxb) as usize;

    // Pass 1: sorted runs. The first one also measures how well this node
    // compresses, to check the finished chunk will fit before merging.
    let mut runs: Vec<std::path::PathBuf> = Vec::new();
    let mut buf: Vec<RawPoint> = Vec::with_capacity(run_points);
    let mut spill = |buf: &mut Vec<RawPoint>| -> Result<()> {
        buf.sort_unstable_by(|a, b| a.gps_time.total_cmp(&b.gps_time));
        if runs.is_empty() {
            let mut raw = Vec::with_capacity(buf.len() * record_len);
            for p in buf.iter() {
                encode_point(p, point_format, &mut raw);
            }
            let bytes_per_point = compress_chunk(&raw, vlr)?.len() as f64 / buf.len() as f64;
            // ×2: the layer buffers grow by doubling.
            let estimate = (bytes_per_point * count as f64 * 2.0) as u64;
            if estimate > max_chunk_bytes || estimate / 2 > i32::MAX as u64 {
                anyhow::bail!(
                    "node {key:?} holds {count} points too close together to split \
                     (e.g. duplicates of one location); compressing it as one LAZ chunk \
                     needs about {} MB, over the {} MB this memory budget allows \
                     (and COPC caps a chunk at 2 GiB). Raise --memory-limit or \
                     thin the duplicate points.",
                    estimate / 1_048_576,
                    max_chunk_bytes / 1_048_576,
                );
            }
        }
        let path = run_dir.join(format!("run_{}", runs.len()));
        write_run(&path, buf, nxb)?;
        runs.push(path);
        buf.clear();
        Ok(())
    };
    builder.stream_node(key, |p| {
        buf.push(p);
        if buf.len() == run_points {
            spill(&mut buf)?;
        }
        Ok(())
    })?;
    if !buf.is_empty() {
        spill(&mut buf)?;
    }
    drop(buf);

    // Pass 2: merge, streamed into the compressor.
    debug!(
        "Oversized node {key:?}: {count} points in {} runs",
        runs.len()
    );
    let mut compressor = LayeredPointRecordCompressor::new(&mut *out);
    compressor.set_fields_from(vlr.items())?;
    let mut raw = Vec::with_capacity(ENCODE_BATCH_POINTS * record_len);
    let mut stats = NodeStats::new();
    let mut i = 0;
    merge_runs_bounded(runs, run_dir, nxb, MAX_MERGE_FANIN, |p| {
        stats.add(&p, i, count, temporal_stride);
        encode_point(&p, point_format, &mut raw);
        i += 1;
        if raw.len() >= ENCODE_BATCH_POINTS * record_len {
            compressor.compress_many(&raw)?;
            raw.clear();
        }
        Ok(())
    })?;
    compressor.compress_many(&raw)?;
    compressor.done()?;
    anyhow::ensure!(i == count, "node {key:?}: merged {i} of {count} points");
    Ok(stats)
}

/// Most sorted runs open at once while merging an oversized node; more are
/// first merged in groups. Keeps file descriptors and read buffers bounded.
const MAX_MERGE_FANIN: usize = 64;

/// Write one sorted run as an uncompressed temp batch. Runs stay
/// uncompressed so `RunReader` can pull points one at a time.
fn write_run(path: &Path, points: &[RawPoint], num_extra_bytes: u16) -> Result<()> {
    let mut f = BufWriter::new(std::fs::File::create(path)?);
    write_temp_batch(&mut f, points, num_extra_bytes, TempCompression::None)?;
    f.flush()?;
    Ok(())
}

/// Merge GPS-sorted runs, calling `emit` with every point in GPS order,
/// with at most `fanin` runs open at once: while there are more, groups of
/// `fanin` are first merged into new runs in `dir`.
fn merge_runs_bounded(
    mut runs: Vec<std::path::PathBuf>,
    dir: &Path,
    num_extra_bytes: u16,
    fanin: usize,
    emit: impl FnMut(RawPoint) -> Result<()>,
) -> Result<()> {
    let fanin = fanin.max(2);
    let mut generation = 0;
    while runs.len() > fanin {
        generation += 1;
        let mut merged = Vec::with_capacity(runs.len().div_ceil(fanin));
        for (i, group) in runs.chunks(fanin).enumerate() {
            let path = dir.join(format!("merge_{generation}_{i}"));
            let mut w = BufWriter::new(std::fs::File::create(&path)?);
            let mut batch = Vec::with_capacity(ENCODE_BATCH_POINTS);
            merge_runs(group, num_extra_bytes, |p| {
                batch.push(p);
                if batch.len() == ENCODE_BATCH_POINTS {
                    write_temp_batch(&mut w, &batch, num_extra_bytes, TempCompression::None)?;
                    batch.clear();
                }
                Ok(())
            })?;
            write_temp_batch(&mut w, &batch, num_extra_bytes, TempCompression::None)?;
            w.flush()?;
            for p in group {
                let _ = std::fs::remove_file(p);
            }
            merged.push(path);
        }
        runs = merged;
    }
    merge_runs(&runs, num_extra_bytes, emit)
}

/// K-way merge of GPS-sorted runs, calling `emit` with points in GPS order.
fn merge_runs(
    runs: &[std::path::PathBuf],
    num_extra_bytes: u16,
    mut emit: impl FnMut(RawPoint) -> Result<()>,
) -> Result<()> {
    let mut readers = runs
        .iter()
        .map(|p| RunReader::open(p, num_extra_bytes))
        .collect::<Result<Vec<_>>>()?;
    let mut heap = std::collections::BinaryHeap::new();
    for (i, r) in readers.iter_mut().enumerate() {
        if let Some(p) = r.next()? {
            heap.push(std::cmp::Reverse(HeapItem(p, i)));
        }
    }
    while let Some(std::cmp::Reverse(HeapItem(p, run))) = heap.pop() {
        emit(p)?;
        if let Some(next) = readers[run].next()? {
            heap.push(std::cmp::Reverse(HeapItem(next, run)));
        }
    }
    Ok(())
}

/// Points encoded per `compress_many` call when streaming a chunk.
const ENCODE_BATCH_POINTS: usize = 65_536;

/// A run's current point, ordered by GPS time (ties by run index).
struct HeapItem(RawPoint, usize);

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}
impl Eq for HeapItem {}
impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .gps_time
            .total_cmp(&other.0.gps_time)
            .then(self.1.cmp(&other.1))
    }
}

/// Pulls points one at a time from a sorted run written by
/// `write_oversized_node` (uncompressed temp batches).
struct RunReader {
    r: std::io::BufReader<std::fs::File>,
    remaining: u32,
    record: Vec<u8>,
    num_extra_bytes: u16,
}

impl RunReader {
    fn open(path: &Path, num_extra_bytes: u16) -> Result<Self> {
        Ok(Self {
            r: std::io::BufReader::new(std::fs::File::open(path)?),
            remaining: 0,
            record: vec![0; RawPoint::record_size(num_extra_bytes)],
            num_extra_bytes,
        })
    }

    fn next(&mut self) -> Result<Option<RawPoint>> {
        use byteorder::ReadBytesExt;
        while self.remaining == 0 {
            match self.r.read_u32::<LittleEndian>() {
                Ok(n) => self.remaining = n,
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
                Err(e) => return Err(e.into()),
            }
        }
        std::io::Read::read_exact(&mut self.r, &mut self.record)?;
        self.remaining -= 1;
        Ok(Some(RawPoint::from_record(
            &self.record,
            self.num_extra_bytes,
        )))
    }
}

/// The header's Generating Software field: `copc_converter <version>`,
/// null-padded (or truncated) to 32 bytes. Release builds get their version
/// from the git tag (CI sets it in Cargo.toml before building); local builds
/// report `0.0.0-dev`.
fn generating_software() -> [u8; 32] {
    let name = concat!("copc_converter ", env!("CARGO_PKG_VERSION"));
    let mut field = [0u8; 32];
    let len = name.len().min(field.len());
    field[..len].copy_from_slice(&name.as_bytes()[..len]);
    field
}

/// Today's date as `(day_of_year, year)` for the LAS header's File Creation
/// fields, derived from the system clock (UTC). Returns `(0, 0)` if the
/// clock is before the Unix epoch.
fn file_creation_date() -> (u16, u16) {
    let secs = match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        Ok(d) => d.as_secs(),
        Err(_) => return (0, 0),
    };
    civil_date_from_unix_days((secs / 86_400) as i64)
}

/// Convert days since 1970-01-01 to `(day_of_year, year)` with day-of-year
/// starting at 1 (Howard Hinnant's `civil_from_days`).
fn civil_date_from_unix_days(days: i64) -> (u16, u16) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let mut year = yoe + era * 400;
    let doy_march = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy_march + 2) / 153;
    let day = doy_march - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    if month <= 2 {
        year += 1;
    }
    let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    const CUMULATIVE: [i64; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    let mut doy = CUMULATIVE[(month - 1) as usize] + day;
    if leap && month > 2 {
        doy += 1;
    }
    (doy as u16, year as u16)
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
            flags: 0b1011_1101,
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
    fn encode_point_preserves_las_1_4_flags() {
        let p = sample_point();
        let mut buf = Vec::new();
        encode_point(&p, 8, &mut buf);
        assert_eq!(buf[15], p.flags);
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
            for entry_bytes in page.as_chunks::<32>().0 {
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
        for chunk in root_bytes.as_chunks::<32>().0 {
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

    #[test]
    fn civil_date_known_days() {
        assert_eq!(civil_date_from_unix_days(0), (1, 1970));
        // 2024-12-31 is day 366 of a leap year (20_088 days since epoch)
        assert_eq!(civil_date_from_unix_days(20_088), (366, 2024));
        // 2025-03-01 = 20_148 days since epoch, day 60 in a common year
        assert_eq!(civil_date_from_unix_days(20_148), (60, 2025));
        // 2024-03-01 = 19_783 days since epoch, day 61 in a leap year
        assert_eq!(civil_date_from_unix_days(19_783), (61, 2024));
    }

    #[test]
    fn bounded_run_merge_sorts_across_passes() {
        // 50 runs with fan-in 4 forces three intermediate merge passes.
        let dir = std::env::temp_dir().join(format!("copc_test_runs_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut runs = Vec::new();
        let mut expected = Vec::new();
        for r in 0..50u32 {
            let mut pts: Vec<RawPoint> = (0..37u32)
                .map(|i| {
                    let mut p = sample_point();
                    p.gps_time = f64::from((i * 50 + r).wrapping_mul(2_654_435_761) % 10_007);
                    p
                })
                .collect();
            pts.sort_unstable_by(|a, b| a.gps_time.total_cmp(&b.gps_time));
            expected.extend(pts.iter().map(|p| p.gps_time));
            let path = dir.join(format!("run_{r}"));
            write_run(&path, &pts, 0).unwrap();
            runs.push(path);
        }
        expected.sort_unstable_by(f64::total_cmp);

        let mut got = Vec::new();
        merge_runs_bounded(runs, &dir, 0, 4, |p| {
            got.push(p.gps_time);
            Ok(())
        })
        .unwrap();
        assert_eq!(got, expected);
        std::fs::remove_dir_all(&dir).ok();
    }
}
