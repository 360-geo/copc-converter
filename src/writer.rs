/// Write a COPC 1.0 file.
///
/// Layout
/// ------
///  [LAS 1.4 header]           375 bytes
///  [copc info VLR]            54 + 160 = 214 bytes
///  [laszip VLR]               54 + 46  = 100 bytes   (user_id "laszip encoded", record_id 22204)
///  --- offset_to_point_data = 689 ---
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
    CopcInfo, EVLR_HEADER_SIZE, HierarchyEntry, VoxelKey, write_evlr, write_vlr,
};
use crate::octree::{OctreeBuilder, RawPoint};
use anyhow::{Context, Result};
use byteorder::{LittleEndian, WriteBytesExt};
use laz::{LazVlrBuilder, ParLasZipCompressor};
use log::info;
use rayon::prelude::*;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

// ---------------------------------------------------------------------------
// LAS 1.4 format 7 raw byte size = 30 (base format 6) + 6 (RGB) = 36
// ---------------------------------------------------------------------------
const POINT_RECORD_LENGTH: u16 = 36;

/// Encode one point as LAS 1.4 format 7 raw bytes (36 bytes, little-endian).
fn encode_point_fmt7(rp: &RawPoint, buf: &mut Vec<u8>) {
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
    buf.extend_from_slice(&rp.red.to_le_bytes());
    buf.extend_from_slice(&rp.green.to_le_bytes());
    buf.extend_from_slice(&rp.blue.to_le_bytes());
    // Total = 4+4+4+2+1+1+1+1+2+2+8+2+2+2 = 36 bytes
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
    memory_budget: u64,
) -> Result<()> {
    let scale_x = builder.scale_x;
    let scale_y = builder.scale_y;
    let scale_z = builder.scale_z;
    let offset_x = builder.offset_x;
    let offset_y = builder.offset_y;
    let offset_z = builder.offset_z;

    // -----------------------------------------------------------------------
    // Build the LAZ VLR (variable-size chunks, format 7 items)
    // -----------------------------------------------------------------------
    let laz_vlr = LazVlrBuilder::default()
        .with_point_format(7, 0)
        .context("LazVlrBuilder for format 7")?
        .with_variable_chunk_size()
        .build();

    let mut laz_vlr_payload: Vec<u8> = Vec::new();
    laz_vlr.write_to(&mut laz_vlr_payload)?;

    // -----------------------------------------------------------------------
    // File layout constants
    // -----------------------------------------------------------------------
    let wkt_crs = &builder.wkt_crs;
    let copc_info_vlr_size: u32 = 54 + 160; // 214
    let laz_vlr_size: u32 = 54 + laz_vlr_payload.len() as u32; // 100
    let wkt_vlr_size: u32 = wkt_crs.as_ref().map(|d| 54 + d.len() as u32).unwrap_or(0);
    let num_vlrs: u32 = if wkt_crs.is_some() { 3 } else { 2 };
    let offset_to_point_data: u32 = 375 + copc_info_vlr_size + laz_vlr_size + wkt_vlr_size;

    let copc_info_payload_pos: u64 = 375 + 54;

    // Use the actual point count from node_keys (not builder.total_points which
    // is the original input count — the write-back sampling may have moved points).
    let actual_total_points: u64 = node_keys.iter().map(|(_, c)| *c as u64).sum();
    info!(
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

    info!(
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
    w.write_u8(128 | 7)?; // point format 135 = LAZ format 7
    w.write_u16::<LittleEndian>(POINT_RECORD_LENGTH)?;
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
    w.write_u32::<LittleEndian>(1)?; // number of EVLRs
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
        spacing: builder.halfsize / (1u64 << builder.depth) as f64,
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

    w.flush()?;

    // -----------------------------------------------------------------------
    // Parallel compression via ParLasZipCompressor
    // -----------------------------------------------------------------------
    let laz_vlr_for_compressor = LazVlrBuilder::default()
        .with_point_format(7, 0)
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

    info!(
        "Encoding {} data nodes ({} empty ancestors) in batches (budget {} MiB)",
        data_keys.len(),
        ordered_keys.len() - data_keys.len(),
        memory_budget / 1_048_576,
    );

    // Split data_keys into batches whose total uncompressed bytes fit within
    // memory_budget, then encode+compress each batch before moving to the next.
    // Also accumulate per-return point counts and GPS time extents.
    let mut return_counts = [0u64; 15];
    let mut gpstime_min = f64::MAX;
    let mut gpstime_max = f64::MIN;

    let mut batch_start = 0;
    while batch_start < data_keys.len() {
        let mut batch_bytes: u64 = 0;
        let mut batch_end = batch_start;
        while batch_end < data_keys.len() {
            let key = &data_keys[batch_end];
            let node_bytes =
                (point_counts.get(key).copied().unwrap_or(0) as u64) * POINT_RECORD_LENGTH as u64;
            // Always include at least one node per batch to avoid stalling.
            if batch_end > batch_start && batch_bytes + node_bytes > memory_budget {
                break;
            }
            batch_bytes += node_bytes;
            batch_end += 1;
        }

        let batch = &data_keys[batch_start..batch_end];
        // Each parallel task returns (encoded_bytes, local_return_counts, local_gps_min, local_gps_max).
        let results: Vec<(Vec<u8>, [u64; 15], f64, f64)> = batch
            .par_iter()
            .map(|key| -> Result<(Vec<u8>, [u64; 15], f64, f64)> {
                let mut pts = builder.read_node(key)?;
                pts.sort_unstable_by(|a, b| {
                    a.gps_time
                        .partial_cmp(&b.gps_time)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let mut local_returns = [0u64; 15];
                let mut local_gps_min = f64::MAX;
                let mut local_gps_max = f64::MIN;
                let mut raw_bytes = Vec::with_capacity(POINT_RECORD_LENGTH as usize * pts.len());
                for rp in &pts {
                    let rn = rp.return_number as usize;
                    if rn >= 1 && rn <= 15 {
                        local_returns[rn - 1] += 1;
                    }
                    if rp.gps_time < local_gps_min {
                        local_gps_min = rp.gps_time;
                    }
                    if rp.gps_time > local_gps_max {
                        local_gps_max = rp.gps_time;
                    }
                    encode_point_fmt7(rp, &mut raw_bytes);
                }
                Ok((raw_bytes, local_returns, local_gps_min, local_gps_max))
            })
            .collect::<Result<Vec<_>>>()?;

        let encoded: Vec<Vec<u8>> = results
            .into_iter()
            .map(|(bytes, local_returns, local_min, local_max)| {
                for i in 0..15 {
                    return_counts[i] += local_returns[i];
                }
                if local_min < gpstime_min {
                    gpstime_min = local_min;
                }
                if local_max > gpstime_max {
                    gpstime_max = local_max;
                }
                bytes
            })
            .collect();

        compressor
            .compress_chunks(encoded)
            .context("compress_chunks")?;

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
        .with_point_format(7, 0)
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
        info!(
            "ERROR: chunk table has {} entries but we compressed {} chunks!",
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
    // EVLR: copc hierarchy
    // -----------------------------------------------------------------------

    let mut hier_payload: Vec<u8> = Vec::with_capacity(chunk_info.len() * 32);
    for (key, offset, byte_size, point_count) in &chunk_info {
        HierarchyEntry {
            key: *key,
            offset: *offset,
            byte_size: *byte_size,
            point_count: *point_count,
        }
        .write(&mut hier_payload)?;
    }

    file.seek(SeekFrom::Start(evlr_start))?;
    let mut w = BufWriter::new(file);
    write_evlr(&mut w, "copc", 1000, "copc hierarchy", &hier_payload)?;
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
        spacing: builder.halfsize / (1u64 << builder.depth) as f64,
        root_hier_offset: evlr_start + EVLR_HEADER_SIZE as u64,
        root_hier_size: hier_payload.len() as u64,
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
