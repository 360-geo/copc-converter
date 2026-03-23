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
    CopcInfo, EVLR_HEADER_SIZE, HierarchyEntry, TemporalIndexEntry, TemporalIndexHeader, VoxelKey,
    write_evlr, write_vlr,
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
// Point record sizes: format 6 = 30, format 7 = 36, format 8 = 38
// ---------------------------------------------------------------------------
fn point_record_length(fmt: u8) -> u16 {
    match fmt {
        6 => 30,
        7 => 36,
        8 => 38,
        _ => 36,
    }
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

/// Encode one point according to the COPC output format (6, 7, or 8).
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
    let point_record_len = point_record_length(point_format);

    // -----------------------------------------------------------------------
    // Build the LAZ VLR (variable-size chunks)
    // -----------------------------------------------------------------------
    let laz_vlr = LazVlrBuilder::default()
        .with_point_format(point_format, 0)
        .context("LazVlrBuilder for format")?
        .with_variable_chunk_size()
        .build();

    let mut laz_vlr_payload: Vec<u8> = Vec::new();
    laz_vlr.write_to(&mut laz_vlr_payload)?;

    // -----------------------------------------------------------------------
    // File layout constants
    // -----------------------------------------------------------------------
    let wkt_crs = &builder.wkt_crs;
    let copc_info_vlr_size: u32 = 54 + 160; // 214
    let laz_vlr_size: u32 = 54 + laz_vlr_payload.len() as u32;
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
    let num_evlrs: u32 = if config.temporal_index { 2 } else { 1 };
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
        .with_point_format(point_format, 0)
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
    let mut temporal_entries: Vec<TemporalIndexEntry> = Vec::new();
    let temporal_index = config.temporal_index;
    let temporal_stride = config.temporal_stride as usize;

    let mut batch_start = 0;
    while batch_start < data_keys.len() {
        let mut batch_bytes: u64 = 0;
        let mut batch_end = batch_start;
        while batch_end < data_keys.len() {
            let key = &data_keys[batch_end];
            let node_bytes =
                (point_counts.get(key).copied().unwrap_or(0) as u64) * point_record_len as u64;
            // Always include at least one node per batch to avoid stalling.
            if batch_end > batch_start && batch_bytes + node_bytes > memory_budget {
                break;
            }
            batch_bytes += node_bytes;
            batch_end += 1;
        }

        let batch = &data_keys[batch_start..batch_end];
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
                    if temporal_index && (i % temporal_stride == 0 || i == pts.len() - 1) {
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

        let encoded: Vec<Vec<u8>> = results
            .into_iter()
            .enumerate()
            .map(
                |(i, (bytes, local_returns, local_min, local_max, samples))| {
                    for j in 0..15 {
                        return_counts[j] += local_returns[j];
                    }
                    if local_min < gpstime_min {
                        gpstime_min = local_min;
                    }
                    if local_max > gpstime_max {
                        gpstime_max = local_max;
                    }
                    if temporal_index {
                        temporal_entries.push(TemporalIndexEntry {
                            key: batch[i],
                            samples,
                        });
                    }
                    bytes
                },
            )
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
        .with_point_format(point_format, 0)
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

    // -----------------------------------------------------------------------
    // EVLR: temporal index (optional)
    // -----------------------------------------------------------------------
    if config.temporal_index {
        let mut temporal_payload = Vec::new();
        TemporalIndexHeader {
            version: 1,
            stride: config.temporal_stride,
            node_count: temporal_entries.len() as u32,
        }
        .write(&mut temporal_payload)?;
        for entry in &temporal_entries {
            entry.write(&mut temporal_payload)?;
        }
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
        }
    }

    #[test]
    fn point_record_lengths() {
        assert_eq!(point_record_length(6), 30);
        assert_eq!(point_record_length(7), 36);
        assert_eq!(point_record_length(8), 38);
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
}
