//! Integration tests comparing our COPC output against an untwine reference.

#![allow(dead_code)]

use byteorder::{LittleEndian, ReadBytesExt};
use std::collections::HashMap;
use std::io::{Cursor, Read, Seek, SeekFrom};
use std::path::Path;
use std::process::Command;

// ---------------------------------------------------------------------------
// Parsed COPC structures (read-only, for test assertions)
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct LasHeader {
    point_format: u8,
    point_record_len: u16,
    offset_to_point_data: u32,
    num_vlrs: u32,
    total_points: u64,
    scale_x: f64,
    scale_y: f64,
    scale_z: f64,
    offset_x: f64,
    offset_y: f64,
    offset_z: f64,
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
    min_z: f64,
    max_z: f64,
    evlr_start: u64,
    num_evlrs: u32,
}

#[derive(Debug)]
struct CopcInfo {
    center_x: f64,
    center_y: f64,
    center_z: f64,
    halfsize: f64,
    spacing: f64,
    root_hier_offset: u64,
    root_hier_size: u64,
    gpstime_min: f64,
    gpstime_max: f64,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
struct VoxelKey {
    level: i32,
    x: i32,
    y: i32,
    z: i32,
}

#[derive(Debug, Clone)]
struct HierarchyEntry {
    key: VoxelKey,
    offset: u64,
    byte_size: i32,
    point_count: i32,
}

// ---------------------------------------------------------------------------
// Parsing helpers
// ---------------------------------------------------------------------------

fn read_las_header(data: &[u8]) -> LasHeader {
    let mut r = Cursor::new(data);

    // Use absolute offsets per LAS 1.4 spec
    r.seek(SeekFrom::Start(94)).unwrap(); // offset 94: header size
    let _header_size = r.read_u16::<LittleEndian>().unwrap();
    let offset_to_point_data = r.read_u32::<LittleEndian>().unwrap(); // 96
    let num_vlrs = r.read_u32::<LittleEndian>().unwrap(); // 100
    let point_format_raw = r.read_u8().unwrap(); // 104
    let point_format = point_format_raw & 0x3F; // strip compression bit
    let point_record_len = r.read_u16::<LittleEndian>().unwrap(); // 105

    // offset 107: legacy point count (4) + legacy return counts (5*4=20)
    r.seek(SeekFrom::Start(131)).unwrap();
    let scale_x = r.read_f64::<LittleEndian>().unwrap(); // 131
    let scale_y = r.read_f64::<LittleEndian>().unwrap();
    let scale_z = r.read_f64::<LittleEndian>().unwrap();
    let offset_x = r.read_f64::<LittleEndian>().unwrap();
    let offset_y = r.read_f64::<LittleEndian>().unwrap();
    let offset_z = r.read_f64::<LittleEndian>().unwrap();
    let max_x = r.read_f64::<LittleEndian>().unwrap(); // 179
    let min_x = r.read_f64::<LittleEndian>().unwrap();
    let max_y = r.read_f64::<LittleEndian>().unwrap();
    let min_y = r.read_f64::<LittleEndian>().unwrap();
    let max_z = r.read_f64::<LittleEndian>().unwrap();
    let min_z = r.read_f64::<LittleEndian>().unwrap();

    r.seek(SeekFrom::Start(227)).unwrap(); // waveform data packet record (8)
    let _waveform = r.read_u64::<LittleEndian>().unwrap();
    let evlr_start = r.read_u64::<LittleEndian>().unwrap(); // 235
    let num_evlrs = r.read_u32::<LittleEndian>().unwrap(); // 243
    let total_points = r.read_u64::<LittleEndian>().unwrap(); // 247

    LasHeader {
        point_format,
        point_record_len,
        offset_to_point_data,
        num_vlrs,
        total_points,
        scale_x,
        scale_y,
        scale_z,
        offset_x,
        offset_y,
        offset_z,
        min_x,
        max_x,
        min_y,
        max_y,
        min_z,
        max_z,
        evlr_start,
        num_evlrs,
    }
}

fn find_vlr(data: &[u8], target_user_id: &str, target_record_id: u16) -> Option<Vec<u8>> {
    let header = read_las_header(data);
    let mut pos = 375u64; // VLRs start after the 375-byte header

    for _ in 0..header.num_vlrs {
        let mut r = Cursor::new(data);
        r.seek(SeekFrom::Start(pos)).unwrap();
        let _reserved = r.read_u16::<LittleEndian>().unwrap();
        let mut uid = [0u8; 16];
        r.read_exact(&mut uid).unwrap();
        let record_id = r.read_u16::<LittleEndian>().unwrap();
        let payload_len = r.read_u16::<LittleEndian>().unwrap() as usize;
        // skip description (32)
        r.seek(SeekFrom::Current(32)).unwrap();

        let user_id = std::str::from_utf8(&uid)
            .unwrap_or("")
            .trim_end_matches('\0');

        if user_id == target_user_id && record_id == target_record_id {
            let offset = r.position() as usize;
            return Some(data[offset..offset + payload_len].to_vec());
        }

        pos += 54 + payload_len as u64;
    }
    None
}

fn read_copc_info(data: &[u8]) -> CopcInfo {
    let payload = find_vlr(data, "copc", 1).expect("copc info VLR not found");
    let mut r = Cursor::new(&payload);
    CopcInfo {
        center_x: r.read_f64::<LittleEndian>().unwrap(),
        center_y: r.read_f64::<LittleEndian>().unwrap(),
        center_z: r.read_f64::<LittleEndian>().unwrap(),
        halfsize: r.read_f64::<LittleEndian>().unwrap(),
        spacing: r.read_f64::<LittleEndian>().unwrap(),
        root_hier_offset: r.read_u64::<LittleEndian>().unwrap(),
        root_hier_size: r.read_u64::<LittleEndian>().unwrap(),
        gpstime_min: r.read_f64::<LittleEndian>().unwrap(),
        gpstime_max: r.read_f64::<LittleEndian>().unwrap(),
    }
}

fn read_hierarchy(data: &[u8]) -> Vec<HierarchyEntry> {
    let info = read_copc_info(data);
    let offset = info.root_hier_offset as usize;
    let size = info.root_hier_size as usize;
    let payload = &data[offset..offset + size];
    let mut r = Cursor::new(payload);
    let mut entries = Vec::new();

    while r.position() < size as u64 {
        let key = VoxelKey {
            level: r.read_i32::<LittleEndian>().unwrap(),
            x: r.read_i32::<LittleEndian>().unwrap(),
            y: r.read_i32::<LittleEndian>().unwrap(),
            z: r.read_i32::<LittleEndian>().unwrap(),
        };
        let entry_offset = r.read_u64::<LittleEndian>().unwrap();
        let byte_size = r.read_i32::<LittleEndian>().unwrap();
        let point_count = r.read_i32::<LittleEndian>().unwrap();
        entries.push(HierarchyEntry {
            key,
            offset: entry_offset,
            byte_size,
            point_count,
        });
    }
    entries
}

#[derive(Debug)]
struct TemporalIndexHeader {
    version: u32,
    stride: u32,
    node_count: u32,
}

#[derive(Debug)]
struct TemporalIndexNodeEntry {
    key: VoxelKey,
    samples: Vec<f64>,
}

fn find_evlr(data: &[u8], target_user_id: &str, target_record_id: u16) -> Option<Vec<u8>> {
    let header = read_las_header(data);
    let mut pos = header.evlr_start;

    for _ in 0..header.num_evlrs {
        let mut r = Cursor::new(data);
        r.seek(SeekFrom::Start(pos)).unwrap();
        let _reserved = r.read_u16::<LittleEndian>().unwrap();
        let mut uid = [0u8; 16];
        r.read_exact(&mut uid).unwrap();
        let record_id = r.read_u16::<LittleEndian>().unwrap();
        let payload_len = r.read_u64::<LittleEndian>().unwrap();
        // skip description (32)
        r.seek(SeekFrom::Current(32)).unwrap();

        let user_id = std::str::from_utf8(&uid)
            .unwrap_or("")
            .trim_end_matches('\0');

        if user_id == target_user_id && record_id == target_record_id {
            let offset = r.position() as usize;
            return Some(data[offset..offset + payload_len as usize].to_vec());
        }

        pos += 60 + payload_len; // EVLR header = 60 bytes
    }
    None
}

fn read_temporal_index(data: &[u8]) -> Option<(TemporalIndexHeader, Vec<TemporalIndexNodeEntry>)> {
    let payload = find_evlr(data, "copc_temporal", 1000)?;
    let mut r = Cursor::new(&payload);

    let header = TemporalIndexHeader {
        version: r.read_u32::<LittleEndian>().unwrap(),
        stride: r.read_u32::<LittleEndian>().unwrap(),
        node_count: r.read_u32::<LittleEndian>().unwrap(),
    };
    let _reserved = r.read_u32::<LittleEndian>().unwrap();

    let mut entries = Vec::new();
    for _ in 0..header.node_count {
        let key = VoxelKey {
            level: r.read_i32::<LittleEndian>().unwrap(),
            x: r.read_i32::<LittleEndian>().unwrap(),
            y: r.read_i32::<LittleEndian>().unwrap(),
            z: r.read_i32::<LittleEndian>().unwrap(),
        };
        let sample_count = r.read_u32::<LittleEndian>().unwrap();
        let mut samples = Vec::with_capacity(sample_count as usize);
        for _ in 0..sample_count {
            samples.push(r.read_f64::<LittleEndian>().unwrap());
        }
        entries.push(TemporalIndexNodeEntry { key, samples });
    }

    Some((header, entries))
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

fn converter_bin() -> std::path::PathBuf {
    // cargo test builds the binary in the same target dir
    let mut path = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    path.push("copc_converter");
    path
}

fn run_converter(input: &Path, output: &Path) {
    run_converter_with_args(input, output, &[]);
}

fn run_converter_with_args(input: &Path, output: &Path, extra_args: &[&str]) {
    let status = Command::new(converter_bin())
        .arg(input)
        .arg(output)
        .args(extra_args)
        .status()
        .expect("failed to run copc_converter");
    assert!(status.success(), "converter exited with error");
}

fn read_file(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| panic!("cannot read {}: {}", path.display(), e))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn header_matches_reference() {
    let output = Path::new("tests/data/test_output.copc.laz");
    run_converter(Path::new("tests/data/input.laz"), output);

    let ours = read_file(output);
    let reference = read_file(Path::new("tests/data/untwine_reference.copc.laz"));

    let h_ours = read_las_header(&ours);
    let h_ref = read_las_header(&reference);

    // Same point format and record length
    assert_eq!(h_ours.point_format, h_ref.point_format, "point format");
    assert_eq!(
        h_ours.point_record_len, h_ref.point_record_len,
        "point record length"
    );

    // Same total point count
    assert_eq!(h_ours.total_points, h_ref.total_points, "total points");

    // Bounds should be very close (different octree constructions may snap differently)
    let tol = 0.01;
    assert!((h_ours.min_x - h_ref.min_x).abs() < tol, "min_x");
    assert!((h_ours.max_x - h_ref.max_x).abs() < tol, "max_x");
    assert!((h_ours.min_y - h_ref.min_y).abs() < tol, "min_y");
    assert!((h_ours.max_y - h_ref.max_y).abs() < tol, "max_y");
    assert!((h_ours.min_z - h_ref.min_z).abs() < tol, "min_z");
    assert!((h_ours.max_z - h_ref.max_z).abs() < tol, "max_z");

    // Scale and offset should match (both derived from input)
    assert_eq!(h_ours.scale_x, h_ref.scale_x, "scale_x");
    assert_eq!(h_ours.scale_y, h_ref.scale_y, "scale_y");
    assert_eq!(h_ours.scale_z, h_ref.scale_z, "scale_z");
    assert_eq!(h_ours.offset_x, h_ref.offset_x, "offset_x");
    assert_eq!(h_ours.offset_y, h_ref.offset_y, "offset_y");
    assert_eq!(h_ours.offset_z, h_ref.offset_z, "offset_z");

    // Should have at least 1 EVLR (hierarchy)
    assert!(h_ours.num_evlrs >= 1, "must have at least 1 EVLR");

    // Clean up
    let _ = std::fs::remove_file(output);
}

#[test]
fn copc_info_matches_reference() {
    let output = Path::new("tests/data/test_copc_info.copc.laz");
    run_converter(Path::new("tests/data/input.laz"), output);

    let ours = read_file(output);
    let reference = read_file(Path::new("tests/data/untwine_reference.copc.laz"));

    let info_ours = read_copc_info(&ours);
    let info_ref = read_copc_info(&reference);

    // Octree center and halfsize may differ slightly due to different construction,
    // but should enclose the same bounds.
    // Check that our octree root at least contains the reference bounds.
    let h_ref = read_las_header(&reference);
    assert!(
        info_ours.center_x - info_ours.halfsize <= h_ref.min_x + 0.01,
        "octree must contain min_x"
    );
    assert!(
        info_ours.center_x + info_ours.halfsize >= h_ref.max_x - 0.01,
        "octree must contain max_x"
    );
    assert!(
        info_ours.center_y - info_ours.halfsize <= h_ref.min_y + 0.01,
        "octree must contain min_y"
    );
    assert!(
        info_ours.center_y + info_ours.halfsize >= h_ref.max_y - 0.01,
        "octree must contain max_y"
    );

    // GPS time range should match
    let tol = 0.001;
    assert!(
        (info_ours.gpstime_min - info_ref.gpstime_min).abs() < tol,
        "gpstime_min: ours={} ref={}",
        info_ours.gpstime_min,
        info_ref.gpstime_min
    );
    assert!(
        (info_ours.gpstime_max - info_ref.gpstime_max).abs() < tol,
        "gpstime_max: ours={} ref={}",
        info_ours.gpstime_max,
        info_ref.gpstime_max
    );

    // Hierarchy must be reachable
    assert!(
        info_ours.root_hier_offset > 0,
        "hierarchy offset must be set"
    );
    assert!(info_ours.root_hier_size > 0, "hierarchy size must be set");

    let _ = std::fs::remove_file(output);
}

#[test]
fn hierarchy_preserves_all_points() {
    let output = Path::new("tests/data/test_hierarchy.copc.laz");
    run_converter(Path::new("tests/data/input.laz"), output);

    let ours = read_file(output);
    let reference = read_file(Path::new("tests/data/untwine_reference.copc.laz"));

    let hier_ours = read_hierarchy(&ours);
    let hier_ref = read_hierarchy(&reference);

    // Total points across all hierarchy entries must match
    let total_ours: i64 = hier_ours
        .iter()
        .filter(|e| e.point_count > 0)
        .map(|e| e.point_count as i64)
        .sum();
    let total_ref: i64 = hier_ref
        .iter()
        .filter(|e| e.point_count > 0)
        .map(|e| e.point_count as i64)
        .sum();
    assert_eq!(total_ours, total_ref, "total points in hierarchy");

    // Both should have a root node (0,0,0,0)
    let root = VoxelKey {
        level: 0,
        x: 0,
        y: 0,
        z: 0,
    };
    assert!(
        hier_ours.iter().any(|e| e.key == root),
        "our hierarchy must have root node"
    );
    assert!(
        hier_ref.iter().any(|e| e.key == root),
        "reference hierarchy must have root node"
    );

    // Every node with points must have a valid offset and byte_size
    for entry in &hier_ours {
        if entry.point_count > 0 {
            assert!(entry.offset > 0, "node {:?} must have offset", entry.key);
            assert!(
                entry.byte_size > 0,
                "node {:?} must have byte_size",
                entry.key
            );
        }
    }

    // Max depth should be similar (within 1 level)
    let max_level_ours = hier_ours.iter().map(|e| e.key.level).max().unwrap();
    let max_level_ref = hier_ref.iter().map(|e| e.key.level).max().unwrap();
    assert!(
        (max_level_ours - max_level_ref).abs() <= 1,
        "max depth should be similar: ours={} ref={}",
        max_level_ours,
        max_level_ref
    );

    let _ = std::fs::remove_file(output);
}

#[test]
fn hierarchy_structure_similar_to_reference() {
    let output = Path::new("tests/data/test_coverage.copc.laz");
    run_converter(Path::new("tests/data/input.laz"), output);

    let ours = read_file(output);
    let reference = read_file(Path::new("tests/data/untwine_reference.copc.laz"));

    let hier_ours = read_hierarchy(&ours);
    let hier_ref = read_hierarchy(&reference);

    // Build point count maps per level
    let level_points = |hier: &[HierarchyEntry]| -> HashMap<i32, i64> {
        let mut map: HashMap<i32, i64> = HashMap::new();
        for e in hier {
            if e.point_count > 0 {
                *map.entry(e.key.level).or_default() += e.point_count as i64;
            }
        }
        map
    };

    let ours_by_level = level_points(&hier_ours);
    let ref_by_level = level_points(&hier_ref);

    // Both should have root-level points
    assert!(
        ours_by_level.contains_key(&0),
        "our hierarchy has no root-level points"
    );
    assert!(
        ref_by_level.contains_key(&0),
        "reference hierarchy has no root-level points"
    );

    // Total points across all levels must match
    let total_ours: i64 = ours_by_level.values().sum();
    let total_ref: i64 = ref_by_level.values().sum();
    assert_eq!(
        total_ours, total_ref,
        "total points across all levels must match"
    );

    // Per-level point distribution should be similar.
    // Different octree builders may distribute points differently across LODs,
    // so we allow each level to differ by up to 20% of the total points.
    let tolerance = (total_ref as f64 * 0.20) as i64;
    for (&level, &ref_count) in &ref_by_level {
        let our_count = ours_by_level.get(&level).copied().unwrap_or(0);
        let diff = (our_count - ref_count).abs();
        assert!(
            diff <= tolerance,
            "level {} point count differs too much: ours={} ref={} diff={} tolerance={}",
            level,
            our_count,
            ref_count,
            diff,
            tolerance,
        );
    }

    // Both should produce a similar number of data nodes.
    // Different octree strategies may subdivide differently, so allow 3x ratio.
    let our_data_nodes = hier_ours.iter().filter(|e| e.point_count > 0).count();
    let ref_data_nodes = hier_ref.iter().filter(|e| e.point_count > 0).count();
    let ratio =
        our_data_nodes.max(ref_data_nodes) as f64 / our_data_nodes.min(ref_data_nodes) as f64;
    assert!(
        ratio < 3.0,
        "node count ratio too high: ours={} ref={} ratio={:.1}",
        our_data_nodes,
        ref_data_nodes,
        ratio,
    );

    let _ = std::fs::remove_file(output);
}

#[test]
fn deterministic_output() {
    // Two runs should produce equivalent output.
    // LAZ parallel compression may introduce minor byte-level differences,
    // so we compare logical content rather than raw bytes.
    let output1 = Path::new("tests/data/test_deterministic_1.copc.laz");
    let output2 = Path::new("tests/data/test_deterministic_2.copc.laz");
    let input = Path::new("tests/data/input.laz");

    run_converter(input, output1);
    run_converter(input, output2);

    let data1 = read_file(output1);
    let data2 = read_file(output2);

    let h1 = read_las_header(&data1);
    let h2 = read_las_header(&data2);

    assert_eq!(h1.total_points, h2.total_points, "point count");
    assert_eq!(h1.point_format, h2.point_format, "point format");
    assert_eq!(h1.min_x, h2.min_x, "min_x");
    assert_eq!(h1.max_x, h2.max_x, "max_x");

    // Hierarchy should have the same nodes with the same point counts
    let hier1 = read_hierarchy(&data1);
    let hier2 = read_hierarchy(&data2);
    assert_eq!(hier1.len(), hier2.len(), "hierarchy node count");

    let map1: HashMap<_, _> = hier1
        .iter()
        .map(|e| (e.key.clone(), e.point_count))
        .collect();
    for e in &hier2 {
        let count1 = map1.get(&e.key).expect("node missing in run 1");
        assert_eq!(
            *count1, e.point_count,
            "point count differs for {:?}",
            e.key
        );
    }

    let _ = std::fs::remove_file(output1);
    let _ = std::fs::remove_file(output2);
}

// ---------------------------------------------------------------------------
// Temporal index tests
// ---------------------------------------------------------------------------

#[test]
fn temporal_index_absent_by_default() {
    let output = Path::new("tests/data/test_no_temporal.copc.laz");
    run_converter(Path::new("tests/data/input.laz"), output);

    let data = read_file(output);
    let header = read_las_header(&data);

    assert_eq!(header.num_evlrs, 1, "should have only hierarchy EVLR");
    assert!(
        read_temporal_index(&data).is_none(),
        "temporal index should not be present without --temporal-index"
    );

    let _ = std::fs::remove_file(output);
}

#[test]
fn temporal_index_present_when_enabled() {
    let output = Path::new("tests/data/test_temporal.copc.laz");
    run_converter_with_args(
        Path::new("tests/data/input.laz"),
        output,
        &["--temporal-index"],
    );

    let data = read_file(output);
    let header = read_las_header(&data);

    assert_eq!(
        header.num_evlrs, 2,
        "should have hierarchy + temporal EVLRs"
    );

    let (ti_header, ti_entries) =
        read_temporal_index(&data).expect("temporal index EVLR must be present");

    assert_eq!(ti_header.version, 1);
    assert_eq!(ti_header.stride, 1000, "default stride");
    assert!(ti_header.node_count > 0, "must have at least one node");
    assert_eq!(
        ti_entries.len(),
        ti_header.node_count as usize,
        "entry count must match header"
    );

    // Every entry must have at least one sample
    for entry in &ti_entries {
        assert!(
            !entry.samples.is_empty(),
            "node {:?} has no samples",
            entry.key
        );
    }

    // The temporal index should cover exactly the data nodes from the hierarchy
    let hierarchy = read_hierarchy(&data);
    let data_node_count = hierarchy.iter().filter(|e| e.point_count > 0).count();
    assert_eq!(
        ti_entries.len(),
        data_node_count,
        "temporal index must have one entry per data node"
    );

    let _ = std::fs::remove_file(output);
}

#[test]
fn temporal_index_samples_are_sorted() {
    let output = Path::new("tests/data/test_temporal_sorted.copc.laz");
    run_converter_with_args(
        Path::new("tests/data/input.laz"),
        output,
        &["--temporal-index", "--temporal-stride", "500"],
    );

    let data = read_file(output);
    let (ti_header, ti_entries) =
        read_temporal_index(&data).expect("temporal index must be present");

    assert_eq!(ti_header.stride, 500, "stride should match CLI arg");

    // GPS time range from COPC info
    let copc_info = read_copc_info(&data);

    for entry in &ti_entries {
        // Samples must be monotonically non-decreasing
        for w in entry.samples.windows(2) {
            assert!(
                w[0] <= w[1],
                "samples not sorted in node {:?}: {} > {}",
                entry.key,
                w[0],
                w[1]
            );
        }

        // First and last sample must be within the global GPS time range
        let first = entry.samples[0];
        let last = *entry.samples.last().unwrap();
        assert!(
            first >= copc_info.gpstime_min,
            "node {:?} first sample {} < global min {}",
            entry.key,
            first,
            copc_info.gpstime_min
        );
        assert!(
            last <= copc_info.gpstime_max,
            "node {:?} last sample {} > global max {}",
            entry.key,
            last,
            copc_info.gpstime_max
        );
    }

    let _ = std::fs::remove_file(output);
}

#[test]
fn temporal_index_custom_stride() {
    let output = Path::new("tests/data/test_temporal_stride.copc.laz");
    run_converter_with_args(
        Path::new("tests/data/input.laz"),
        output,
        &["--temporal-index", "--temporal-stride", "100"],
    );

    let data = read_file(output);
    let (header_s100, entries_s100) =
        read_temporal_index(&data).expect("temporal index must be present");
    assert_eq!(header_s100.stride, 100);

    let _ = std::fs::remove_file(output);

    // Run again with larger stride
    let output2 = Path::new("tests/data/test_temporal_stride2.copc.laz");
    run_converter_with_args(
        Path::new("tests/data/input.laz"),
        output2,
        &["--temporal-index", "--temporal-stride", "5000"],
    );

    let data2 = read_file(output2);
    let (header_s5000, entries_s5000) =
        read_temporal_index(&data2).expect("temporal index must be present");
    assert_eq!(header_s5000.stride, 5000);

    // Smaller stride should produce more samples per node
    let total_samples_s100: usize = entries_s100.iter().map(|e| e.samples.len()).sum();
    let total_samples_s5000: usize = entries_s5000.iter().map(|e| e.samples.len()).sum();
    assert!(
        total_samples_s100 > total_samples_s5000,
        "stride 100 should produce more samples than stride 5000: {} vs {}",
        total_samples_s100,
        total_samples_s5000,
    );

    let _ = std::fs::remove_file(output2);
}
