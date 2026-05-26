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

/// Parse every hierarchy entry reachable from the root page, following
/// page pointers (entries with `point_count = -1`). Returns only data
/// entries (page pointers are expanded in place).
fn read_hierarchy(data: &[u8]) -> Vec<HierarchyEntry> {
    let info = read_copc_info(data);
    let mut entries = Vec::new();
    read_hierarchy_page(
        data,
        info.root_hier_offset,
        info.root_hier_size,
        &mut entries,
    );
    entries
}

fn read_hierarchy_page(data: &[u8], offset: u64, size: u64, out: &mut Vec<HierarchyEntry>) {
    let start = offset as usize;
    let end = start + size as usize;
    let payload = &data[start..end];
    let mut r = Cursor::new(payload);
    while r.position() < size {
        let key = VoxelKey {
            level: r.read_i32::<LittleEndian>().unwrap(),
            x: r.read_i32::<LittleEndian>().unwrap(),
            y: r.read_i32::<LittleEndian>().unwrap(),
            z: r.read_i32::<LittleEndian>().unwrap(),
        };
        let entry_offset = r.read_u64::<LittleEndian>().unwrap();
        let byte_size = r.read_i32::<LittleEndian>().unwrap();
        let point_count = r.read_i32::<LittleEndian>().unwrap();
        if point_count == -1 {
            // Page pointer: recurse into the child page.
            read_hierarchy_page(data, entry_offset, byte_size as u64, out);
        } else {
            out.push(HierarchyEntry {
                key,
                offset: entry_offset,
                byte_size,
                point_count,
            });
        }
    }
}

#[derive(Debug)]
struct TemporalIndexHeader {
    version: u32,
    stride: u32,
    node_count: u32,
    page_count: u32,
    root_page_offset: u64,
    root_page_size: u32,
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
        page_count: r.read_u32::<LittleEndian>().unwrap(),
        root_page_offset: r.read_u64::<LittleEndian>().unwrap(),
        root_page_size: r.read_u32::<LittleEndian>().unwrap(),
    };
    let _reserved = r.read_u32::<LittleEndian>().unwrap();

    // Read all entries from all pages sequentially.
    // In the v2 layout, pages are written sequentially after the header.
    // We read all entries by scanning the remaining payload, distinguishing
    // node entries (sample_count >= 1) from page pointers (sample_count == 0).
    let mut entries = Vec::new();
    while (r.position() as usize) < payload.len() {
        let key = VoxelKey {
            level: r.read_i32::<LittleEndian>().unwrap(),
            x: r.read_i32::<LittleEndian>().unwrap(),
            y: r.read_i32::<LittleEndian>().unwrap(),
            z: r.read_i32::<LittleEndian>().unwrap(),
        };
        let sample_count = r.read_u32::<LittleEndian>().unwrap();
        if sample_count == 0 {
            // Page pointer: skip child_page_offset(8) + child_page_size(4) +
            // subtree_time_min(8) + subtree_time_max(8) = 28 bytes
            r.seek(SeekFrom::Current(28)).unwrap();
        } else {
            let mut samples = Vec::with_capacity(sample_count as usize);
            for _ in 0..sample_count {
                samples.push(r.read_f64::<LittleEndian>().unwrap());
            }
            entries.push(TemporalIndexNodeEntry { key, samples });
        }
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
        .args(["--progress", "plain"])
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
fn spacing_matches_copc_spec() {
    // COPC 1.0 spec: spacing is the inter-point distance at the root
    // node, halving with each level. Untwine writes this as
    // 2 * halfsize / CellCount with CellCount = 128. Verify our writer
    // matches both the spec formula and the untwine reference value.
    let output = Path::new("tests/data/test_copc_spacing.copc.laz");
    run_converter(Path::new("tests/data/input.laz"), output);

    let ours = read_file(output);
    let reference = read_file(Path::new("tests/data/untwine_reference.copc.laz"));

    let info_ours = read_copc_info(&ours);
    let info_ref = read_copc_info(&reference);

    let expected_ours = 2.0 * info_ours.halfsize / 128.0;
    assert!(
        (info_ours.spacing - expected_ours).abs() < 1e-9,
        "spacing should be 2*halfsize/128: got {}, expected {} (halfsize={})",
        info_ours.spacing,
        expected_ours,
        info_ours.halfsize
    );

    let expected_ref = 2.0 * info_ref.halfsize / 128.0;
    assert!(
        (info_ref.spacing - expected_ref).abs() < 1e-9,
        "untwine reference spacing should be 2*halfsize/128: got {}, expected {} \
         (halfsize={}) — sanity check on the reference fixture",
        info_ref.spacing,
        expected_ref,
        info_ref.halfsize
    );

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
        &["--temporal-index", "1000"],
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
        &["--temporal-index", "500"],
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
        &["--temporal-index", "100"],
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
        &["--temporal-index", "5000"],
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

// ---------------------------------------------------------------------------
// Low-memory / streaming path tests
// ---------------------------------------------------------------------------

/// Run with a tiny memory limit to exercise the build under tight memory
/// pressure. Verify the output is a valid COPC file with the correct
/// total point count.
#[test]
fn low_memory_produces_valid_output() {
    let output = Path::new("tests/data/test_low_mem.copc.laz");
    // 1 MB budget forces many small chunks, exercising the merge step
    // under tight memory pressure.
    run_converter_with_args(
        Path::new("tests/data/input.laz"),
        output,
        &["--memory-limit", "1M"],
    );

    let data = read_file(output);
    let header = read_las_header(&data);
    let reference = read_file(Path::new("tests/data/untwine_reference.copc.laz"));
    let ref_header = read_las_header(&reference);

    // Must preserve all points
    assert_eq!(
        header.total_points, ref_header.total_points,
        "total points must match reference"
    );

    // Hierarchy must be present and have a root node
    let hier = read_hierarchy(&data);
    let root = VoxelKey {
        level: 0,
        x: 0,
        y: 0,
        z: 0,
    };
    assert!(
        hier.iter().any(|e| e.key == root),
        "hierarchy must have root node"
    );

    // Total points across hierarchy must match header
    let hier_total: i64 = hier
        .iter()
        .filter(|e| e.point_count > 0)
        .map(|e| e.point_count as i64)
        .sum();
    assert_eq!(
        hier_total, header.total_points as i64,
        "hierarchy point sum must match header"
    );

    // Every data node must have valid offset and byte_size
    for entry in &hier {
        if entry.point_count > 0 {
            assert!(entry.offset > 0, "node {:?} must have offset", entry.key);
            assert!(
                entry.byte_size > 0,
                "node {:?} must have byte_size",
                entry.key
            );
        }
    }

    let _ = std::fs::remove_file(output);
}

/// Same end-to-end path as `low_memory_produces_valid_output`, but with
/// `--node-storage packed` so the pack-file backend runs through the full
/// build + merge + writer pipeline.
#[test]
fn packed_storage_produces_valid_output() {
    let output = Path::new("tests/data/test_packed_storage.copc.laz");
    run_converter_with_args(
        Path::new("tests/data/input.laz"),
        output,
        &["--memory-limit", "1M", "--node-storage", "packed"],
    );

    let data = read_file(output);
    let header = read_las_header(&data);
    let reference = read_file(Path::new("tests/data/untwine_reference.copc.laz"));
    let ref_header = read_las_header(&reference);

    assert_eq!(
        header.total_points, ref_header.total_points,
        "total points must match reference with packed storage"
    );

    let hier = read_hierarchy(&data);
    let hier_total: i64 = hier
        .iter()
        .filter(|e| e.point_count > 0)
        .map(|e| e.point_count as i64)
        .sum();
    assert_eq!(
        hier_total, header.total_points as i64,
        "hierarchy point sum must match header with packed storage"
    );

    let _ = std::fs::remove_file(output);
}

// ---------------------------------------------------------------------------
// copc-streaming / copc-temporal round-trip test
// ---------------------------------------------------------------------------

/// Verify that copc-streaming and copc-temporal can read the COPC output
/// produced by our converter (temporal index enabled). This exercises the
/// same code paths as the inspect_temporal tool.
#[cfg(feature = "tools")]
#[test]
fn temporal_index_readable_by_streaming_crate() {
    use copc_streaming::{CopcStreamingReader, FileSource};
    use copc_temporal::TemporalCache;

    let output = Path::new("tests/data/test_temporal_streaming.copc.laz");
    run_converter_with_args(
        Path::new("tests/data/input.laz"),
        output,
        &["--temporal-index", "1000"],
    );

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let source = FileSource::open(output.to_str().unwrap()).unwrap();
        let mut reader = CopcStreamingReader::open(source).await.unwrap();

        // Header should be readable — copy values before mutable borrow
        let point_format = reader.header().las_header().point_format().to_u8().unwrap();
        let num_points = reader.header().las_header().number_of_points();
        let halfsize = reader.header().copc_info().halfsize;
        let gpstime_min = reader.header().copc_info().gpstime_minimum;
        let gpstime_max = reader.header().copc_info().gpstime_maximum;

        assert_eq!(point_format, 6);
        assert!(num_points > 0);
        assert!(halfsize > 0.0);
        assert!(gpstime_min <= gpstime_max);

        // Hierarchy should load
        reader.load_all_hierarchy().await.unwrap();
        assert!(reader.node_count() > 0, "hierarchy must have nodes");

        let hier_data_nodes = reader.entries().filter(|(_, e)| e.point_count > 0).count();
        assert!(hier_data_nodes > 0, "must have data nodes");

        // Temporal index should load
        let temporal = TemporalCache::from_reader(&reader).await.unwrap();
        let mut temporal = temporal.expect("temporal index must be present");

        let th = temporal.header().unwrap();
        assert_eq!(th.version, 1);
        assert_eq!(th.stride, 1000);
        assert!(th.node_count > 0);

        // Load all temporal pages
        let source = FileSource::open(output.to_str().unwrap()).unwrap();
        temporal.load_all_pages(&source).await.unwrap();

        // Every temporal entry should have samples and a valid time range
        let mut temporal_count = 0;
        for (_key, entry) in temporal.iter() {
            temporal_count += 1;
            assert!(
                !entry.samples().is_empty(),
                "node must have at least one sample"
            );
            let (t_min, t_max) = entry.time_range();
            assert!(t_min.0 <= t_max.0, "time_min must be <= time_max");
            assert!(t_min.0 >= gpstime_min, "sample time must be >= global min");
            assert!(t_max.0 <= gpstime_max, "sample time must be <= global max");
        }

        // Temporal entries must match hierarchy data nodes
        assert_eq!(
            temporal_count, hier_data_nodes,
            "temporal entries must match hierarchy data nodes"
        );
    });

    let _ = std::fs::remove_file(output);
}

// ---------------------------------------------------------------------------
// Build tests
//
// Build output is NOT bit-identical to any fixed reference because
// grid_sample tie-breaking depends on point ordering at the leaf level.
// These tests verify that output is valid and point-conserving rather
// than comparing bytes.
// ---------------------------------------------------------------------------

#[test]
fn produces_valid_copc() {
    let output = Path::new("tests/data/test_chunked_basic.copc.laz");
    run_converter(Path::new("tests/data/input.laz"), output);

    let data = read_file(output);
    let header = read_las_header(&data);
    let reference = read_file(Path::new("tests/data/untwine_reference.copc.laz"));
    let ref_header = read_las_header(&reference);

    // Same point format and total point count as the reference.
    assert_eq!(header.point_format, ref_header.point_format, "point format");
    assert_eq!(
        header.total_points, ref_header.total_points,
        "total points must match reference"
    );

    // Same scale and offset (both derived from input file).
    assert_eq!(header.scale_x, ref_header.scale_x, "scale_x");
    assert_eq!(header.scale_y, ref_header.scale_y, "scale_y");
    assert_eq!(header.scale_z, ref_header.scale_z, "scale_z");
    assert_eq!(header.offset_x, ref_header.offset_x, "offset_x");
    assert_eq!(header.offset_y, ref_header.offset_y, "offset_y");
    assert_eq!(header.offset_z, ref_header.offset_z, "offset_z");

    // Bounds should be within tolerance of the reference.
    let tol = 0.01;
    assert!((header.min_x - ref_header.min_x).abs() < tol, "min_x");
    assert!((header.max_x - ref_header.max_x).abs() < tol, "max_x");

    // Must have at least one EVLR (the COPC hierarchy).
    assert!(header.num_evlrs >= 1, "must have at least 1 EVLR");

    let _ = std::fs::remove_file(output);
}

#[test]
fn preserves_all_points() {
    let output = Path::new("tests/data/test_chunked_conservation.copc.laz");
    run_converter(Path::new("tests/data/input.laz"), output);

    let data = read_file(output);
    let header = read_las_header(&data);
    let hier = read_hierarchy(&data);

    // Hierarchy must contain a root node at (0, 0, 0, 0).
    let root = VoxelKey {
        level: 0,
        x: 0,
        y: 0,
        z: 0,
    };
    assert!(
        hier.iter().any(|e| e.key == root),
        "chunked hierarchy must have root node"
    );

    // Sum of point counts across all data nodes must equal the header total.
    let hier_total: i64 = hier
        .iter()
        .filter(|e| e.point_count > 0)
        .map(|e| e.point_count as i64)
        .sum();
    assert_eq!(
        hier_total, header.total_points as i64,
        "chunked hierarchy point sum must match header"
    );

    // Every data node must have a valid offset and byte_size.
    for entry in &hier {
        if entry.point_count > 0 {
            assert!(
                entry.offset > 0,
                "chunked node {:?} must have offset",
                entry.key
            );
            assert!(
                entry.byte_size > 0,
                "chunked node {:?} must have byte_size",
                entry.key
            );
        }
    }

    let _ = std::fs::remove_file(output);
}

/// Regression test for a point-loss bug in an earlier revision of the
/// chunked distribute path.
///
/// An earlier version used inlined floor-based grid-cell math for speed,
/// while `build_chunk_in_memory` used `point_to_key` for leaf classification.
/// Floating-point precision at cell boundaries made the two disagree on a
/// tiny fraction of points, which then landed in the wrong chunk and were
/// silently lost. Fixed by using `point_to_key` in both places.
///
/// This test forces a multi-chunk plan on the small 830K-point test input
/// by passing `--chunk-target 100000` (hidden dev flag). That produces ~20
/// chunks at varying levels, which exercises the merge step AND the
/// classification boundary between chunks. Total point count must be
/// conserved exactly.
#[test]
fn chunked_multi_chunk_preserves_all_points() {
    let output = Path::new("tests/data/test_chunked_multichunk.copc.laz");
    let input = Path::new("tests/data/input.laz");

    run_converter_with_args(
        input,
        output,
        // --chunk-target 100000 forces ~8-21 chunks on the 830K-point input
        // (well below the dynamic 1M minimum). This exercises the merge step
        // and the cross-chunk classification boundary — the exact conditions
        // that triggered the original point-loss regression.
        &["--chunk-target", "100000"],
    );

    let data = read_file(output);
    let header = read_las_header(&data);
    let hier = read_hierarchy(&data);

    // Sum of hierarchy point counts must EXACTLY equal the LAS header total
    // (no tolerance — point loss is always a bug).
    let hier_total: u64 = hier
        .iter()
        .filter(|e| e.point_count > 0)
        .map(|e| e.point_count as u64)
        .sum();
    assert_eq!(
        hier_total, header.total_points,
        "chunked hierarchy point sum ({}) must equal header total ({})",
        hier_total, header.total_points
    );

    // The header total must also match the reference input's total point
    // count. The untwine reference file is what we validate against.
    let reference = read_file(Path::new("tests/data/untwine_reference.copc.laz"));
    let ref_header = read_las_header(&reference);
    assert_eq!(
        header.total_points, ref_header.total_points,
        "chunked total points ({}) must match reference input ({})",
        header.total_points, ref_header.total_points
    );

    // Sanity: this test only means anything if we actually got multiple
    // chunks. Verify by checking that the hierarchy has nodes at levels
    // below the max (indicating a multi-level tree formed by the merge step).
    let max_level = hier.iter().map(|e| e.key.level).max().unwrap_or(0);
    assert!(
        max_level >= 3,
        "chunked_multi_chunk test must produce a multi-level tree \
         (got max_level={}), otherwise the test isn't exercising what it's \
         supposed to exercise",
        max_level
    );

    let _ = std::fs::remove_file(output);
}

/// Run with `--temp-compression=lz4` and verify the pipeline still conserves
/// every point. Exercises `write_temp_batch` + `read_temp_batches` under LZ4.
#[test]
fn temp_compression_lz4_preserves_all_points() {
    let output = Path::new("tests/data/test_lz4_conservation.copc.laz");
    run_converter_with_args(
        Path::new("tests/data/input.laz"),
        output,
        &["--temp-compression", "lz4"],
    );

    let data = read_file(output);
    let header = read_las_header(&data);
    let hier = read_hierarchy(&data);

    // Sum of point counts across all data nodes must equal the header total.
    let hier_total: i64 = hier
        .iter()
        .filter(|e| e.point_count > 0)
        .map(|e| e.point_count as i64)
        .sum();
    assert_eq!(
        hier_total, header.total_points as i64,
        "lz4 hierarchy point sum must match header"
    );

    // Must match the reference input's total point count.
    let reference = read_file(Path::new("tests/data/untwine_reference.copc.laz"));
    let ref_header = read_las_header(&reference);
    assert_eq!(
        header.total_points, ref_header.total_points,
        "lz4 total points must match reference input"
    );

    let _ = std::fs::remove_file(output);
}

/// Run multi-chunk merge under LZ4. Exercises multi-frame append: each
/// `ChunkWriterCache::append` call writes a self-contained LZ4 frame into
/// the shard file, and the reader must walk multiple concatenated frames
/// to recover every point.
#[test]
fn temp_compression_lz4_multi_chunk_preserves_all_points() {
    let output = Path::new("tests/data/test_lz4_multichunk.copc.laz");
    run_converter_with_args(
        Path::new("tests/data/input.laz"),
        output,
        &["--temp-compression", "lz4", "--chunk-target", "100000"],
    );

    let data = read_file(output);
    let header = read_las_header(&data);
    let hier = read_hierarchy(&data);

    let hier_total: u64 = hier
        .iter()
        .filter(|e| e.point_count > 0)
        .map(|e| e.point_count as u64)
        .sum();
    assert_eq!(
        hier_total, header.total_points,
        "lz4 multi-chunk hierarchy point sum ({}) must equal header total ({})",
        hier_total, header.total_points
    );

    // Sanity: verify we actually got a multi-level tree from the merge step.
    let max_level = hier.iter().map(|e| e.key.level).max().unwrap_or(0);
    assert!(
        max_level >= 3,
        "lz4 multi-chunk test must produce a multi-level tree (got max_level={})",
        max_level
    );

    let _ = std::fs::remove_file(output);
}

// ---------------------------------------------------------------------------
// LAS Extra Bytes pass-through
// ---------------------------------------------------------------------------
//
// `tests/data/extra_bytes_input.laz` is a LAS 1.4 point-format-3 file with a
// LASF_Spec/4 Extra Bytes VLR declaring four trailing fields (treeId,
// semantic, bisemantic, zNorm), totalling 12 bytes per point. The
// converter must:
//   1. Preserve the VLR byte-for-byte in the output.
//   2. Advertise a point record length that includes the 12 extra bytes.
//   3. Carry the per-point extra bytes through to the LAZ-compressed
//      payload so readers can decode the same field values.

const EXTRA_BYTES_VLR_SIZE: usize = 4 * 192; // 4 entries × 192 bytes

#[test]
fn extra_bytes_vlr_preserved() {
    let input = Path::new("tests/data/extra_bytes_input.laz");
    let output = Path::new("tests/data/test_extras_vlr.copc.laz");
    run_converter(input, output);

    let input_bytes = read_file(input);
    let output_bytes = read_file(output);

    let in_vlr = find_vlr(&input_bytes, "LASF_Spec", 4).expect("input must have Extra Bytes VLR");
    let out_vlr =
        find_vlr(&output_bytes, "LASF_Spec", 4).expect("output must preserve Extra Bytes VLR");
    assert_eq!(
        in_vlr.len(),
        EXTRA_BYTES_VLR_SIZE,
        "input VLR should be 4 × 192 bytes"
    );
    assert_eq!(
        out_vlr, in_vlr,
        "output Extra Bytes VLR must be byte-identical to input"
    );

    let _ = std::fs::remove_file(output);
}

#[test]
fn extra_bytes_header_record_length() {
    let output = Path::new("tests/data/test_extras_record_len.copc.laz");
    run_converter(Path::new("tests/data/extra_bytes_input.laz"), output);

    let bytes = read_file(output);
    let header = read_las_header(&bytes);
    // Input is point format 3 → COPC promotes it to format 7. Format-7 base
    // is 36 bytes; the 12 trailing extras land on top of that.
    assert_eq!(
        header.point_format, 7,
        "format-3 input should be promoted to format 7"
    );
    assert_eq!(
        header.point_record_len, 48,
        "header record length must include extras (36 base + 12 extras)"
    );

    let _ = std::fs::remove_file(output);
}

#[test]
fn extra_bytes_per_point_payload_round_trips() {
    let input = Path::new("tests/data/extra_bytes_input.laz");
    let output = Path::new("tests/data/test_extras_per_point.copc.laz");
    run_converter(input, output);

    // Read input extras stats: compute per-field min/max across all input
    // points, then assert the output covers the same value range (the
    // converter does not drop points; thinning only redistributes them
    // across LOD nodes — every input point still appears somewhere).
    let mut in_reader = las::Reader::from_path(input).expect("open input");
    let mut in_pts: Vec<las::Point> = Vec::new();
    in_reader
        .read_points_into(u64::MAX, &mut in_pts)
        .expect("read input points");
    assert!(!in_pts.is_empty(), "input must have points");
    assert!(
        in_pts.iter().all(|p| p.extra_bytes.len() == 12),
        "every input point should carry 12 extras"
    );

    // Per-byte max across all input points. We don't depend on which
    // logical field maps to which byte offset (the Extra Bytes VLR
    // declares that). We only need to know which bytes carry non-zero
    // values so the assertions catch a regression where the converter
    // writes zeros instead of pass-through bytes.
    let n_extras = in_pts[0].extra_bytes.len();
    let in_byte_max: Vec<u8> = (0..n_extras)
        .map(|i| in_pts.iter().map(|p| p.extra_bytes[i]).max().unwrap_or(0))
        .collect();
    assert!(
        in_byte_max.iter().any(|m| *m > 0),
        "input extras must have at least one non-zero byte position \
         (test would be vacuous otherwise)"
    );

    let mut out_reader = las::Reader::from_path(output).expect("open output");
    let mut out_pts: Vec<las::Point> = Vec::new();
    out_reader
        .read_points_into(u64::MAX, &mut out_pts)
        .expect("read output points");

    assert!(!out_pts.is_empty(), "output must have points");
    assert!(
        out_pts.iter().all(|p| p.extra_bytes.len() == 12),
        "every output point must carry 12 extras"
    );

    let out_byte_max: Vec<u8> = (0..n_extras)
        .map(|i| out_pts.iter().map(|p| p.extra_bytes[i]).max().unwrap_or(0))
        .collect();

    // The converter does not drop points; LOD thinning only redistributes
    // them across nodes. So every byte-position max must round-trip exactly.
    assert_eq!(
        in_byte_max, out_byte_max,
        "per-byte max of extras must round-trip exactly"
    );

    // Every point's extras must match a corresponding input point exactly.
    // We can't rely on order since LOD thinning reorders points, so we
    // sample a few points by (x, y, z) coordinate. The chance of two
    // distinct input points sharing all three scaled-int coordinates is
    // vanishingly small for a real lidar dataset.
    use std::collections::HashMap;
    let in_lookup: HashMap<(i32, i32, i32), &[u8]> = in_pts
        .iter()
        .map(|p| {
            // las::Point coordinates are already in scaled f64 — quantise
            // back to ints for stable hashing.
            let key = (
                (p.x * 1000.0).round() as i32,
                (p.y * 1000.0).round() as i32,
                (p.z * 1000.0).round() as i32,
            );
            (key, p.extra_bytes.as_slice())
        })
        .collect();

    let mut matched = 0usize;
    let mut checked = 0usize;
    for out_p in out_pts.iter().take(5000) {
        let key = (
            (out_p.x * 1000.0).round() as i32,
            (out_p.y * 1000.0).round() as i32,
            (out_p.z * 1000.0).round() as i32,
        );
        if let Some(in_extras) = in_lookup.get(&key) {
            checked += 1;
            if &out_p.extra_bytes[..] == *in_extras {
                matched += 1;
            }
        }
    }
    assert!(
        checked >= 100,
        "should find at least 100 output points whose coordinates match an input point \
         (got {checked}); coordinate quantisation may have changed"
    );
    assert_eq!(
        matched, checked,
        "every matched-coordinate point must have identical extras (matched={matched}/{checked})"
    );

    let _ = std::fs::remove_file(output);
}

/// Two real input files declare structurally incompatible Extra Bytes
/// schemas (different field counts, different `treeId` data type). The
/// converter must surface this via `Error::ExtraBytesMismatch` with a
/// `detail` enumerating every difference — and must reference the
/// canonical input as `file_a`, not the offending one.
///
/// Source files: tests/data/extra_bytes_input.laz (4 fields:
/// treeId u16, semantic u8, bisemantic u8, zNorm f64) and
/// tests/data/extra_bytes_mismatched.laz (2 fields: treeId u32, zNorm
/// f64) — both are real LAS 1.4 tiles from the same upstream pipeline.
#[test]
fn extra_bytes_schema_mismatch_reports_all_differences() {
    let canonical = Path::new("tests/data/extra_bytes_input.laz");
    let mismatched = Path::new("tests/data/extra_bytes_mismatched.laz");

    // Build a temp input directory holding both fixtures so the
    // converter scans them as a multi-file batch.
    let tmp_dir = std::env::temp_dir().join(format!("copc_extras_mismatch_{}", std::process::id()));
    std::fs::create_dir_all(&tmp_dir).unwrap();
    let canonical_in = tmp_dir.join("a_canonical.laz");
    let mismatched_in = tmp_dir.join("b_mismatched.laz");
    std::fs::copy(canonical, &canonical_in).unwrap();
    std::fs::copy(mismatched, &mismatched_in).unwrap();

    let output = tmp_dir.join("out.copc.laz");
    let result = Command::new(converter_bin())
        .arg(&tmp_dir)
        .arg(&output)
        .args(["--progress", "plain"])
        .output()
        .expect("failed to run copc_converter");

    let _ = std::fs::remove_dir_all(&tmp_dir);

    assert!(
        !result.status.success(),
        "converter should fail when input files have mismatched extras schemas"
    );

    let stderr = String::from_utf8_lossy(&result.stderr);
    let combined = format!("{stderr}\n{}", String::from_utf8_lossy(&result.stdout));

    // The two failure modes that must appear in the detail block:
    //   - field count differs (4 vs 2, includes the missing field names)
    //   - per-field data_type / option / etc. diffs not relevant here
    //     because field count short-circuits the per-field walk.
    assert!(
        combined.contains("Extra Bytes schema mismatch"),
        "expected ExtraBytesMismatch error header in output, got:\n{combined}"
    );
    assert!(
        combined.contains("field count differs"),
        "expected 'field count differs' line in detail, got:\n{combined}"
    );
    // The 4-field canonical names must appear so the user can see what
    // they're missing in the offending file.
    assert!(
        combined.contains("semantic") && combined.contains("bisemantic"),
        "expected canonical field names listed in error, got:\n{combined}"
    );
    // The reference file should be the canonical one (first scanned
    // with extras), not the offender.
    let canonical_idx = combined.find("a_canonical.laz");
    let mismatched_idx = combined.find("b_mismatched.laz");
    assert!(
        canonical_idx.is_some() && mismatched_idx.is_some(),
        "both file paths must appear in the error, got:\n{combined}"
    );
    assert!(
        canonical_idx.unwrap() < mismatched_idx.unwrap(),
        "canonical file must be named first (as the reference), got:\n{combined}"
    );
}
