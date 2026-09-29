//! Structural COPC 1.0 / LAS 1.4 validator for integration tests.
//!
//! Parses a COPC file independently of the converter's own code, decodes
//! every point chunk from its hierarchy offset, and reports every deviation
//! from the COPC 1.0 and LAS 1.4 R15 specs (plus a few invariants this
//! converter guarantees, marked as such). Returns a list of human-readable
//! issues rather than panicking, so tests can also assert that deliberately
//! corrupted files are caught.

pub mod rechunk;

use byteorder::{LittleEndian as LE, ReadBytesExt};
use laz::LazVlr;
use laz::record::{LayeredPointRecordDecompressor, RecordDecompressor};
use std::collections::HashSet;
use std::io::{Cursor, Seek, SeekFrom};
use std::path::Path;

const HEADER_SIZE: usize = 375;
const VLR_HEADER_SIZE: usize = 54;
const EVLR_HEADER_SIZE: usize = 60;
const COPC_INFO_SIZE: usize = 160;
const HIERARCHY_ENTRY_SIZE: u64 = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Key {
    level: i32,
    x: i32,
    y: i32,
    z: i32,
}

impl Key {
    fn parent(self) -> Option<Key> {
        (self.level > 0).then(|| Key {
            level: self.level - 1,
            x: self.x >> 1,
            y: self.y >> 1,
            z: self.z >> 1,
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct Entry {
    key: Key,
    offset: u64,
    byte_size: i32,
    point_count: i32,
}

/// A node's bounds as copc.js (the reference reader, behind copc-validator)
/// computes them: from the cube `center ± halfsize`, split at
/// `min + (max − min) / 2` once per level along the key's path.
fn node_bounds(center: [f64; 3], halfsize: f64, key: Key) -> ([f64; 3], [f64; 3]) {
    let mut min = center.map(|c| c - halfsize);
    let mut max = center.map(|c| c + halfsize);
    let coords = [key.x, key.y, key.z];
    for level in (0..key.level).rev() {
        for a in 0..3 {
            let mid = min[a] + (max[a] - min[a]) / 2.0;
            if (coords[a] >> level) & 1 == 1 {
                min[a] = mid;
            } else {
                max[a] = mid;
            }
        }
    }
    (min, max)
}

fn cstr(b: &[u8]) -> String {
    String::from_utf8_lossy(b)
        .trim_end_matches('\0')
        .to_string()
}

/// Panic with every issue found if `path` is not a valid COPC file.
pub fn assert_valid_copc(path: &Path) {
    let issues = validate_copc(path);
    assert!(
        issues.is_empty(),
        "{} is not a valid COPC file:\n  - {}",
        path.display(),
        issues.join("\n  - ")
    );
}

/// Validate the COPC file at `path`, returning every issue found.
pub fn validate_copc(path: &Path) -> Vec<String> {
    let data = std::fs::read(path).unwrap_or_else(|e| panic!("cannot read {path:?}: {e}"));
    validate_copc_bytes(&data)
}

/// Validate an in-memory COPC file, returning every issue found.
pub fn validate_copc_bytes(data: &[u8]) -> Vec<String> {
    let mut issues = Vec::new();
    if let Err(e) = check(data, &mut issues) {
        issues.push(format!("could not parse file: {e}"));
    }
    issues
}

fn check(data: &[u8], issues: &mut Vec<String>) -> std::io::Result<()> {
    // ---- LAS 1.4 header ----------------------------------------------------
    if data.len() < HEADER_SIZE || &data[0..4] != b"LASF" {
        issues.push("missing LASF signature or truncated header".into());
        return Ok(());
    }
    let mut c = Cursor::new(data);
    c.seek(SeekFrom::Start(6))?;
    let global_encoding = c.read_u16::<LE>()?;
    c.seek(SeekFrom::Start(24))?;
    let (version_major, version_minor) = (c.read_u8()?, c.read_u8()?);
    c.seek(SeekFrom::Start(94))?;
    let header_size = c.read_u16::<LE>()?;
    let offset_to_point_data = c.read_u32::<LE>()? as u64;
    let num_vlrs = c.read_u32::<LE>()?;
    let format_raw = c.read_u8()?;
    let record_len = c.read_u16::<LE>()?;
    let legacy_total = c.read_u32::<LE>()?;
    let mut legacy_by_return = [0u32; 5];
    for v in &mut legacy_by_return {
        *v = c.read_u32::<LE>()?;
    }
    let mut scale = [0f64; 3];
    for v in &mut scale {
        *v = c.read_f64::<LE>()?;
    }
    let mut offset = [0f64; 3];
    for v in &mut offset {
        *v = c.read_f64::<LE>()?;
    }
    let (mut hdr_max, mut hdr_min) = ([0f64; 3], [0f64; 3]);
    for a in 0..3 {
        hdr_max[a] = c.read_f64::<LE>()?;
        hdr_min[a] = c.read_f64::<LE>()?;
    }
    let _waveform = c.read_u64::<LE>()?;
    let evlr_start = c.read_u64::<LE>()?;
    let num_evlrs = c.read_u32::<LE>()?;
    let total_points = c.read_u64::<LE>()?;
    let mut by_return = [0u64; 15];
    for v in &mut by_return {
        *v = c.read_u64::<LE>()?;
    }
    let format = format_raw & 0x3f;

    if (version_major, version_minor) != (1, 4) {
        issues.push(format!(
            "LAS version {version_major}.{version_minor}, COPC requires 1.4"
        ));
    }
    if header_size as usize != HEADER_SIZE {
        issues.push(format!("header size {header_size}, expected {HEADER_SIZE}"));
    }
    if ![6, 7, 8].contains(&format) {
        issues.push(format!(
            "point data record format {format}, COPC requires 6, 7 or 8"
        ));
    }
    if format_raw & 0x80 == 0 {
        issues.push("point format byte lacks the LAZ compression bit (0x80)".into());
    }
    if global_encoding & 0x10 == 0 {
        issues.push("global encoding WKT bit (4) not set; required for formats 6+".into());
    }
    if legacy_total != 0 || legacy_by_return.iter().any(|&v| v != 0) {
        issues.push("legacy point counts must be zero for formats 6+ (LAS 1.4 R15)".into());
    }
    if by_return.iter().sum::<u64>() != total_points {
        issues.push(format!(
            "points by return sum to {}, header total is {total_points}",
            by_return.iter().sum::<u64>()
        ));
    }

    // ---- VLRs --------------------------------------------------------------
    let mut pos = HEADER_SIZE;
    let mut copc_info: Option<&[u8]> = None;
    let mut laz_vlr: Option<LazVlr> = None;
    for i in 0..num_vlrs {
        if pos + VLR_HEADER_SIZE > data.len() {
            issues.push(format!("VLR {i} header runs past end of file"));
            return Ok(());
        }
        let user_id = cstr(&data[pos + 2..pos + 18]);
        let record_id = u16::from_le_bytes([data[pos + 18], data[pos + 19]]);
        let len = u16::from_le_bytes([data[pos + 20], data[pos + 21]]) as usize;
        let payload = &data[pos + VLR_HEADER_SIZE..pos + VLR_HEADER_SIZE + len];
        let is_copc_info = user_id == "copc" && record_id == 1;
        if i == 0 && !(is_copc_info && len == COPC_INFO_SIZE) {
            issues.push("first VLR must be the 160-byte copc info VLR (copc/1)".into());
        }
        if is_copc_info {
            copc_info = Some(payload);
        }
        if user_id == "laszip encoded" && record_id == 22204 {
            match LazVlr::read_from(payload) {
                Ok(v) => laz_vlr = Some(v),
                Err(e) => issues.push(format!("unreadable LAZ VLR: {e}")),
            }
        }
        if user_id == "LASF_Projection" && matches!(record_id, 34735..=34737) {
            issues.push("GeoTIFF CRS VLR present; formats 6+ must use WKT".into());
        }
        pos += VLR_HEADER_SIZE + len;
    }
    if pos as u64 != offset_to_point_data {
        issues.push(format!(
            "VLRs end at {pos}, offset to point data is {offset_to_point_data}"
        ));
    }
    let Some(info) = copc_info else {
        issues.push("no copc info VLR".into());
        return Ok(());
    };
    let Some(laz_vlr) = laz_vlr else {
        issues.push("no LAZ VLR (laszip encoded/22204)".into());
        return Ok(());
    };
    if laz_vlr.chunk_size() != u32::MAX {
        issues.push("LAZ chunk size must be variable (u32::MAX)".into());
    }
    let item_bytes: u64 = laz_vlr.items().iter().map(|it| it.size() as u64).sum();
    if item_bytes != record_len as u64 {
        issues.push(format!(
            "LAZ items total {item_bytes} bytes, point record length is {record_len}"
        ));
    }

    // ---- COPC info ---------------------------------------------------------
    let mut ic = Cursor::new(info);
    let center = [
        ic.read_f64::<LE>()?,
        ic.read_f64::<LE>()?,
        ic.read_f64::<LE>()?,
    ];
    let halfsize = ic.read_f64::<LE>()?;
    let _spacing = ic.read_f64::<LE>()?;
    let root_page_offset = ic.read_u64::<LE>()?;
    let root_page_size = ic.read_u64::<LE>()?;
    let gps_min = ic.read_f64::<LE>()?;
    let gps_max = ic.read_f64::<LE>()?;
    for _ in 0..11 {
        if ic.read_u64::<LE>()? != 0 {
            issues.push("copc info reserved fields must be zero".into());
            break;
        }
    }
    if !(halfsize > 0.0 && halfsize.is_finite()) {
        issues.push(format!(
            "copc info halfsize {halfsize} is not positive and finite"
        ));
    }
    for a in 0..3 {
        if hdr_min[a] < center[a] - halfsize || hdr_max[a] > center[a] + halfsize {
            issues.push(format!(
                "header bounds on axis {a} extend outside the octree cube"
            ));
        }
    }

    // ---- EVLRs -------------------------------------------------------------
    let mut p = evlr_start as usize;
    let mut hierarchy: Option<(u64, u64)> = None;
    for i in 0..num_evlrs {
        if p + EVLR_HEADER_SIZE > data.len() {
            issues.push(format!("EVLR {i} header runs past end of file"));
            return Ok(());
        }
        let user_id = cstr(&data[p + 2..p + 18]);
        let record_id = u16::from_le_bytes([data[p + 18], data[p + 19]]);
        let len = u64::from_le_bytes(data[p + 20..p + 28].try_into().unwrap());
        if user_id == "copc" && record_id == 1000 {
            hierarchy = Some(((p + EVLR_HEADER_SIZE) as u64, len));
        }
        p += EVLR_HEADER_SIZE + len as usize;
    }
    if p != data.len() {
        issues.push(format!("EVLRs end at {p}, file is {} bytes", data.len()));
    }
    let Some((hier_start, hier_len)) = hierarchy else {
        issues.push("no copc hierarchy EVLR (copc/1000)".into());
        return Ok(());
    };
    let hier_end = hier_start + hier_len;
    if root_page_size == 0 {
        issues.push("root hierarchy page is empty; readers need at least the root entry".into());
    }
    if root_page_offset < hier_start || root_page_offset + root_page_size > hier_end {
        issues.push("root hierarchy page lies outside the hierarchy EVLR".into());
        return Ok(());
    }

    // ---- Hierarchy pages ---------------------------------------------------
    let mut entries: Vec<Entry> = Vec::new();
    let mut pages: Vec<(u64, u64)> = Vec::new();
    let mut page_pointers: Vec<(Key, u64, u64)> = Vec::new();
    let mut stack = vec![(root_page_offset, root_page_size)];
    while let Some((page_off, page_size)) = stack.pop() {
        if page_size % HIERARCHY_ENTRY_SIZE != 0 {
            issues.push(format!(
                "hierarchy page size {page_size} is not a multiple of 32"
            ));
            continue;
        }
        if page_off < hier_start || page_off + page_size > hier_end {
            issues.push(format!(
                "hierarchy page {page_off}+{page_size} lies outside the EVLR"
            ));
            continue;
        }
        pages.push((page_off, page_size));
        let mut pc = Cursor::new(&data[page_off as usize..(page_off + page_size) as usize]);
        for _ in 0..page_size / HIERARCHY_ENTRY_SIZE {
            let key = Key {
                level: pc.read_i32::<LE>()?,
                x: pc.read_i32::<LE>()?,
                y: pc.read_i32::<LE>()?,
                z: pc.read_i32::<LE>()?,
            };
            let e = Entry {
                key,
                offset: pc.read_u64::<LE>()?,
                byte_size: pc.read_i32::<LE>()?,
                point_count: pc.read_i32::<LE>()?,
            };
            if e.point_count == -1 {
                page_pointers.push((key, e.offset, e.byte_size as u64));
                stack.push((e.offset, e.byte_size as u64));
            } else {
                entries.push(e);
            }
        }
    }
    pages.sort_unstable();
    if pages.windows(2).any(|w| w[0].0 + w[0].1 > w[1].0) {
        issues.push("hierarchy pages overlap".into());
    }

    let keys: HashSet<Key> = entries.iter().map(|e| e.key).collect();
    if keys.len() != entries.len() {
        issues.push("duplicate hierarchy keys".into());
    }
    let root = Key {
        level: 0,
        x: 0,
        y: 0,
        z: 0,
    };
    if !keys.contains(&root) {
        issues.push("hierarchy has no root entry (0-0-0-0)".into());
    }
    for (key, _, _) in &page_pointers {
        if !keys.contains(key) {
            issues.push(format!(
                "page pointer {key:?} has no entry in its child page"
            ));
        }
    }
    let point_counts: std::collections::HashMap<Key, i32> =
        entries.iter().map(|e| (e.key, e.point_count)).collect();
    for e in &entries {
        let span = 1i64 << e.key.level.clamp(0, 62);
        if e.key.level < 0
            || [e.key.x, e.key.y, e.key.z]
                .iter()
                .any(|&v| v < 0 || v as i64 >= span)
        {
            issues.push(format!("key {:?} is outside its level's grid", e.key));
        }
        if e.point_count < -1 {
            issues.push(format!("{:?} has point count {}", e.key, e.point_count));
        }
        if e.point_count == 0 && (e.offset != 0 || e.byte_size != 0) {
            issues.push(format!(
                "zero-point entry {:?} must have offset and byte size 0",
                e.key
            ));
        }
        if e.point_count > 0 && e.byte_size <= 0 {
            issues.push(format!(
                "{:?} has points but byte size {}",
                e.key, e.byte_size
            ));
        }
        // Every ancestor must be present so readers can reach the node.
        // Converter invariant (not spec): ancestors of a data node hold
        // points too — an empty interior node renders as a sparse hole.
        let mut k = e.key;
        while let Some(parent) = k.parent() {
            match point_counts.get(&parent) {
                None => {
                    issues.push(format!("{:?} is missing ancestor {parent:?}", e.key));
                    break;
                }
                Some(0) if e.point_count > 0 => {
                    issues.push(format!(
                        "interior node {parent:?} is empty above {:?}",
                        e.key
                    ));
                    break;
                }
                _ => {}
            }
            k = parent;
        }
    }

    let mut data_nodes: Vec<Entry> = entries
        .iter()
        .filter(|e| e.point_count > 0)
        .copied()
        .collect();
    data_nodes.sort_unstable_by_key(|e| e.offset);
    let hier_total: u64 = data_nodes.iter().map(|e| e.point_count as u64).sum();
    if hier_total != total_points {
        issues.push(format!(
            "hierarchy point counts sum to {hier_total}, header total is {total_points}"
        ));
    }

    // ---- LAZ chunk table ---------------------------------------------------
    let point_data = offset_to_point_data as usize;
    let chunk_table_offset =
        i64::from_le_bytes(data[point_data..point_data + 8].try_into().unwrap()) as u64;
    let mut cc = Cursor::new(data);
    cc.seek(SeekFrom::Start(offset_to_point_data))?;
    let chunk_table = match laz::laszip::ChunkTable::read_from(&mut cc, &laz_vlr) {
        Ok(t) => t,
        Err(e) => {
            issues.push(format!("unreadable LAZ chunk table: {e}"));
            return Ok(());
        }
    };
    if chunk_table.len() != data_nodes.len() {
        issues.push(format!(
            "chunk table has {} entries, hierarchy has {} data nodes",
            chunk_table.len(),
            data_nodes.len()
        ));
    }
    let mut expected_offset = offset_to_point_data + 8;
    for (i, e) in data_nodes.iter().enumerate() {
        if e.offset != expected_offset {
            issues.push(format!(
                "chunk {i} ({:?}) starts at {}, expected {expected_offset} (chunks must be contiguous)",
                e.key, e.offset
            ));
            return Ok(());
        }
        if i < chunk_table.len() {
            let ct = &chunk_table[i];
            if ct.byte_count != e.byte_size as u64 || ct.point_count != e.point_count as u64 {
                issues.push(format!(
                    "chunk {i} ({:?}): chunk table says {} bytes / {} points, hierarchy says {} / {}",
                    e.key, ct.byte_count, ct.point_count, e.byte_size, e.point_count
                ));
            }
        }
        expected_offset += e.byte_size as u64;
    }
    if expected_offset != chunk_table_offset {
        issues.push(format!(
            "chunks end at {expected_offset}, chunk table is at {chunk_table_offset}"
        ));
    }

    // ---- Point data --------------------------------------------------------
    // Decode each chunk on its own, from its hierarchy offset, as a reader
    // fetching a single node would.
    let mut record = vec![0u8; record_len as usize];
    let mut decoded_by_return = [0u64; 15];
    let (mut decoded_min, mut decoded_max) = ([f64::MAX; 3], [f64::MIN; 3]);
    let (mut decoded_gps_min, mut decoded_gps_max) = (f64::MAX, f64::MIN);
    let mut outside_node = 0u64;
    let mut first_outside: Option<String> = None;
    let mut unsorted_nodes: Vec<Key> = Vec::new();
    for e in &data_nodes {
        let chunk = &data[e.offset as usize..(e.offset + e.byte_size as u64) as usize];
        let mut decompressor = LayeredPointRecordDecompressor::new(Cursor::new(chunk));
        if let Err(err) = decompressor.set_fields_from(laz_vlr.items()) {
            issues.push(format!("cannot set up decoder for {:?}: {err}", e.key));
            return Ok(());
        }
        let (node_min, node_max) = node_bounds(center, halfsize, e.key);
        let mut last_gps = f64::MIN;
        for _ in 0..e.point_count {
            if let Err(err) = decompressor.decompress_next(&mut record) {
                issues.push(format!("chunk {:?} failed to decode: {err}", e.key));
                return Ok(());
            }
            let raw = [
                i32::from_le_bytes(record[0..4].try_into().unwrap()),
                i32::from_le_bytes(record[4..8].try_into().unwrap()),
                i32::from_le_bytes(record[8..12].try_into().unwrap()),
            ];
            let return_number = record[14] & 0x0f;
            let gps_time = f64::from_le_bytes(record[22..30].try_into().unwrap());
            if (1..=15).contains(&return_number) {
                decoded_by_return[return_number as usize - 1] += 1;
            }
            // Converter invariant (not spec): points within a node are
            // sorted by GPS time, which the temporal index relies on.
            if gps_time < last_gps && unsorted_nodes.last() != Some(&e.key) {
                unsorted_nodes.push(e.key);
            }
            last_gps = gps_time;
            decoded_gps_min = decoded_gps_min.min(gps_time);
            decoded_gps_max = decoded_gps_max.max(gps_time);
            for a in 0..3 {
                let w = raw[a] as f64 * scale[a] + offset[a];
                decoded_min[a] = decoded_min[a].min(w);
                decoded_max[a] = decoded_max[a].max(w);
                if w < node_min[a] || w > node_max[a] {
                    outside_node += 1;
                    first_outside.get_or_insert_with(|| {
                        format!(
                            "{:?} axis {a}: {w} not in [{}, {}]",
                            e.key, node_min[a], node_max[a]
                        )
                    });
                }
            }
        }
    }
    if !unsorted_nodes.is_empty() {
        issues.push(format!(
            "{} nodes are not sorted by GPS time (first: {:?})",
            unsorted_nodes.len(),
            unsorted_nodes[0]
        ));
    }
    if outside_node > 0 {
        issues.push(format!(
            "{outside_node} point coordinates lie outside their node's bounds (first: {})",
            first_outside.unwrap_or_default()
        ));
    }
    if total_points > 0 {
        for a in 0..3 {
            // LAS 1.4: the header min/max are the actual extents. They are
            // encoded like the points, so the extreme points match exactly;
            // allow a hair of float slack in how either was computed.
            let slack = scale[a] * 1e-6;
            if (decoded_min[a] - hdr_min[a]).abs() > slack
                || (decoded_max[a] - hdr_max[a]).abs() > slack
            {
                issues.push(format!(
                    "header bounds on axis {a} are [{}, {}], but the points span [{}, {}]",
                    hdr_min[a], hdr_max[a], decoded_min[a], decoded_max[a]
                ));
            }
        }
        if (decoded_gps_min, decoded_gps_max) != (gps_min, gps_max) {
            issues.push(format!(
                "copc info GPS range [{gps_min}, {gps_max}], decoded [{decoded_gps_min}, {decoded_gps_max}]"
            ));
        }
    } else if (gps_min, gps_max) != (0.0, 0.0) {
        issues.push(format!(
            "empty file declares GPS range [{gps_min}, {gps_max}]"
        ));
    }
    if decoded_by_return != by_return {
        issues.push(format!(
            "header points by return {by_return:?}, decoded {decoded_by_return:?}"
        ));
    }
    Ok(())
}
