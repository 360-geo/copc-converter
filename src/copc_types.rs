/// COPC data structures following the COPC 1.0 specification.
///
/// A COPC file is a LAZ 1.4 file with:
///  - A LAS 1.4 header
///  - A "copc info" VLR  (user_id = "copc", record_id = 1)
///  - A "copc hierarchy" EVLR (user_id = "copc", record_id = 1000)
///  - Point data stored in LAZ chunks, one chunk per octree node
use byteorder::{LittleEndian, WriteBytesExt};
use std::io::Write;

// ---------------------------------------------------------------------------
// VoxelKey – identifies a node in the octree
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VoxelKey {
    pub level: i32,
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl VoxelKey {
    /// Parent key (None for root).
    pub fn parent(self) -> Option<VoxelKey> {
        if self.level == 0 {
            return None;
        }
        Some(VoxelKey {
            level: self.level - 1,
            x: self.x / 2,
            y: self.y / 2,
            z: self.z / 2,
        })
    }
}

// ---------------------------------------------------------------------------
// CopcInfo VLR payload (160 bytes)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct CopcInfo {
    /// Center of the octree root (world coordinates)
    pub center_x: f64,
    pub center_y: f64,
    pub center_z: f64,
    /// Half-size of the root voxel (cube)
    pub halfsize: f64,
    /// Point spacing at the root level (spacing between points)
    pub spacing: f64,
    /// Byte offset of the root hierarchy page in the file
    pub root_hier_offset: u64,
    /// Byte size of the root hierarchy page
    pub root_hier_size: u64,
    /// GPS time of the first/last points (0 if not applicable)
    pub gpstime_minimum: f64,
    pub gpstime_maximum: f64,
    // 11 reserved u64 words
}

impl CopcInfo {
    pub fn write<W: Write>(&self, w: &mut W) -> anyhow::Result<()> {
        w.write_f64::<LittleEndian>(self.center_x)?;
        w.write_f64::<LittleEndian>(self.center_y)?;
        w.write_f64::<LittleEndian>(self.center_z)?;
        w.write_f64::<LittleEndian>(self.halfsize)?;
        w.write_f64::<LittleEndian>(self.spacing)?;
        w.write_u64::<LittleEndian>(self.root_hier_offset)?;
        w.write_u64::<LittleEndian>(self.root_hier_size)?;
        w.write_f64::<LittleEndian>(self.gpstime_minimum)?;
        w.write_f64::<LittleEndian>(self.gpstime_maximum)?;
        // 11 reserved u64 words → 88 bytes
        for _ in 0..11 {
            w.write_u64::<LittleEndian>(0)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// HierarchyEntry – one record in the hierarchy page (32 bytes)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct HierarchyEntry {
    pub key: VoxelKey,
    /// Byte offset of the chunk in the file (0 if virtual / no data)
    pub offset: u64,
    /// Byte size of the compressed chunk (-1 if no data, positive otherwise)
    pub byte_size: i32,
    /// Number of points in the chunk (-1 for hierarchy-only node)
    pub point_count: i32,
}

impl HierarchyEntry {
    pub fn write<W: Write>(&self, w: &mut W) -> anyhow::Result<()> {
        w.write_i32::<LittleEndian>(self.key.level)?;
        w.write_i32::<LittleEndian>(self.key.x)?;
        w.write_i32::<LittleEndian>(self.key.y)?;
        w.write_i32::<LittleEndian>(self.key.z)?;
        w.write_u64::<LittleEndian>(self.offset)?;
        w.write_i32::<LittleEndian>(self.byte_size)?;
        w.write_i32::<LittleEndian>(self.point_count)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// VLR / EVLR helpers
// ---------------------------------------------------------------------------

/// Write a LAS 1.4 VLR header + payload.
pub fn write_vlr<W: Write>(
    w: &mut W,
    user_id: &str,
    record_id: u16,
    description: &str,
    payload: &[u8],
) -> anyhow::Result<()> {
    // reserved (2 bytes)
    w.write_u16::<LittleEndian>(0)?;
    // user_id (16 bytes, null-padded)
    let mut uid = [0u8; 16];
    let b = user_id.as_bytes();
    uid[..b.len().min(16)].copy_from_slice(&b[..b.len().min(16)]);
    w.write_all(&uid)?;
    // record_id
    w.write_u16::<LittleEndian>(record_id)?;
    // record_length_after_header
    w.write_u16::<LittleEndian>(payload.len() as u16)?;
    // description (32 bytes, null-padded)
    let mut desc = [0u8; 32];
    let db = description.as_bytes();
    desc[..db.len().min(32)].copy_from_slice(&db[..db.len().min(32)]);
    w.write_all(&desc)?;
    // payload
    w.write_all(payload)?;
    Ok(())
}

/// Write a LAS 1.4 EVLR header + payload.
pub fn write_evlr<W: Write>(
    w: &mut W,
    user_id: &str,
    record_id: u16,
    description: &str,
    payload: &[u8],
) -> anyhow::Result<()> {
    // reserved (2 bytes)
    w.write_u16::<LittleEndian>(0)?;
    // user_id (16 bytes)
    let mut uid = [0u8; 16];
    let b = user_id.as_bytes();
    uid[..b.len().min(16)].copy_from_slice(&b[..b.len().min(16)]);
    w.write_all(&uid)?;
    // record_id
    w.write_u16::<LittleEndian>(record_id)?;
    // record_length_after_header (u64!)
    w.write_u64::<LittleEndian>(payload.len() as u64)?;
    // description (32 bytes)
    let mut desc = [0u8; 32];
    let db = description.as_bytes();
    desc[..db.len().min(32)].copy_from_slice(&db[..db.len().min(32)]);
    w.write_all(&desc)?;
    // payload
    w.write_all(payload)?;
    Ok(())
}

/// EVLR header size in bytes (60).
pub const EVLR_HEADER_SIZE: usize = 60;
