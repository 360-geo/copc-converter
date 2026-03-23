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

// ---------------------------------------------------------------------------
// Temporal Index EVLR (optional extension)
// ---------------------------------------------------------------------------

/// Header for the temporal index EVLR (16 bytes).
pub struct TemporalIndexHeader {
    pub version: u32,
    pub stride: u32,
    pub node_count: u32,
}

impl TemporalIndexHeader {
    pub fn write<W: Write>(&self, w: &mut W) -> anyhow::Result<()> {
        w.write_u32::<LittleEndian>(self.version)?;
        w.write_u32::<LittleEndian>(self.stride)?;
        w.write_u32::<LittleEndian>(self.node_count)?;
        w.write_u32::<LittleEndian>(0)?; // reserved
        Ok(())
    }
}

/// One node's temporal samples in the temporal index EVLR.
pub struct TemporalIndexEntry {
    pub key: VoxelKey,
    pub samples: Vec<f64>,
}

impl TemporalIndexEntry {
    pub fn write<W: Write>(&self, w: &mut W) -> anyhow::Result<()> {
        w.write_i32::<LittleEndian>(self.key.level)?;
        w.write_i32::<LittleEndian>(self.key.x)?;
        w.write_i32::<LittleEndian>(self.key.y)?;
        w.write_i32::<LittleEndian>(self.key.z)?;
        w.write_u32::<LittleEndian>(self.samples.len() as u32)?;
        for &t in &self.samples {
            w.write_f64::<LittleEndian>(t)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use byteorder::{LittleEndian, ReadBytesExt};
    use std::io::Cursor;

    #[test]
    fn temporal_header_size() {
        let mut buf = Vec::new();
        TemporalIndexHeader {
            version: 1,
            stride: 1000,
            node_count: 5,
        }
        .write(&mut buf)
        .unwrap();
        assert_eq!(buf.len(), 16);
    }

    #[test]
    fn temporal_entry_roundtrip() {
        let entry = TemporalIndexEntry {
            key: VoxelKey {
                level: 3,
                x: 1,
                y: 2,
                z: 4,
            },
            samples: vec![100.0, 200.5, 300.75],
        };
        let mut buf = Vec::new();
        entry.write(&mut buf).unwrap();

        // VoxelKey (16) + sample_count (4) + 3 * f64 (24) = 44
        assert_eq!(buf.len(), 44);

        let mut r = Cursor::new(&buf);
        assert_eq!(r.read_i32::<LittleEndian>().unwrap(), 3);
        assert_eq!(r.read_i32::<LittleEndian>().unwrap(), 1);
        assert_eq!(r.read_i32::<LittleEndian>().unwrap(), 2);
        assert_eq!(r.read_i32::<LittleEndian>().unwrap(), 4);
        assert_eq!(r.read_u32::<LittleEndian>().unwrap(), 3);
        assert_eq!(r.read_f64::<LittleEndian>().unwrap(), 100.0);
        assert_eq!(r.read_f64::<LittleEndian>().unwrap(), 200.5);
        assert_eq!(r.read_f64::<LittleEndian>().unwrap(), 300.75);
    }

    #[test]
    fn temporal_entry_empty_samples() {
        let entry = TemporalIndexEntry {
            key: VoxelKey {
                level: 0,
                x: 0,
                y: 0,
                z: 0,
            },
            samples: vec![],
        };
        let mut buf = Vec::new();
        entry.write(&mut buf).unwrap();
        // VoxelKey (16) + sample_count (4) = 20
        assert_eq!(buf.len(), 20);
    }
}
