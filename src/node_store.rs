//! Storage backend for per-node point data during build.
//!
//! The build and merge stages produce up to ~`total_points / MAX_LEAF_POINTS`
//! octree nodes, each holding some points. A naive one-file-per-node layout
//! exhausts the inode budget on shared scratch filesystems once node counts
//! climb into the hundred-thousands. Callers pick a backend via
//! [`crate::NodeStorage`]:
//!
//! * [`FileNodeStore`] — one temp file per node. Simple, zero dead space,
//!   matches the original pipeline behaviour. Inode-hungry.
//! * [`PackedNodeStore`] — one append-only pack file per rayon worker plus
//!   an in-memory `VoxelKey → location` index. Uses a handful of files
//!   regardless of node count; trades disk space for inodes (overwrites
//!   leak dead space).

use crate::TempCompression;
use crate::copc_types::VoxelKey;
use crate::octree::{RawPoint, count_temp_file_points, read_temp_batches, write_temp_batch};
use anyhow::{Context, Result};
use dashmap::DashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Storage backend for per-node point data during build.
///
/// All methods are safe to call concurrently from rayon workers. `count`
/// on a key that was never written returns `Ok(0)`; `read` returns
/// `Ok(vec![])`. Writes overwrite any previous data for the key.
pub(crate) trait NodeStore: Send + Sync {
    fn write(&self, key: &VoxelKey, points: &[RawPoint]) -> Result<()>;
    fn read(&self, key: &VoxelKey) -> Result<Vec<RawPoint>>;
    fn count(&self, key: &VoxelKey) -> Result<u64>;
}

// ---------------------------------------------------------------------------
// FileNodeStore — one file per node
// ---------------------------------------------------------------------------

pub(crate) struct FileNodeStore {
    tmp_dir: PathBuf,
    num_extra_bytes: u16,
    codec: TempCompression,
}

impl FileNodeStore {
    pub(crate) fn new(tmp_dir: PathBuf, num_extra_bytes: u16, codec: TempCompression) -> Self {
        Self {
            tmp_dir,
            num_extra_bytes,
            codec,
        }
    }

    fn node_path(&self, key: &VoxelKey) -> PathBuf {
        self.tmp_dir
            .join(format!("{}_{}_{}_{}", key.level, key.x, key.y, key.z))
    }
}

impl NodeStore for FileNodeStore {
    fn write(&self, key: &VoxelKey, points: &[RawPoint]) -> Result<()> {
        let path = self.node_path(key);
        let f = File::create(&path)?;
        let mut w = BufWriter::new(f);
        write_temp_batch(&mut w, points, self.num_extra_bytes, self.codec)?;
        w.flush().context("flush node temp file")?;
        Ok(())
    }

    fn read(&self, key: &VoxelKey) -> Result<Vec<RawPoint>> {
        let path = self.node_path(key);
        let f = match File::open(&path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(vec![]),
            Err(e) => return Err(e.into()),
        };
        read_temp_batches(f, self.num_extra_bytes, self.codec)
    }

    fn count(&self, key: &VoxelKey) -> Result<u64> {
        count_temp_file_points(&self.node_path(key), self.num_extra_bytes, self.codec)
    }
}

// ---------------------------------------------------------------------------
// PackedNodeStore — one append-only pack file per rayon worker
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct NodeLocation {
    pack_id: u16,
    offset: u64,
    byte_len: u32,
    point_count: u32,
}

pub(crate) struct PackedNodeStore {
    packs_dir: PathBuf,
    num_extra_bytes: u16,
    codec: TempCompression,
    /// One writer per pack file, wrapped in a Mutex so rayon workers that
    /// map to the same pack id serialize their appends.
    packs: Vec<Mutex<BufWriter<File>>>,
    /// Current append offset for each pack. Tracked separately from the
    /// writer so we can capture the offset before the write without having
    /// to consult the file system.
    pack_offsets: Vec<Mutex<u64>>,
    /// `VoxelKey → NodeLocation`. Read-heavy during merge and writer phases;
    /// `DashMap` gives us concurrent reads and writes without a global lock.
    index: DashMap<VoxelKey, NodeLocation>,
}

impl PackedNodeStore {
    pub(crate) fn new(
        tmp_dir: &Path,
        num_extra_bytes: u16,
        codec: TempCompression,
        pack_count: usize,
    ) -> Result<Self> {
        let packs_dir = tmp_dir.join("nodes");
        std::fs::create_dir_all(&packs_dir)
            .with_context(|| format!("creating packs dir {:?}", packs_dir))?;

        let mut packs = Vec::with_capacity(pack_count);
        let mut pack_offsets = Vec::with_capacity(pack_count);
        for i in 0..pack_count {
            let path = packs_dir.join(format!("pack_{i}.bin"));
            let f = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&path)
                .with_context(|| format!("creating pack file {:?}", path))?;
            packs.push(Mutex::new(BufWriter::new(f)));
            pack_offsets.push(Mutex::new(0));
        }

        Ok(Self {
            packs_dir,
            num_extra_bytes,
            codec,
            packs,
            pack_offsets,
            index: DashMap::new(),
        })
    }

    fn current_pack_id(&self) -> usize {
        rayon::current_thread_index()
            .map(|i| i % self.packs.len())
            .unwrap_or(0)
    }

    fn pack_path(&self, pack_id: u16) -> PathBuf {
        self.packs_dir.join(format!("pack_{pack_id}.bin"))
    }
}

impl NodeStore for PackedNodeStore {
    fn write(&self, key: &VoxelKey, points: &[RawPoint]) -> Result<()> {
        // Serialize first so we know the exact byte length and the pack
        // Mutex is held for the shortest possible time.
        let mut buf = Vec::new();
        write_temp_batch(&mut buf, points, self.num_extra_bytes, self.codec)?;
        let byte_len = buf.len() as u32;

        let pack_id = self.current_pack_id();
        let offset = {
            let mut writer = self.packs[pack_id]
                .lock()
                .expect("pack writer mutex poisoned");
            let mut cursor = self.pack_offsets[pack_id]
                .lock()
                .expect("pack offset mutex poisoned");
            let offset = *cursor;
            writer
                .write_all(&buf)
                .with_context(|| format!("appending to pack {pack_id}"))?;
            *cursor += byte_len as u64;
            offset
        };

        self.index.insert(
            *key,
            NodeLocation {
                pack_id: pack_id as u16,
                offset,
                byte_len,
                point_count: points.len() as u32,
            },
        );
        Ok(())
    }

    fn read(&self, key: &VoxelKey) -> Result<Vec<RawPoint>> {
        let loc = match self.index.get(key) {
            Some(loc) => *loc,
            None => return Ok(vec![]),
        };

        // Flush the pack writer's buffer so the bytes we're about to read
        // from disk are actually there. We only flush, don't drop the writer,
        // so later writes keep appending through the same BufWriter.
        {
            let mut writer = self.packs[loc.pack_id as usize]
                .lock()
                .expect("pack writer mutex poisoned");
            writer.flush().context("flush pack before read")?;
        }

        let path = self.pack_path(loc.pack_id);
        let mut f = File::open(&path).with_context(|| format!("opening pack file {:?}", path))?;
        f.seek(SeekFrom::Start(loc.offset))
            .context("seek to node offset")?;
        let mut limited = f.take(loc.byte_len as u64);
        read_temp_batches(&mut limited, self.num_extra_bytes, self.codec)
    }

    fn count(&self, key: &VoxelKey) -> Result<u64> {
        Ok(self
            .index
            .get(key)
            .map(|loc| loc.point_count as u64)
            .unwrap_or(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn sample_point(x: i32) -> RawPoint {
        RawPoint {
            x,
            y: x * 2,
            z: x * 3,
            intensity: 100,
            return_number: 1,
            number_of_returns: 1,
            flags: 0,
            classification: 0,
            scan_angle: 0,
            user_data: 0,
            point_source_id: 0,
            gps_time: x as f64,
            red: 0,
            green: 0,
            blue: 0,
            nir: 0,
            extras: Box::<[u8]>::default(),
        }
    }

    #[test]
    fn packed_write_read_roundtrip() {
        let tmp = std::env::temp_dir().join(format!("copc_test_packed_{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let store = PackedNodeStore::new(&tmp, 0, TempCompression::None, 2).unwrap();

        let key = VoxelKey {
            level: 3,
            x: 1,
            y: 2,
            z: 3,
        };
        let pts = vec![sample_point(1), sample_point(2), sample_point(3)];
        store.write(&key, &pts).unwrap();

        let got = store.read(&key).unwrap();
        assert_eq!(got.len(), pts.len());
        assert_eq!(got[0].x, 1);
        assert_eq!(got[2].z, 9);
        assert_eq!(store.count(&key).unwrap(), 3);

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn packed_overwrite_returns_latest() {
        let tmp = std::env::temp_dir().join(format!("copc_test_packed_ow_{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let store = PackedNodeStore::new(&tmp, 0, TempCompression::None, 1).unwrap();

        let key = VoxelKey {
            level: 2,
            x: 0,
            y: 0,
            z: 0,
        };
        store.write(&key, &[sample_point(10)]).unwrap();
        store
            .write(&key, &[sample_point(20), sample_point(21)])
            .unwrap();

        let got = store.read(&key).unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].x, 20);
        assert_eq!(got[1].x, 21);
        assert_eq!(store.count(&key).unwrap(), 2);

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn packed_missing_key_is_empty() {
        let tmp =
            std::env::temp_dir().join(format!("copc_test_packed_miss_{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let store = PackedNodeStore::new(&tmp, 0, TempCompression::None, 1).unwrap();

        let key = VoxelKey {
            level: 1,
            x: 0,
            y: 0,
            z: 0,
        };
        assert!(store.read(&key).unwrap().is_empty());
        assert_eq!(store.count(&key).unwrap(), 0);

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn packed_concurrent_writes() {
        let tmp = std::env::temp_dir().join(format!(
            "copc_test_packed_concurrent_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&tmp).unwrap();
        let store = Arc::new(PackedNodeStore::new(&tmp, 0, TempCompression::None, 4).unwrap());

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();

        pool.install(|| {
            use rayon::prelude::*;
            (0..200).into_par_iter().for_each(|i| {
                let key = VoxelKey {
                    level: 4,
                    x: i,
                    y: 0,
                    z: 0,
                };
                let pts: Vec<_> = (0..5).map(|j| sample_point(i * 10 + j)).collect();
                store.write(&key, &pts).unwrap();
            });
        });

        for i in 0..200 {
            let key = VoxelKey {
                level: 4,
                x: i,
                y: 0,
                z: 0,
            };
            let got = store.read(&key).unwrap();
            assert_eq!(got.len(), 5);
            assert_eq!(got[0].x, i * 10);
            assert_eq!(got[4].x, i * 10 + 4);
        }

        std::fs::remove_dir_all(&tmp).ok();
    }
}
