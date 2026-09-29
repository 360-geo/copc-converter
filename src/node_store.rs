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
use crate::octree::{RawPoint, count_temp_file_points, stream_temp_batches, write_temp_batch};
use anyhow::{Context, Result};
use dashmap::DashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Storage backend for per-node point data during build.
///
/// All methods are safe to call concurrently from rayon workers. A key that
/// was never written has `count` 0 and streams no points. Writes overwrite
/// any previous data for the key.
pub(crate) trait NodeStore: Send + Sync {
    fn write(&self, key: &VoxelKey, points: &[RawPoint]) -> Result<()>;
    fn count(&self, key: &VoxelKey) -> Result<u64>;
    /// Visit every point of a node without materialising them all, so a
    /// node larger than memory can be processed in bounded space.
    fn stream(&self, key: &VoxelKey, f: &mut dyn FnMut(RawPoint) -> Result<()>) -> Result<()>;
    /// Start replacing a node's data batch by batch. The old data stays
    /// readable (e.g. streamed by the caller) until `finish` swaps it in.
    fn writer(&self, key: &VoxelKey) -> Result<Box<dyn NodeWriter + '_>>;
}

/// Incremental writer from [`NodeStore::writer`]. Dropping it without
/// calling `finish` leaves the node's previous data in place.
pub(crate) trait NodeWriter {
    fn append(&mut self, points: &[RawPoint]) -> Result<()>;
    fn finish(self: Box<Self>) -> Result<()>;
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

    fn count(&self, key: &VoxelKey) -> Result<u64> {
        count_temp_file_points(&self.node_path(key), self.num_extra_bytes, self.codec)
    }

    fn stream(&self, key: &VoxelKey, f: &mut dyn FnMut(RawPoint) -> Result<()>) -> Result<()> {
        let file = match File::open(self.node_path(key)) {
            Ok(file) => file,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(e.into()),
        };
        stream_temp_batches(file, self.num_extra_bytes, self.codec, f)
    }

    fn writer(&self, key: &VoxelKey) -> Result<Box<dyn NodeWriter + '_>> {
        // Write beside the node and rename over it on finish, so the node's
        // current file can still be streamed while its replacement is built.
        let path = self.node_path(key);
        let partial = path.with_extension("partial");
        let file = File::create(&partial).with_context(|| format!("creating {partial:?}"))?;
        Ok(Box::new(FileNodeWriter {
            out: BufWriter::new(file),
            partial,
            path,
            num_extra_bytes: self.num_extra_bytes,
            codec: self.codec,
        }))
    }
}

struct FileNodeWriter {
    out: BufWriter<File>,
    partial: PathBuf,
    path: PathBuf,
    num_extra_bytes: u16,
    codec: TempCompression,
}

impl NodeWriter for FileNodeWriter {
    fn append(&mut self, points: &[RawPoint]) -> Result<()> {
        if !points.is_empty() {
            write_temp_batch(&mut self.out, points, self.num_extra_bytes, self.codec)?;
        }
        Ok(())
    }

    fn finish(mut self: Box<Self>) -> Result<()> {
        self.out.flush().context("flush node temp file")?;
        std::fs::rename(&self.partial, &self.path)
            .with_context(|| format!("renaming {:?} into place", self.partial))?;
        Ok(())
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
    /// `VoxelKey → segments`. A node written in one go has one segment; a
    /// node streamed through [`NodeStore::writer`] has one per batch.
    /// Read-heavy during merge and writer phases; `DashMap` gives us
    /// concurrent reads and writes without a global lock.
    index: DashMap<VoxelKey, Vec<NodeLocation>>,
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

impl PackedNodeStore {
    /// Serialize one batch and append it to the current thread's pack.
    fn append_segment(&self, points: &[RawPoint]) -> Result<NodeLocation> {
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
        Ok(NodeLocation {
            pack_id: pack_id as u16,
            offset,
            byte_len,
            point_count: points.len() as u32,
        })
    }

    /// Open a reader over one segment's bytes.
    fn open_segment(&self, loc: &NodeLocation) -> Result<std::io::Take<File>> {
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
        Ok(f.take(loc.byte_len as u64))
    }

    fn segments(&self, key: &VoxelKey) -> Vec<NodeLocation> {
        self.index.get(key).map(|s| s.clone()).unwrap_or_default()
    }
}

impl NodeStore for PackedNodeStore {
    fn write(&self, key: &VoxelKey, points: &[RawPoint]) -> Result<()> {
        let loc = self.append_segment(points)?;
        self.index.insert(*key, vec![loc]);
        Ok(())
    }

    fn count(&self, key: &VoxelKey) -> Result<u64> {
        Ok(self
            .index
            .get(key)
            .map(|segs| segs.iter().map(|l| l.point_count as u64).sum())
            .unwrap_or(0))
    }

    fn stream(&self, key: &VoxelKey, f: &mut dyn FnMut(RawPoint) -> Result<()>) -> Result<()> {
        for loc in &self.segments(key) {
            stream_temp_batches(
                self.open_segment(loc)?,
                self.num_extra_bytes,
                self.codec,
                &mut *f,
            )?;
        }
        Ok(())
    }

    fn writer(&self, key: &VoxelKey) -> Result<Box<dyn NodeWriter + '_>> {
        Ok(Box::new(PackedNodeWriter {
            store: self,
            key: *key,
            segments: Vec::new(),
        }))
    }
}

struct PackedNodeWriter<'a> {
    store: &'a PackedNodeStore,
    key: VoxelKey,
    segments: Vec<NodeLocation>,
}

impl NodeWriter for PackedNodeWriter<'_> {
    fn append(&mut self, points: &[RawPoint]) -> Result<()> {
        if !points.is_empty() {
            self.segments.push(self.store.append_segment(points)?);
        }
        Ok(())
    }

    fn finish(self: Box<Self>) -> Result<()> {
        // Replacing the index entry is the atomic swap; the old segments
        // become dead space in the packs, as with any overwrite.
        self.store.index.insert(self.key, self.segments);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn read_all(store: &dyn NodeStore, key: &VoxelKey) -> Vec<RawPoint> {
        let mut out = Vec::new();
        store
            .stream(key, &mut |p| {
                out.push(p);
                Ok(())
            })
            .unwrap();
        out
    }

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

        let got = read_all(&store, &key);
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

        let got = read_all(&store, &key);
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
        assert!(read_all(&store, &key).is_empty());
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
            let got = read_all(&*store, &key);
            assert_eq!(got.len(), 5);
            assert_eq!(got[0].x, i * 10);
            assert_eq!(got[4].x, i * 10 + 4);
        }

        std::fs::remove_dir_all(&tmp).ok();
    }

    /// Replace a node through `writer` while streaming its old contents,
    /// as the merge does: the old data must stay readable until `finish`.
    fn check_streamed_rewrite(store: &dyn NodeStore) {
        let key = VoxelKey {
            level: 5,
            x: 1,
            y: 1,
            z: 1,
        };
        store
            .write(&key, &[sample_point(1), sample_point(2), sample_point(3)])
            .unwrap();

        let mut w = store.writer(&key).unwrap();
        let mut seen = Vec::new();
        store
            .stream(&key, &mut |p| {
                seen.push(p.x);
                // Keep every point but the first, in two appends.
                if p.x > 1 {
                    w.append(&[p])?;
                }
                Ok(())
            })
            .unwrap();
        assert_eq!(seen, vec![1, 2, 3]);
        // Old data still visible until finish.
        assert_eq!(store.count(&key).unwrap(), 3);
        w.finish().unwrap();

        let got: Vec<i32> = read_all(store, &key).iter().map(|p| p.x).collect();
        assert_eq!(got, vec![2, 3]);
        assert_eq!(store.count(&key).unwrap(), 2);
    }

    #[test]
    fn file_store_streamed_rewrite() {
        let tmp = std::env::temp_dir().join(format!("copc_test_file_rw_{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        check_streamed_rewrite(&FileNodeStore::new(tmp.clone(), 0, TempCompression::None));
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn packed_store_streamed_rewrite() {
        let tmp = std::env::temp_dir().join(format!("copc_test_packed_rw_{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        check_streamed_rewrite(&PackedNodeStore::new(&tmp, 0, TempCompression::Lz4, 2).unwrap());
        std::fs::remove_dir_all(&tmp).ok();
    }
}
