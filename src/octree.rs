/// Out-of-core octree builder.
///
/// Strategy
/// --------
/// 1. Pass 1 – scan all input files in parallel, collect bounding box + point count.
/// 2. Determine octree depth so that leaf nodes contain ≤ MAX_LEAF_POINTS on average.
/// 3. Pass 2 – read every point, assign it to the leaf voxel key, and
///    accumulate into per-key temporary files on disk.
///    Point classification (key + coordinate conversion) is parallelized via rayon.
///    Memory-aware: fast path (full file) or batched path depending on budget.
/// 4. Normalize leaves: any leaf with > MAX_LEAF_POINTS is split into children on disk.
/// 5. Build the tree bottom-up in parallel: each parent node gets a thinned sample
///    of its children's points written back to disk.
/// 6. Produce the list of (VoxelKey, point_count) for the writer, which reads from disk.
///
/// Memory usage is bounded by the configurable memory budget.
use crate::PipelineConfig;
use crate::copc_types::VoxelKey;
use anyhow::{Context, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::{debug, info};

// ---------------------------------------------------------------------------
// Chunked-build constants and helper types
// ---------------------------------------------------------------------------

/// Maximum number of leaf chunk files kept open simultaneously across all
/// distribute workers in the chunked path. Divided evenly between workers.
///
/// Each entry holds a `BufWriter<File>` (~8 KiB internal buffer) plus an OS
/// file descriptor. Capping prevents FD exhaustion on hosts with low ulimits
/// (default 256 on macOS, 1024 on most Linux containers).
const CHUNKED_OPEN_FILES_CAP: usize = 512;

/// Minimum per-worker LRU capacity for the chunk writer cache.
const MIN_PER_WORKER_CHUNK_FILES: usize = 32;

/// Bounded LRU cache of append-mode `BufWriter`s for chunk temp files.
///
/// Keys are chunk indices (`u32`). On eviction the writer is flushed and
/// dropped, releasing both its buffer and its file descriptor. Subsequent
/// writes to the same chunk index reopen the file in append mode.
///
/// This is the chunk-keyed analog of the per-leaf path's writer cache: the
/// design is identical, only the key type changes.
struct ChunkWriterCache {
    writers: HashMap<u32, BufWriter<File>>,
    /// Insertion / access order. Front = least recently used.
    order: VecDeque<u32>,
    capacity: usize,
    codec: TempCompression,
}

impl ChunkWriterCache {
    fn new(capacity: usize, codec: TempCompression) -> Self {
        ChunkWriterCache {
            writers: HashMap::new(),
            order: VecDeque::new(),
            capacity,
            codec,
        }
    }

    /// Append the given points to the temp file for `chunk_idx`, opening (and
    /// possibly evicting another entry) as needed.
    fn append(&mut self, chunk_idx: u32, shard_dir: &Path, points: &[RawPoint]) -> Result<()> {
        if points.is_empty() {
            return Ok(());
        }

        if self.writers.contains_key(&chunk_idx) {
            // Move to back (most-recently-used).
            if let Some(pos) = self.order.iter().position(|k| *k == chunk_idx) {
                self.order.remove(pos);
            }
            self.order.push_back(chunk_idx);
        } else {
            // Evict if at capacity.
            while self.writers.len() >= self.capacity {
                if let Some(victim) = self.order.pop_front() {
                    if let Some(mut w) = self.writers.remove(&victim) {
                        w.flush().context("flush evicted chunk writer")?;
                    }
                } else {
                    break;
                }
            }
            let path = chunk_shard_path(shard_dir, chunk_idx);
            let f = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .with_context(|| format!("Cannot open chunk file {:?}", path))?;
            self.writers.insert(chunk_idx, BufWriter::new(f));
            self.order.push_back(chunk_idx);
        }

        let w = self.writers.get_mut(&chunk_idx).expect("just inserted");
        write_temp_batch(w, points, self.codec)?;
        Ok(())
    }

    /// Flush and drop every cached writer.
    fn flush_all(&mut self) -> Result<()> {
        for (_, mut w) in self.writers.drain() {
            w.flush().context("flush chunk writer")?;
        }
        self.order.clear();
        Ok(())
    }
}

/// Path for a chunk's per-worker shard file.
fn chunk_shard_path(shard_dir: &Path, chunk_idx: u32) -> PathBuf {
    shard_dir.join(format!("chunk_{chunk_idx}"))
}

/// Path for a chunk's canonical (merged) file under `tmp_dir/chunks/`.
fn chunk_canonical_path(chunks_dir: &Path, chunk_idx: u32) -> PathBuf {
    chunks_dir.join(format!("chunk_{chunk_idx}"))
}

/// Same heuristic as the per-leaf parallel distribute path: ~2/3 of cores,
/// leaving headroom for the LAZ parallel decoder. Capped by input file count.
fn chunked_distribute_worker_count(input_file_count: usize) -> usize {
    let cores = rayon::current_num_threads();
    let target = ((cores * 2) / 3).max(2);
    target.min(input_file_count).max(1)
}

/// How many extra octree levels a chunk needs to subdivide internally so its
/// leaves hold ~`MAX_LEAF_POINTS` on average.
///
/// Computes `ceil(log8(point_count / MAX_LEAF_POINTS))`, clamped to a sane
/// upper bound. Returns 0 for chunks at or below `MAX_LEAF_POINTS` (the
/// chunk root is the only leaf).
///
/// **Uniform subdivision** — dense regions inside the chunk may exceed
/// `MAX_LEAF_POINTS` per leaf, which is acceptable per the design doc §9.3.
/// `grid_sample`'s natural cascading handles the imbalance during the
/// bottom-up build.
fn chunk_local_extra_levels(point_count: u64) -> u32 {
    if point_count <= MAX_LEAF_POINTS {
        return 0;
    }
    let mut d = 0u32;
    while (point_count as f64) / (8u64.pow(d) as f64) > MAX_LEAF_POINTS as f64 {
        d += 1;
        // Cap at a generous limit. With 8^9 = 134M, even a 100B-point
        // chunk would only need d = 9, but we allow a couple more levels
        // for highly imbalanced chunks before giving up.
        if d > 12 {
            break;
        }
    }
    d
}

/// Build the cell → chunk_index lookup table from a chunk plan.
///
/// The grid has `grid_size³` fine cells; for each chunk we fill in every
/// fine cell it covers. A chunk at level L covers a `2^(grid_depth - L)`
/// cube per axis, since the grid corresponds to octree cells at `grid_depth`.
///
/// Returns a flat `Vec<u32>` indexed by `gx + gy*G + gz*G²`. Each entry
/// holds the chunk index this cell belongs to. Cells not covered by any
/// chunk (which should not happen for a well-formed plan) get `u32::MAX`
/// as a sentinel; we treat any point landing in such a cell as a bug.
fn build_chunk_lut(plan: &crate::chunking::ChunkPlan) -> Vec<u32> {
    let g = plan.grid_size as usize;
    let n_cells = g * g * g;
    let mut lut = vec![u32::MAX; n_cells];

    for (chunk_idx, chunk) in plan.chunks.iter().enumerate() {
        // The chunk lives at chunk.level. The fine grid is at plan.grid_depth.
        // Each chunk cell covers a stride of 2^(grid_depth - chunk.level)
        // along each axis in the fine grid.
        let level_diff = plan.grid_depth as i32 - chunk.level as i32;
        debug_assert!(
            level_diff >= 0,
            "chunk level {} above grid depth {}",
            chunk.level,
            plan.grid_depth
        );
        let stride: usize = 1usize << level_diff as u32;

        // Top-left corner of this chunk in fine-grid coordinates, clamped
        // defensively against malformed chunks. (Should never trigger for
        // a plan produced by `merge_sparse_cells`.)
        let base_x = (chunk.gx as usize * stride).min(g);
        let base_y = (chunk.gy as usize * stride).min(g);
        let base_z = (chunk.gz as usize * stride).min(g);
        let end_x = (base_x + stride).min(g);
        let end_y = (base_y + stride).min(g);
        let end_z = (base_z + stride).min(g);

        for z in base_z..end_z {
            for y in base_y..end_y {
                let row_start = base_x + y * g + z * g * g;
                let row_end = end_x + y * g + z * g * g;
                for cell in &mut lut[row_start..row_end] {
                    *cell = chunk_idx as u32;
                }
            }
        }
    }

    lut
}

/// Concatenate per-worker shard files into the canonical
/// `tmp_dir/chunks/chunk_N` location. Runs in parallel over chunk indices.
///
/// After this returns successfully, the `shards/` subdirectory is removed
/// and only the canonical chunk files remain under `chunks/`.
fn merge_chunk_shards(shards_root: &Path, chunks_root: &Path, n_chunks: u32) -> Result<()> {
    let num_workers = match std::fs::read_dir(shards_root) {
        Ok(it) => it
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .count(),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => 0,
        Err(e) => return Err(e.into()),
    };
    debug!(
        "Merging {} chunks across {} worker shards",
        n_chunks, num_workers
    );

    (0..n_chunks)
        .into_par_iter()
        .try_for_each(|chunk_idx| -> Result<()> {
            let canonical = chunk_canonical_path(chunks_root, chunk_idx);
            let f = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&canonical)
                .with_context(|| format!("opening canonical chunk file {:?}", canonical))?;
            let mut out = BufWriter::new(f);
            for w in 0..num_workers {
                let shard_path = shards_root
                    .join(w.to_string())
                    .join(format!("chunk_{chunk_idx}"));
                match File::open(&shard_path) {
                    Ok(f) => {
                        let mut reader = BufReader::new(f);
                        std::io::copy(&mut reader, &mut out).with_context(|| {
                            format!("copying shard {:?} into {:?}", shard_path, canonical)
                        })?;
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
                    Err(e) => return Err(e.into()),
                }
            }
            out.flush().context("flush merged chunk file")?;
            Ok(())
        })?;

    std::fs::remove_dir_all(shards_root)
        .with_context(|| format!("removing shards root {:?}", shards_root))?;
    Ok(())
}

/// Task fed into the parallel grid-sampling step: parent key, child keys, and
/// indexed points (child-index, point).
type SampleTask = (VoxelKey, Vec<VoxelKey>, Vec<(usize, RawPoint)>);

/// Result coming back from parallel grid-sampling: parent key, child keys,
/// promoted points, and per-child remaining points.
type SampleResult = (VoxelKey, Vec<VoxelKey>, Vec<RawPoint>, Vec<Vec<RawPoint>>);

// ---------------------------------------------------------------------------
// Morton code helper (used for spatially coherent traversal order)
// ---------------------------------------------------------------------------

fn morton3(x: u32, y: u32, z: u32) -> u64 {
    #[inline]
    fn spread(mut v: u64) -> u64 {
        v &= 0x1F_FFFF;
        v = (v | (v << 32)) & 0x1F00000000FFFF;
        v = (v | (v << 16)) & 0x1F0000FF0000FF;
        v = (v | (v << 8)) & 0x100F00F00F00F00F;
        v = (v | (v << 4)) & 0x10C30C30C30C30C3;
        v = (v | (v << 2)) & 0x1249249249249249;
        v
    }
    spread(x as u64) | (spread(y as u64) << 1) | (spread(z as u64) << 2)
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum points per leaf voxel before we subdivide further.
const MAX_LEAF_POINTS: u64 = 100_000;

/// Grid cells per axis for LOD thinning. Matches untwine's CellCount = 128.
/// Higher values keep more points at coarse LOD levels (better progressive rendering).
const GRID_CELLS_PER_AXIS: i64 = 128;

// ---------------------------------------------------------------------------
// Raw point storage
// ---------------------------------------------------------------------------

/// A raw point stored as scaled integer coordinates plus attributes.
/// Scaled ints allow exact LAS round-trip without floating-point loss.
#[derive(Debug, Clone)]
pub struct RawPoint {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub intensity: u16,
    pub return_number: u8,
    pub number_of_returns: u8,
    pub classification: u8,
    pub scan_angle: i16,
    pub user_data: u8,
    pub point_source_id: u16,
    pub gps_time: f64,
    pub red: u16,
    pub green: u16,
    pub blue: u16,
    pub nir: u16,
}

impl RawPoint {
    pub const BYTE_SIZE: usize = 4 + 4 + 4 + 2 + 1 + 1 + 1 + 2 + 1 + 2 + 8 + 2 + 2 + 2 + 2; // 38

    #[allow(unused)]
    pub fn write<W: std::io::Write>(&self, w: &mut W) -> Result<()> {
        let mut buf = [0u8; Self::BYTE_SIZE];
        {
            use std::io::Cursor;
            let mut c = Cursor::new(&mut buf[..]);
            c.write_i32::<LittleEndian>(self.x)?;
            c.write_i32::<LittleEndian>(self.y)?;
            c.write_i32::<LittleEndian>(self.z)?;
            c.write_u16::<LittleEndian>(self.intensity)?;
            c.write_u8(self.return_number)?;
            c.write_u8(self.number_of_returns)?;
            c.write_u8(self.classification)?;
            c.write_i16::<LittleEndian>(self.scan_angle)?;
            c.write_u8(self.user_data)?;
            c.write_u16::<LittleEndian>(self.point_source_id)?;
            c.write_f64::<LittleEndian>(self.gps_time)?;
            c.write_u16::<LittleEndian>(self.red)?;
            c.write_u16::<LittleEndian>(self.green)?;
            c.write_u16::<LittleEndian>(self.blue)?;
            c.write_u16::<LittleEndian>(self.nir)?;
        }
        w.write_all(&buf)?;
        Ok(())
    }

    pub fn read<R: std::io::Read>(r: &mut R) -> Result<Self> {
        let mut buf = [0u8; Self::BYTE_SIZE];
        r.read_exact(&mut buf)?;
        let mut c = std::io::Cursor::new(&buf[..]);
        Ok(RawPoint {
            x: c.read_i32::<LittleEndian>()?,
            y: c.read_i32::<LittleEndian>()?,
            z: c.read_i32::<LittleEndian>()?,
            intensity: c.read_u16::<LittleEndian>()?,
            return_number: c.read_u8()?,
            number_of_returns: c.read_u8()?,
            classification: c.read_u8()?,
            scan_angle: c.read_i16::<LittleEndian>()?,
            user_data: c.read_u8()?,
            point_source_id: c.read_u16::<LittleEndian>()?,
            gps_time: c.read_f64::<LittleEndian>()?,
            red: c.read_u16::<LittleEndian>()?,
            green: c.read_u16::<LittleEndian>()?,
            blue: c.read_u16::<LittleEndian>()?,
            nir: c.read_u16::<LittleEndian>()?,
        })
    }

    /// Write multiple points to a writer in a single bulk operation.
    pub fn write_bulk<W: std::io::Write>(points: &[RawPoint], w: &mut W) -> Result<()> {
        let mut buf = vec![0u8; Self::BYTE_SIZE * points.len()];
        let mut c = std::io::Cursor::new(&mut buf[..]);
        for p in points {
            c.write_i32::<LittleEndian>(p.x)?;
            c.write_i32::<LittleEndian>(p.y)?;
            c.write_i32::<LittleEndian>(p.z)?;
            c.write_u16::<LittleEndian>(p.intensity)?;
            c.write_u8(p.return_number)?;
            c.write_u8(p.number_of_returns)?;
            c.write_u8(p.classification)?;
            c.write_i16::<LittleEndian>(p.scan_angle)?;
            c.write_u8(p.user_data)?;
            c.write_u16::<LittleEndian>(p.point_source_id)?;
            c.write_f64::<LittleEndian>(p.gps_time)?;
            c.write_u16::<LittleEndian>(p.red)?;
            c.write_u16::<LittleEndian>(p.green)?;
            c.write_u16::<LittleEndian>(p.blue)?;
            c.write_u16::<LittleEndian>(p.nir)?;
        }
        w.write_all(&buf)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Temp-file frame format
// ---------------------------------------------------------------------------
//
// Every scratch file on disk (per-node files and per-chunk shard/canonical
// files) is a sequence of zero or more *batches*. Each batch contains:
//
//     u32 LE point_count      (4 bytes)
//     point_count × RawPoint  (38 bytes per point)
//
// - In `TempCompression::None` mode, batches are laid down directly in the
//   file, back-to-back.
// - In `TempCompression::Lz4` mode, each batch is wrapped in its own
//   self-contained LZ4 frame (frame format). `FrameDecoder` walks multiple
//   concatenated frames transparently, so the reader logic is identical:
//   read u32 count + payload, repeat until EOF.
//
// Per-node files (`write_node_to_temp`) always contain exactly one batch,
// including zero-point batches. Per-chunk shard files contain one batch per
// `ChunkWriterCache::append` call — an LZ4 frame encoder cannot be resumed
// across re-opens, so each append finalises its own frame. The merged
// canonical chunk files inherit the multi-batch layout via
// `merge_chunk_shards`'s byte-level concatenation.

use crate::TempCompression;

/// Adapter that flattens a sequence of concatenated LZ4 frames into a single
/// logical stream.
///
/// `lz4_flex::frame::FrameDecoder` returns `Ok(0)` at the end of each frame
/// even when more frames follow. Most `Read` consumers (including
/// `Read::read_exact` and `byteorder::ReadBytesExt`) treat `Ok(0)` as
/// end-of-stream, which truncates multi-frame files after the first frame.
///
/// This adapter re-probes the underlying reader whenever it sees an
/// end-of-frame `Ok(0)`: the subsequent `read` call on the decoder will
/// attempt to parse the next frame's header and yield its first block.
/// Two consecutive zero-returns — end of frame followed by genuine EOF on
/// the underlying reader — signal true end of stream.
struct MultiFrameReader<R: std::io::Read> {
    inner: R,
}

impl<R: std::io::Read> std::io::Read for MultiFrameReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        match self.inner.read(buf)? {
            0 => {
                // End of current frame. Try once more to see if another
                // frame follows; a second zero means genuine EOF.
                self.inner.read(buf)
            }
            n => Ok(n),
        }
    }
}

/// Write a single batch: a 4-byte little-endian point count followed by
/// `points.len() × RawPoint::BYTE_SIZE` payload bytes. If `codec` is
/// `Lz4`, the entire batch is encapsulated in a self-contained LZ4 frame.
fn write_temp_batch<W: Write>(
    w: &mut W,
    points: &[RawPoint],
    codec: TempCompression,
) -> Result<()> {
    match codec {
        TempCompression::None => {
            w.write_u32::<LittleEndian>(points.len() as u32)?;
            RawPoint::write_bulk(points, w)?;
        }
        TempCompression::Lz4 => {
            let mut enc = lz4_flex::frame::FrameEncoder::new(w);
            enc.write_u32::<LittleEndian>(points.len() as u32)?;
            RawPoint::write_bulk(points, &mut enc)?;
            enc.finish().context("finishing lz4 frame")?;
        }
    }
    Ok(())
}

/// Read every batch in a reader to EOF, concatenating their points.
/// For `Lz4` the reader is wrapped in a `FrameDecoder` + `MultiFrameReader`,
/// which together walk multi-frame streams transparently. An empty reader
/// produces an empty Vec.
fn read_temp_batches<R: std::io::Read>(r: R, codec: TempCompression) -> Result<Vec<RawPoint>> {
    match codec {
        TempCompression::None => read_batches_loop(&mut BufReader::new(r)),
        TempCompression::Lz4 => {
            let dec = lz4_flex::frame::FrameDecoder::new(BufReader::new(r));
            let mut mf = MultiFrameReader { inner: dec };
            read_batches_loop(&mut mf)
        }
    }
}

fn read_batches_loop<R: std::io::Read>(r: &mut R) -> Result<Vec<RawPoint>> {
    let mut out: Vec<RawPoint> = Vec::new();
    loop {
        let count = match r.read_u32::<LittleEndian>() {
            Ok(n) => n as usize,
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        };
        out.reserve(count);
        for _ in 0..count {
            out.push(RawPoint::read(r)?);
        }
    }
    Ok(out)
}

/// Count the points in a temp file by walking batch headers. For `None` mode
/// this seeks past payloads without decoding. For `Lz4` mode there is no
/// cheap way to skip a compressed payload, so we run the decoder and count
/// headers (still cheaper than materialising all points — payload bytes are
/// streamed through a throwaway sink). Returns 0 if the file does not exist.
fn count_temp_file_points(path: &Path, codec: TempCompression) -> Result<u64> {
    let f = match File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(e) => return Err(e.into()),
    };
    match codec {
        TempCompression::None => {
            use std::io::{Seek, SeekFrom};
            let mut f = f;
            let mut total: u64 = 0;
            loop {
                let count = match f.read_u32::<LittleEndian>() {
                    Ok(n) => n as u64,
                    Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                    Err(e) => return Err(e.into()),
                };
                total += count;
                let payload = count * RawPoint::BYTE_SIZE as u64;
                f.seek(SeekFrom::Current(payload as i64))?;
            }
            Ok(total)
        }
        TempCompression::Lz4 => {
            let dec = lz4_flex::frame::FrameDecoder::new(BufReader::new(f));
            let mut mf = MultiFrameReader { inner: dec };
            let mut total: u64 = 0;
            let mut sink = [0u8; 4096];
            loop {
                let count = match mf.read_u32::<LittleEndian>() {
                    Ok(n) => n as u64,
                    Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                    Err(e) => return Err(e.into()),
                };
                total += count;
                let mut remaining = count * RawPoint::BYTE_SIZE as u64;
                while remaining > 0 {
                    let want = remaining.min(sink.len() as u64) as usize;
                    mf.read_exact(&mut sink[..want])?;
                    remaining -= want as u64;
                }
            }
            Ok(total)
        }
    }
}

// ---------------------------------------------------------------------------
// Bounds
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Bounds {
    pub min_x: f64,
    pub min_y: f64,
    pub min_z: f64,
    pub max_x: f64,
    pub max_y: f64,
    pub max_z: f64,
}

impl Bounds {
    pub fn empty() -> Self {
        Bounds {
            min_x: f64::MAX,
            min_y: f64::MAX,
            min_z: f64::MAX,
            max_x: f64::MIN,
            max_y: f64::MIN,
            max_z: f64::MIN,
        }
    }

    pub fn expand_with(&mut self, x: f64, y: f64, z: f64) {
        if x < self.min_x {
            self.min_x = x;
        }
        if y < self.min_y {
            self.min_y = y;
        }
        if z < self.min_z {
            self.min_z = z;
        }
        if x > self.max_x {
            self.max_x = x;
        }
        if y > self.max_y {
            self.max_y = y;
        }
        if z > self.max_z {
            self.max_z = z;
        }
    }

    pub fn merge(&mut self, other: &Bounds) {
        if other.min_x < self.min_x {
            self.min_x = other.min_x;
        }
        if other.min_y < self.min_y {
            self.min_y = other.min_y;
        }
        if other.min_z < self.min_z {
            self.min_z = other.min_z;
        }
        if other.max_x > self.max_x {
            self.max_x = other.max_x;
        }
        if other.max_y > self.max_y {
            self.max_y = other.max_y;
        }
        if other.max_z > self.max_z {
            self.max_z = other.max_z;
        }
    }

    /// Cube that contains this AABB.
    pub fn to_cube(&self) -> (f64, f64, f64, f64) {
        let cx = (self.min_x + self.max_x) / 2.0;
        let cy = (self.min_y + self.max_y) / 2.0;
        let cz = (self.min_z + self.max_z) / 2.0;
        let half = ((self.max_x - self.min_x)
            .max(self.max_y - self.min_y)
            .max(self.max_z - self.min_z))
            / 2.0
            * 1.0001; // tiny epsilon
        (cx, cy, cz, half)
    }
}

// ---------------------------------------------------------------------------
// VoxelKey assignment
// ---------------------------------------------------------------------------

/// Assign a point to the voxel at the given tree depth.
#[allow(clippy::too_many_arguments)]
pub fn point_to_key(
    x: f64,
    y: f64,
    z: f64,
    cx: f64,
    cy: f64,
    cz: f64,
    halfsize: f64,
    depth: u32,
) -> VoxelKey {
    let mut vx = 0i32;
    let mut vy = 0i32;
    let mut vz = 0i32;
    let mut half = halfsize;
    let mut ox = cx;
    let mut oy = cy;
    let mut oz = cz;

    for _ in 0..depth {
        half /= 2.0;
        let bx = if x >= ox {
            vx = vx * 2 + 1;
            ox + half
        } else {
            vx *= 2;
            ox - half
        };
        let by = if y >= oy {
            vy = vy * 2 + 1;
            oy + half
        } else {
            vy *= 2;
            oy - half
        };
        let bz = if z >= oz {
            vz = vz * 2 + 1;
            oz + half
        } else {
            vz *= 2;
            oz - half
        };
        ox = bx;
        oy = by;
        oz = bz;
    }

    VoxelKey {
        level: depth as i32,
        x: vx,
        y: vy,
        z: vz,
    }
}

// ---------------------------------------------------------------------------
// OctreeBuilder
// ---------------------------------------------------------------------------

/// Map an input LAS point format ID (0–10) to a COPC-compatible output format (6, 7, or 8).
pub fn input_to_copc_format(id: u8) -> u8 {
    match id {
        2 | 3 | 5 | 7 => 7,
        8 | 10 => 8,
        _ => 6, // 0, 1, 4, 6, 9
    }
}

/// Per-file results from the scan phase, used by validation.
pub struct ScanResult {
    pub bounds: Bounds,
    pub point_count: u64,
    pub scale_x: f64,
    pub scale_y: f64,
    pub scale_z: f64,
    pub offset_x: f64,
    pub offset_y: f64,
    pub offset_z: f64,
    pub wkt_crs: Option<Vec<u8>>,
    pub point_format_id: u8,
}

/// Builds a COPC octree from scanned input files.
pub struct OctreeBuilder {
    /// Spatial bounds of all input points.
    pub bounds: Bounds,
    /// Total number of points across all input files.
    pub total_points: u64,
    /// Octree root center X.
    pub cx: f64,
    /// Octree root center Y.
    pub cy: f64,
    /// Octree root center Z.
    pub cz: f64,
    /// Half-size of the root voxel.
    pub halfsize: f64,
    /// X scale factor from the first input file.
    pub scale_x: f64,
    /// Y scale factor.
    pub scale_y: f64,
    /// Z scale factor.
    pub scale_z: f64,
    /// X offset.
    pub offset_x: f64,
    /// Y offset.
    pub offset_y: f64,
    /// Z offset.
    pub offset_z: f64,
    /// Temp directory where node files are written.
    pub tmp_dir: PathBuf,
    /// WKT CRS payload from the first input file (if present).
    pub wkt_crs: Option<Vec<u8>>,
    /// COPC output point format (6, 7, or 8), derived from input files.
    pub point_format: u8,
    /// Chunk plan computed by `distribute_chunked` and consumed by
    /// `build_node_map_chunked`. `None` for the per-leaf path.
    pub(crate) chunked_plan: Option<crate::chunking::ChunkPlan>,
    /// Compression codec applied to scratch temp files.
    pub(crate) temp_compression: crate::TempCompression,
}

impl OctreeBuilder {
    /// Pass 1: scan all files in parallel to get bounds and total point count.
    pub fn scan(input_files: &[PathBuf], config: &PipelineConfig) -> Result<Vec<ScanResult>> {
        let done = std::sync::atomic::AtomicU64::new(0);
        let results: Result<Vec<ScanResult>> = input_files
            .par_iter()
            .map(|path| -> Result<ScanResult> {
                debug!("Scanning {:?}", path);
                let reader = las::Reader::from_path(path)
                    .with_context(|| format!("Cannot open {:?}", path))?;
                let hdr = reader.header();
                let b = hdr.bounds();
                let mut bounds = Bounds::empty();
                bounds.expand_with(b.min.x, b.min.y, b.min.z);
                bounds.expand_with(b.max.x, b.max.y, b.max.z);
                let t = hdr.transforms();
                let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                config.report(crate::ProgressEvent::StageProgress { done: n });
                Ok(ScanResult {
                    bounds,
                    point_count: hdr.number_of_points(),
                    scale_x: t.x.scale,
                    scale_y: t.y.scale,
                    scale_z: t.z.scale,
                    offset_x: t.x.offset,
                    offset_y: t.y.offset,
                    offset_z: t.z.offset,
                    wkt_crs: hdr
                        .all_vlrs()
                        .find(|v| v.is_wkt_crs())
                        .map(|v| v.data.clone()),
                    point_format_id: hdr.point_format().to_u8().unwrap_or(0),
                })
            })
            .collect();
        results
    }

    /// Build an OctreeBuilder from scan results and validated inputs.
    pub fn from_scan(
        scan_results: &[ScanResult],
        validated: &crate::validate::ValidatedInputs,
        config: &PipelineConfig,
    ) -> Result<Self> {
        let mut bounds = Bounds::empty();
        let mut total_points = 0u64;
        for r in scan_results {
            bounds.merge(&r.bounds);
            total_points += r.point_count;
        }

        let first = &scan_results[0];
        let (scale_x, scale_y, scale_z) = (first.scale_x, first.scale_y, first.scale_z);
        let (offset_x, offset_y, offset_z) = (first.offset_x, first.offset_y, first.offset_z);

        let (cx, cy, cz, halfsize) = bounds.to_cube();

        // Choose depth so that leaf voxels hold ≤ MAX_LEAF_POINTS on average.
        let depth = {
            let mut d = 0u32;
            while (total_points as f64) / (8u64.pow(d) as f64) > MAX_LEAF_POINTS as f64 {
                d += 1;
                if d > 16 {
                    break;
                }
            }
            d.max(1)
        };
        debug!("Octree depth = {depth}, total points = {total_points}");

        let sys_tmp = std::env::temp_dir();
        let base_tmp = config.temp_dir.as_deref().unwrap_or(&sys_tmp);
        let tmp_dir = base_tmp.join(format!("copc_{}", std::process::id()));
        // Remove any leftover temp dir from a previous run (e.g. crashed pod
        // that never ran Drop cleanup). Without this, append-mode writes in
        // distribute/normalize would double-count points from the old run.
        if tmp_dir.exists() {
            info!("Removing stale temp dir {:?}", tmp_dir);
            std::fs::remove_dir_all(&tmp_dir)?;
        }
        std::fs::create_dir_all(&tmp_dir)?;

        Ok(OctreeBuilder {
            bounds,
            total_points,
            cx,
            cy,
            cz,
            halfsize,
            scale_x,
            scale_y,
            scale_z,
            offset_x,
            offset_y,
            offset_z,
            tmp_dir,
            wkt_crs: validated.wkt_crs.clone(),
            point_format: validated.point_format,
            chunked_plan: None,
            temp_compression: config.temp_compression,
        })
    }

    /// Path for a node's temp file.
    fn node_path(&self, key: &VoxelKey) -> PathBuf {
        self.tmp_dir
            .join(format!("{}_{}_{}_{}", key.level, key.x, key.y, key.z))
    }

    /// Convert a `las::Point` to a `RawPoint` using the builder's scale/offset.
    /// Read all raw points for a given node key from disk.
    pub fn read_node(&self, key: &VoxelKey) -> Result<Vec<RawPoint>> {
        let path = self.node_path(key);
        let f = match File::open(&path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(vec![]),
            Err(e) => return Err(e.into()),
        };
        read_temp_batches(f, self.temp_compression)
    }

    /// Write points to a temp file for the given node key (overwrites if
    /// exists). Always writes exactly one batch, even for an empty slice.
    pub fn write_node_to_temp(&self, key: &VoxelKey, points: &[RawPoint]) -> Result<()> {
        let path = self.node_path(key);
        let f = File::create(&path)?;
        let mut w = BufWriter::new(f);
        write_temp_batch(&mut w, points, self.temp_compression)?;
        w.flush().context("flush node temp file")?;
        Ok(())
    }

    /// Bottom-up grid-sample loop over an in-memory `nodes` HashMap.
    ///
    /// Walks levels `(min_level..actual_max_depth)` in reverse, grouping
    /// children of each parent and running `grid_sample` to produce that
    /// parent. After the loop completes, every node in the resulting map is
    /// written to its canonical temp file location.
    ///
    /// With `min_level = 0` the loop walks all the way to the global root
    /// (per-leaf path). With a positive `min_level` the loop stops once it
    /// has produced parents at `min_level`, leaving any coarser ancestors
    /// to a later merge step (chunked path: each chunk's bottom-up build
    /// stops at the chunk's root level).
    ///
    /// `report_progress` controls whether per-level `StageProgress` events
    /// are emitted. The per-leaf path passes `true` (the level loop is the
    /// main unit of progress). The chunked path passes `false` because its
    /// outer stage tracks chunks-done, not levels-done.
    fn bottom_up_levels(
        &self,
        mut nodes: HashMap<VoxelKey, Vec<RawPoint>>,
        actual_max_depth: u32,
        min_level: u32,
        report_progress: bool,
        config: &crate::PipelineConfig,
    ) -> Result<Vec<(VoxelKey, usize)>> {
        for d in (min_level..actual_max_depth).rev() {
            if report_progress {
                config.report(crate::ProgressEvent::StageProgress {
                    done: (actual_max_depth - d) as u64,
                });
            }
            let level_points: usize = nodes
                .iter()
                .filter(|(k, _)| k.level as u32 == d + 1)
                .map(|(_, v)| v.len())
                .sum();
            debug!(
                "In-memory level {d}: {} total points at child level, {} nodes in map ({} MB est)",
                level_points,
                nodes.len(),
                (nodes.values().map(|v| v.len()).sum::<usize>() * std::mem::size_of::<RawPoint>())
                    / 1_048_576,
            );

            // Group children at level d+1 by parent (iterate keys, no disk I/O).
            let mut parent_children: HashMap<VoxelKey, Vec<VoxelKey>> = HashMap::new();
            for k in nodes.keys() {
                if k.level as u32 == d + 1
                    && let Some(p) = k.parent()
                {
                    parent_children.entry(p).or_default().push(*k);
                }
            }
            if parent_children.is_empty() {
                continue;
            }

            // Remove children data from `nodes` and build owned tasks (no cloning).
            let tasks: Vec<SampleTask> = parent_children
                .into_iter()
                .map(|(parent, children)| {
                    let all_pts: Vec<(usize, RawPoint)> = children
                        .iter()
                        .enumerate()
                        .flat_map(|(ci, ck)| {
                            nodes
                                .remove(ck)
                                .unwrap_or_default()
                                .into_iter()
                                .map(move |p| (ci, p))
                        })
                        .collect();
                    (parent, children, all_pts)
                })
                .collect();

            // Grid-sample in parallel.
            let results: Vec<SampleResult> = tasks
                .into_par_iter()
                .map(|(parent, children, all_pts)| -> Result<_> {
                    if all_pts.is_empty() {
                        let n = children.len();
                        return Ok((parent, children, vec![], vec![vec![]; n]));
                    }
                    let n = children.len();
                    let (parent_pts, remaining) = self.grid_sample(&parent, all_pts, n);
                    Ok((parent, children, parent_pts, remaining))
                })
                .collect::<Result<_>>()?;

            // Apply updates to `nodes` (sequential, needs &mut).
            for (parent, children, parent_pts, remaining) in results {
                for (ck, rem) in children.into_iter().zip(remaining.into_iter()) {
                    if !rem.is_empty() {
                        nodes.insert(ck, rem);
                    }
                }
                if !parent_pts.is_empty() {
                    nodes.insert(parent, parent_pts);
                }
            }
        }

        // Write final nodes to disk for the writer.
        nodes
            .par_iter()
            .map(|(k, pts)| -> Result<()> { self.write_node_to_temp(k, pts) })
            .collect::<Result<Vec<_>>>()?;

        Ok(nodes
            .iter()
            .filter(|(_, pts)| !pts.is_empty())
            .map(|(k, pts)| (*k, pts.len()))
            .collect())
    }

    /// Streaming grid-sample for a single oversized parent.
    ///
    /// Used as a fallback by `merge_chunk_tops` when a parent group's combined
    /// child points exceed the memory budget. Processes one child at a time so
    /// only one child's points are ever in memory. The grid occupancy set
    /// (max 128³ entries ≈ 48 MB) and the parent accumulator (max 128³ points
    /// ≈ 80 MB) stay resident; everything else is streamed from/to disk.
    ///
    /// In practice this path is rarely hit for well-formed chunk plans
    /// (merge-sparse-cells sizes chunks to fit the budget by construction),
    /// but it's kept as a safety net for degenerate inputs.
    ///
    /// Tradeoff: points are not Morton-sorted across children, so spatial
    /// coherence of the parent is slightly worse than in-memory grid_sample.
    fn grid_sample_streaming(&self, parent: &VoxelKey, children: &[VoxelKey]) -> Result<()> {
        let voxel_size_world = 2.0 * self.halfsize / (1u64 << parent.level) as f64;
        let origin_x = ((self.cx - self.halfsize + parent.x as f64 * voxel_size_world
            - self.offset_x)
            / self.scale_x)
            .round() as i64;
        let origin_y = ((self.cy - self.halfsize + parent.y as f64 * voxel_size_world
            - self.offset_y)
            / self.scale_y)
            .round() as i64;
        let origin_z = ((self.cz - self.halfsize + parent.z as f64 * voxel_size_world
            - self.offset_z)
            / self.scale_z)
            .round() as i64;
        let int_size =
            (voxel_size_world / self.scale_x.min(self.scale_y).min(self.scale_z)).round() as i64;
        let cell = (int_size / GRID_CELLS_PER_AXIS).max(1);

        let grid_key = |p: &RawPoint| -> (i32, i32, i32) {
            (
                ((p.x as i64 - origin_x) / cell) as i32,
                ((p.y as i64 - origin_y) / cell) as i32,
                ((p.z as i64 - origin_z) / cell) as i32,
            )
        };

        let max_accepted =
            (GRID_CELLS_PER_AXIS * GRID_CELLS_PER_AXIS * GRID_CELLS_PER_AXIS) as usize;
        let mut occupied: HashSet<(i32, i32, i32)> = HashSet::new();
        let mut parent_pts: Vec<RawPoint> = Vec::new();

        // Track which children had points, to guarantee at least one per child.
        let mut child_donated: Vec<bool> = vec![false; children.len()];

        // Pass 1: iterate children one at a time, accept points into the parent
        // grid, write remaining back to the child's temp file.
        for (ci, ck) in children.iter().enumerate() {
            let pts = self.read_node(ck)?;
            if pts.is_empty() {
                continue;
            }
            debug!(
                "    streaming child {}/{}: {:?} → {} points ({} MB), grid {}/{} full",
                ci + 1,
                children.len(),
                ck,
                pts.len(),
                (pts.len() * std::mem::size_of::<RawPoint>()) / 1_048_576,
                occupied.len(),
                max_accepted,
            );
            let mut remaining: Vec<RawPoint> = Vec::new();
            for p in pts {
                if parent_pts.len() < max_accepted && occupied.insert(grid_key(&p)) {
                    child_donated[ci] = true;
                    parent_pts.push(p);
                } else {
                    remaining.push(p);
                }
            }
            // If this child had points but none remain (all promoted), we need to
            // return at least one. Done in pass 2 below.
            if remaining.is_empty() && child_donated[ci] {
                // Will be fixed in pass 2.
                child_donated[ci] = false;
            }
            self.write_node_to_temp(ck, &remaining)?;
        }

        // Pass 2: guarantee every child that contributed keeps at least one point.
        for (ci, ck) in children.iter().enumerate() {
            if !child_donated[ci] {
                // Check if this child originally had points (file might now be
                // empty because all were promoted).
                let n =
                    count_temp_file_points(&self.node_path(ck), self.temp_compression).unwrap_or(0);
                if n == 0 && !parent_pts.is_empty() {
                    // Return one point from the parent back to this child.
                    // Find the last parent point that came from any child — we
                    // can't track origin in the streaming path, so just pop the last.
                    if let Some(p) = parent_pts.pop() {
                        self.write_node_to_temp(ck, &[p])?;
                    }
                }
            }
        }

        if !parent_pts.is_empty() {
            self.write_node_to_temp(parent, &parent_pts)?;
        }

        Ok(())
    }

    /// Grid-based spatial sampling for one parent node.
    ///
    /// Divides the parent voxel into a uniform grid of GRID_CELLS_PER_AXIS³ cells.
    /// Points are sorted by Morton code and iterated in that order; the first point
    /// that falls into each unoccupied cell is accepted for the parent.  All others
    /// are returned to their originating child so every point lands in exactly one node.
    ///
    /// This mirrors untwine's approach and produces spatially homogeneous LOD levels.
    fn grid_sample(
        &self,
        parent: &VoxelKey,
        mut pts: Vec<(usize, RawPoint)>, // takes ownership — no cloning
        n_children: usize,
    ) -> (Vec<RawPoint>, Vec<Vec<RawPoint>>) {
        if pts.is_empty() {
            return (vec![], vec![vec![]; n_children]);
        }

        // Parent voxel geometry in integer coordinate space.
        let voxel_size_world = 2.0 * self.halfsize / (1u64 << parent.level) as f64;
        let origin_x = ((self.cx - self.halfsize + parent.x as f64 * voxel_size_world
            - self.offset_x)
            / self.scale_x)
            .round() as i64;
        let origin_y = ((self.cy - self.halfsize + parent.y as f64 * voxel_size_world
            - self.offset_y)
            / self.scale_y)
            .round() as i64;
        let origin_z = ((self.cz - self.halfsize + parent.z as f64 * voxel_size_world
            - self.offset_z)
            / self.scale_z)
            .round() as i64;
        let int_size =
            (voxel_size_world / self.scale_x.min(self.scale_y).min(self.scale_z)).round() as i64;

        // Grid resolution: fixed cells per axis, matching untwine's CellCount.
        let cell = (int_size / GRID_CELLS_PER_AXIS).max(1);

        // Sort by Morton code within the parent voxel for spatially coherent traversal.
        pts.sort_unstable_by_key(|(_, p)| {
            let dx = (p.x as i64 - origin_x).max(0) as u32;
            let dy = (p.y as i64 - origin_y).max(0) as u32;
            let dz = (p.z as i64 - origin_z).max(0) as u32;
            morton3(dx, dy, dz)
        });

        let grid_key = |p: &RawPoint| -> (i32, i32, i32) {
            (
                ((p.x as i64 - origin_x) / cell) as i32,
                ((p.y as i64 - origin_y) / cell) as i32,
                ((p.z as i64 - origin_z) / cell) as i32,
            )
        };

        // Track which children actually have points so we can protect them.
        let mut child_has_pts = vec![false; n_children];
        for (ci, _) in &pts {
            child_has_pts[*ci] = true;
        }

        // Partition: accepted for parent vs remaining for children. No cloning.
        let mut occupied: HashSet<(i32, i32, i32)> = HashSet::new();
        let max_accepted =
            (GRID_CELLS_PER_AXIS * GRID_CELLS_PER_AXIS * GRID_CELLS_PER_AXIS) as usize;
        let mut parent_pts: Vec<(usize, RawPoint)> = Vec::with_capacity(max_accepted);
        let mut remaining: Vec<Vec<RawPoint>> = vec![Vec::new(); n_children];

        for (ci, p) in pts {
            if parent_pts.len() < max_accepted && occupied.insert(grid_key(&p)) {
                parent_pts.push((ci, p));
            } else {
                remaining[ci].push(p);
            }
        }

        // Guarantee every child that contributed points keeps at least one.
        // This prevents zero-point intermediate nodes in the COPC hierarchy
        // (which confuse validators that check point_count > 0 for all entries).
        for ci in 0..n_children {
            if child_has_pts[ci]
                && remaining[ci].is_empty()
                && let Some(pos) = parent_pts.iter().rposition(|(c, _)| *c == ci)
            {
                let (_, p) = parent_pts.remove(pos);
                remaining[ci].push(p);
            }
        }

        let parent_pts = parent_pts.into_iter().map(|(_, p)| p).collect();
        (parent_pts, remaining)
    }

    /// Chunked distribute. Computes the chunk plan via the counting-sort
    /// analyzer, builds a fast cell→chunk lookup table, then streams each
    /// input file in parallel and appends each point to its chunk's temp
    /// file via a bounded LRU writer cache.
    pub fn distribute(&mut self, input_files: &[PathBuf], config: &PipelineConfig) -> Result<()> {
        // 1. Compute the chunk plan (counting + merge-sparse-cells). This
        //    runs a full pass over the input to populate an occupancy grid,
        //    which is its own user-visible stage.
        //    `chunk_target_override` lets tests force multi-chunk plans on
        //    small inputs; None = dynamic derivation from memory_budget.
        config.report(crate::ProgressEvent::StageStart {
            name: "Counting",
            total: self.total_points,
        });
        let plan = crate::chunking::compute_chunk_plan(
            self,
            input_files,
            config,
            config.chunk_target_override,
        )?;
        config.report(crate::ProgressEvent::StageDone);
        info!(
            "Chunked distribute: {} chunks, target {} points each, grid {}³",
            plan.chunks.len(),
            plan.chunk_target,
            plan.grid_size
        );

        // 2. Build the cell → chunk_index lookup table. The grid has
        //    grid_size³ fine cells; for each chunk we fill in every fine cell
        //    it covers (a 2^(grid_depth - chunk.level) cube per axis).
        let lut = build_chunk_lut(&plan);

        // 3. Set up shard subdirectories under tmp_dir/chunks/shards/{worker}.
        let chunks_root = self.tmp_dir.join("chunks");
        if chunks_root.exists() {
            std::fs::remove_dir_all(&chunks_root)
                .with_context(|| format!("removing stale chunks dir {:?}", chunks_root))?;
        }
        std::fs::create_dir_all(&chunks_root)?;
        let shards_root = chunks_root.join("shards");
        std::fs::create_dir_all(&shards_root)?;

        let num_workers = chunked_distribute_worker_count(input_files.len());
        let shard_dirs: Vec<PathBuf> = (0..num_workers)
            .map(|w| {
                let d = shards_root.join(w.to_string());
                std::fs::create_dir_all(&d)?;
                Ok(d)
            })
            .collect::<Result<Vec<_>>>()?;

        // 4. Compute per-worker resources.
        let per_worker_cap = (CHUNKED_OPEN_FILES_CAP / num_workers).max(MIN_PER_WORKER_CHUNK_FILES);

        // Per-worker memory budget for the input chunk size. The transient
        // peak is `Vec<las::Point>` (~120 B/pt) plus the cache's BufWriter
        // overhead (negligible). Size for 1/8 of the per-worker budget so
        // there's headroom for the LAZ decoder + grouping overhead.
        const BYTES_PER_POINT_TRANSIENT: u64 = 120;
        let per_worker_budget = config.memory_budget / num_workers as u64;
        let read_chunk_size =
            ((per_worker_budget / 8) / BYTES_PER_POINT_TRANSIENT).clamp(50_000, 2_000_000) as usize;

        debug!(
            "Chunked distribute: budget={} MB, workers={}, read_chunk_size={}, open-file cap={} (per worker)",
            config.memory_budget / 1_048_576,
            num_workers,
            read_chunk_size,
            per_worker_cap
        );

        // Grid geometry for cell index computation. We classify points via
        // `point_to_key(..., grid_depth)` — the same octree descent that the
        // per-chunk build uses via `point_to_key(..., leaf_depth)`. This
        // guarantees bit-for-bit consistency between the LUT classification
        // and the build-phase classification, so a point's chunk assignment
        // and its leaf's chunk-root ancestor always agree. An earlier version
        // used inlined floor-based math for speed (~3× faster), but
        // floating-point precision at cell boundaries made it disagree with
        // `point_to_key` on a tiny fraction of points (~0.13%), and those
        // points ended up written to the wrong chunk where they were silently
        // lost during build.
        let g = plan.grid_size;
        let grid_depth = plan.grid_depth;
        let g_usize = g as usize;

        // Capture the LUT and per-worker file ranges for the parallel block.
        let files_per_worker = input_files.len().div_ceil(num_workers);
        let progress = AtomicU64::new(0);

        // 5. Partition pass — stream every point into its chunk's shard
        //    file. This is the second full pass over the input and is
        //    reported as its own `Distributing` stage.
        config.report(crate::ProgressEvent::StageStart {
            name: "Distributing",
            total: self.total_points,
        });

        // 5. Parallel workers stream input files and append points to chunks.
        shard_dirs
            .par_iter()
            .enumerate()
            .try_for_each(|(worker_id, shard_dir)| -> Result<()> {
                let start = worker_id * files_per_worker;
                let end = (start + files_per_worker).min(input_files.len());
                if start >= end {
                    return Ok(());
                }

                let mut cache = ChunkWriterCache::new(per_worker_cap, self.temp_compression);
                // Reusable scratch buffer for decompressed raw points in
                // *file frame*. Reprojected to the builder frame below.
                let mut raw_points: Vec<RawPoint> = Vec::with_capacity(read_chunk_size);
                // Group buffer reused across batches so capacity amortizes.
                // Drained at the end of each batch so retained Vec capacity
                // stays bounded by the working set, not by all-time peak.
                let mut groups: HashMap<u32, Vec<RawPoint>> = HashMap::new();

                for path in &input_files[start..end] {
                    let meta = crate::laz_read::LazFileMeta::read(path)?;
                    let file_pts = meta.n_points;
                    let (fsx, fsy, fsz) = meta.scale;
                    let (fox, foy, foz) = meta.offset;
                    // True when the file and the builder share the exact
                    // same scale/offset, in which case the file-frame i32
                    // coordinates are already the builder-frame values.
                    let same_frame = fsx == self.scale_x
                        && fsy == self.scale_y
                        && fsz == self.scale_z
                        && fox == self.offset_x
                        && foy == self.offset_y
                        && foz == self.offset_z;
                    let mut reader = crate::laz_read::RawLazReader::open_full(path, meta)?;
                    debug!(
                        "Chunked distribute worker {} file {:?}: {} points",
                        worker_id,
                        path.file_name().unwrap_or_default(),
                        file_pts
                    );

                    loop {
                        raw_points.clear();
                        let n = reader.read_full_into(&mut raw_points, read_chunk_size)?;
                        if n == 0 {
                            break;
                        }

                        // Classify points into chunks via the LUT and group
                        // them in memory before writing. Grouping batches
                        // many points per writer call → fewer LRU touches
                        // and better amortization than per-point appends.
                        for raw in raw_points.drain(..) {
                            // Reproject file-frame scaled ints into the
                            // builder frame. When the file and builder
                            // share scale/offset (the common single-file
                            // case), this is a no-op.
                            let mut raw = raw;
                            if !same_frame {
                                let wx = raw.x as f64 * fsx + fox;
                                let wy = raw.y as f64 * fsy + foy;
                                let wz = raw.z as f64 * fsz + foz;
                                raw.x = ((wx - self.offset_x) / self.scale_x).round() as i32;
                                raw.y = ((wy - self.offset_y) / self.scale_y).round() as i32;
                                raw.z = ((wz - self.offset_z) / self.scale_z).round() as i32;
                            }

                            // Classify to grid cell via point_to_key at the
                            // grid depth, using round-tripped world coords.
                            // This is the same function the per-chunk build
                            // will call (at leaf_depth, which is chunk.level
                            // + extra_levels); using it here guarantees the
                            // chunk-root ancestor of every leaf produced
                            // during build matches the chunk assignment made
                            // during distribute — no boundary-point loss.
                            let wx = raw.x as f64 * self.scale_x + self.offset_x;
                            let wy = raw.y as f64 * self.scale_y + self.offset_y;
                            let wz = raw.z as f64 * self.scale_z + self.offset_z;
                            let key = point_to_key(
                                wx,
                                wy,
                                wz,
                                self.cx,
                                self.cy,
                                self.cz,
                                self.halfsize,
                                grid_depth,
                            );
                            // Defensive clamp for points exactly on the
                            // upper boundary that round out by one.
                            let gx = (key.x as usize).min(g_usize - 1);
                            let gy = (key.y as usize).min(g_usize - 1);
                            let gz = (key.z as usize).min(g_usize - 1);

                            let cell_idx = gx + gy * g_usize + gz * g_usize * g_usize;
                            let chunk_idx = lut[cell_idx];
                            groups.entry(chunk_idx).or_default().push(raw);
                        }

                        // Flush each group to the cache. Drain so vec capacity
                        // doesn't accumulate across batches.
                        for (chunk_idx, pts) in groups.drain() {
                            cache.append(chunk_idx, shard_dir, &pts)?;
                        }

                        let done = progress.fetch_add(n as u64, Ordering::Relaxed) + n as u64;
                        config.report(crate::ProgressEvent::StageProgress { done });
                    }
                }

                cache.flush_all()
            })?;

        // 6. Merge per-worker shards into canonical chunk files.
        merge_chunk_shards(&shards_root, &chunks_root, plan.chunks.len() as u32)?;
        config.report(crate::ProgressEvent::StageDone);

        // 7. Stash the chunk plan for the build phase to consume.
        self.chunked_plan = Some(plan);
        Ok(())
    }

    /// Read all `RawPoint`s from a chunk file in one shot.
    fn read_chunk_file(&self, chunk_idx: u32) -> Result<Vec<RawPoint>> {
        let path = chunk_canonical_path(&self.tmp_dir.join("chunks"), chunk_idx);
        let f = File::open(&path).with_context(|| format!("opening chunk file {:?}", path))?;
        read_temp_batches(f, self.temp_compression)
    }

    /// Build a single chunk's sub-octree fully in memory and write its node
    /// files to canonical temp paths.
    ///
    /// Returns the list of `(VoxelKey, point_count)` for nodes at levels in
    /// `[chunk.level, chunk_leaf_depth]` that the chunk produced. The chunk
    /// root at `chunk.level` is included; coarser ancestors are left to the
    /// merge step.
    fn build_chunk_in_memory(
        &self,
        chunk: &crate::chunking::PlannedChunk,
        chunk_idx: u32,
        config: &crate::PipelineConfig,
    ) -> Result<Vec<(VoxelKey, usize)>> {
        // Load all points for this chunk into memory.
        let points = self.read_chunk_file(chunk_idx)?;
        let n_points = points.len();

        if n_points == 0 {
            // Empty chunk → no nodes produced. Should not happen for a plan
            // generated by `merge_sparse_cells`, but handle defensively.
            debug!(
                "Chunk {} (L{} {},{},{}) is empty, skipping",
                chunk_idx, chunk.level, chunk.gx, chunk.gy, chunk.gz
            );
            return Ok(Vec::new());
        }

        // Compute the chunk-local leaf depth so each leaf holds ~MAX_LEAF_POINTS
        // on average. Slightly larger leaves are acceptable in dense regions.
        let extra_levels = chunk_local_extra_levels(n_points as u64);
        let leaf_depth = chunk.level + extra_levels;
        debug!(
            "Chunk {} (L{} {},{},{}): {} points → leaf depth {} (extra {})",
            chunk_idx,
            chunk.level,
            chunk.gx,
            chunk.gy,
            chunk.gz,
            n_points,
            leaf_depth,
            extra_levels
        );

        // Classify each point into its leaf voxel at the global leaf depth.
        // This uses point_to_key on round-tripped world coordinates so the
        // voxel assignment matches what the per-leaf path would compute.
        let mut leaves: HashMap<VoxelKey, Vec<RawPoint>> = HashMap::new();
        for raw in points {
            let wx = raw.x as f64 * self.scale_x + self.offset_x;
            let wy = raw.y as f64 * self.scale_y + self.offset_y;
            let wz = raw.z as f64 * self.scale_z + self.offset_z;
            let key = point_to_key(
                wx,
                wy,
                wz,
                self.cx,
                self.cy,
                self.cz,
                self.halfsize,
                leaf_depth,
            );

            // Debug-only defensive check: the leaf key's ancestor at
            // chunk.level must equal the chunk root. If it doesn't, the
            // chunked distribute classified this point into the wrong
            // chunk. Distribute and build must use the same key derivation
            // (`point_to_key`) to stay consistent — if this assertion ever
            // fires, the classification has drifted again and points will
            // be silently lost unless we reject here.
            #[cfg(debug_assertions)]
            {
                let mut ancestor = key;
                while ancestor.level > chunk.level as i32 {
                    ancestor = ancestor.parent().expect("leaf above chunk level");
                }
                debug_assert!(
                    ancestor.level == chunk.level as i32
                        && ancestor.x == chunk.gx
                        && ancestor.y == chunk.gy
                        && ancestor.z == chunk.gz,
                    "leaf {:?} does not belong to chunk root (L{} {},{},{}): ancestor at L{} = ({},{},{})",
                    key,
                    chunk.level,
                    chunk.gx,
                    chunk.gy,
                    chunk.gz,
                    ancestor.level,
                    ancestor.x,
                    ancestor.y,
                    ancestor.z,
                );
            }

            leaves.entry(key).or_default().push(raw);
        }

        // Adaptive leaf subdivision. The uniform `leaf_depth` computed from
        // total point count produces leaves that can be 1.4–2× MAX_LEAF_POINTS
        // in dense regions. Repeatedly split any leaf that exceeds
        // MAX_LEAF_POINTS into 8 children at one level deeper, until no leaf
        // is oversized or a depth cap is hit.
        //
        // Collisions are impossible because newly-created children are always
        // at a deeper level than any existing leaf, so their keys can't clash.
        const CHUNK_DEPTH_CAP: u32 = 20; // same spirit as from_scan's `d > 16` cap
        let mut effective_max_depth = leaf_depth;
        loop {
            let oversized: Vec<VoxelKey> = leaves
                .iter()
                .filter(|(_, pts)| pts.len() as u64 > MAX_LEAF_POINTS)
                .map(|(k, _)| *k)
                .collect();
            if oversized.is_empty() {
                break;
            }
            // Safety cap: if any oversized leaf is already at the depth cap,
            // give up further subdivision and accept the residual overflow.
            // In practice this means some leaves in pathologically dense
            // spots (coincident points, volumetric data) will exceed the
            // target — an explicit trade-off per design §9.3.
            let hit_cap = oversized.iter().any(|k| k.level as u32 >= CHUNK_DEPTH_CAP);
            if hit_cap {
                debug!(
                    "Chunk {}: stopping subdivision at depth cap {} \
                     ({} leaves still oversized)",
                    chunk_idx,
                    CHUNK_DEPTH_CAP,
                    oversized.len()
                );
                break;
            }
            for key in oversized {
                let pts = leaves.remove(&key).expect("just listed");
                let new_level = (key.level as u32) + 1;
                if new_level > effective_max_depth {
                    effective_max_depth = new_level;
                }
                for raw in pts {
                    let wx = raw.x as f64 * self.scale_x + self.offset_x;
                    let wy = raw.y as f64 * self.scale_y + self.offset_y;
                    let wz = raw.z as f64 * self.scale_z + self.offset_z;
                    let new_key = point_to_key(
                        wx,
                        wy,
                        wz,
                        self.cx,
                        self.cy,
                        self.cz,
                        self.halfsize,
                        new_level,
                    );
                    leaves.entry(new_key).or_default().push(raw);
                }
            }
        }

        debug!(
            "Chunk {}: {} leaves after subdivision, effective depth {}",
            chunk_idx,
            leaves.len(),
            effective_max_depth
        );

        // Run the bottom-up grid-sample loop, stopping at chunk.level.
        // bottom_up_levels writes every produced node to its canonical temp
        // file path, so the chunk's sub-octree is on disk after this returns.
        // Suppress per-level progress events: the chunked-build outer stage
        // tracks chunks-done, not levels.
        self.bottom_up_levels(leaves, effective_max_depth, chunk.level, false, config)
    }

    /// Merge chunk roots upward from `max_chunk_level` to the global root.
    ///
    /// This is the chunked-build equivalent of the per-leaf path's coarse
    /// `bottom_up_on_disk` levels. Starting from a set of "chunk root" keys
    /// (one per chunk, at varying levels), we walk levels in reverse and at
    /// each level group nodes by their parent and run `grid_sample` to
    /// produce the parent.
    ///
    /// **Variable-level chunks are handled naturally**: when level `d`'s
    /// children are processed, the set may include both chunk roots that
    /// happened to be at level `d+1` AND parents that the merge produced
    /// from level `d+2`. They're treated identically — they're all just
    /// "nodes at level `d+1`, ready to be grouped by their level-`d` parent".
    ///
    /// Modifies node files on disk: when a parent is produced from its
    /// children, the children's files are rewritten with the points that
    /// did NOT get promoted to the parent (`grid_sample` returns this
    /// "remaining" set per child).
    ///
    /// `chunk_root_keys` lists every chunk root produced by the per-chunk
    /// build phase. Returns the merge-produced parent keys at every level
    /// from 0 up to `max_chunk_level - 1`.
    fn merge_chunk_tops(
        &self,
        chunk_root_keys: &[VoxelKey],
        config: &crate::PipelineConfig,
    ) -> Result<Vec<VoxelKey>> {
        if chunk_root_keys.is_empty() {
            return Ok(Vec::new());
        }

        // Organise chunk roots by level. Like bottom_up_on_disk, we use a
        // HashSet per level so we never get duplicate keys (which can happen
        // if two chunks somehow share the same root key — should not occur
        // for a well-formed plan, but defensive).
        let mut keys_by_level: HashMap<i32, HashSet<VoxelKey>> = HashMap::new();
        for k in chunk_root_keys {
            keys_by_level.entry(k.level).or_default().insert(*k);
        }

        let max_chunk_level: i32 = keys_by_level.keys().copied().max().unwrap_or(0);
        debug!(
            "Merge chunk tops: {} chunk roots, max_chunk_level={}",
            chunk_root_keys.len(),
            max_chunk_level
        );

        // Reuse the same per-point cost estimate as bottom_up_on_disk's
        // small-parent batching: input vec + per-child remaining.
        const MEM_PER_POINT: u64 =
            (std::mem::size_of::<(usize, RawPoint)>() + std::mem::size_of::<RawPoint>()) as u64;

        let mut all_new_parents: Vec<VoxelKey> = Vec::new();

        // Walk from the deepest chunk level down to level 0. d is the parent
        // level we're producing in this iteration.
        for d in (0..max_chunk_level).rev() {
            let child_level = d + 1;
            let child_keys: Vec<VoxelKey> = match keys_by_level.get(&child_level) {
                Some(v) => v.iter().copied().collect(),
                None => continue,
            };
            if child_keys.is_empty() {
                continue;
            }

            // Group children at level d+1 by their parent at level d.
            let mut parent_children: HashMap<VoxelKey, Vec<VoxelKey>> = HashMap::new();
            for ck in &child_keys {
                if let Some(parent) = ck.parent() {
                    parent_children.entry(parent).or_default().push(*ck);
                }
            }
            if parent_children.is_empty() {
                continue;
            }

            // Estimate per-parent memory cost from the children's file sizes
            // and split into "small" (fit in budget) vs "large" (don't).
            // For chunked merge, large parents shouldn't happen because
            // chunks are sized to fit, but be defensive.
            let mut small_parents: Vec<(VoxelKey, Vec<VoxelKey>, u64)> = Vec::new();
            let mut large_parents: Vec<(VoxelKey, Vec<VoxelKey>, u64)> = Vec::new();

            for (parent, children) in parent_children {
                let est_points: u64 = children
                    .iter()
                    .map(|ck| {
                        count_temp_file_points(&self.node_path(ck), self.temp_compression)
                            .unwrap_or(0)
                    })
                    .sum();
                let est_mem = est_points * MEM_PER_POINT;
                if est_mem > config.memory_budget {
                    large_parents.push((parent, children, est_mem));
                } else {
                    small_parents.push((parent, children, est_mem));
                }
            }

            debug!(
                "Merge level {}→{}: {} parents ({} small, {} large), budget={} MB",
                child_level,
                d,
                small_parents.len() + large_parents.len(),
                small_parents.len(),
                large_parents.len(),
                config.memory_budget / 1_048_576,
            );

            if !large_parents.is_empty() {
                // Should not happen with a well-formed chunk plan, but if it
                // does, fall back to grid_sample_streaming which is bounded
                // (one child at a time, ~80 MB per parent for the grid set
                // and accumulator).
                for (parent, children, est_mem) in &large_parents {
                    debug!(
                        "  STREAMING merge parent {:?}: {} children, est {} MB",
                        parent,
                        children.len(),
                        est_mem / 1_048_576
                    );
                    self.grid_sample_streaming(parent, children)?;
                    all_new_parents.push(*parent);
                    keys_by_level.entry(d).or_default().insert(*parent);
                }
            }

            // Sort small parents by descending estimated memory so the
            // batching greedy stays balanced.
            small_parents.sort_by(|a, b| b.2.cmp(&a.2));

            let mut batch_start = 0;
            while batch_start < small_parents.len() {
                let mut batch_mem: u64 = 0;
                let mut batch_end = batch_start;
                while batch_end < small_parents.len() {
                    if batch_end > batch_start
                        && batch_mem + small_parents[batch_end].2 > config.memory_budget
                    {
                        break;
                    }
                    batch_mem += small_parents[batch_end].2;
                    batch_end += 1;
                }

                let batch = &small_parents[batch_start..batch_end];
                debug!(
                    "  Merge batch: {} parents, est {} MB",
                    batch.len(),
                    batch_mem / 1_048_576,
                );
                batch
                    .par_iter()
                    .map(|(parent, children, _)| -> Result<()> {
                        let mut all_pts: Vec<(usize, RawPoint)> = Vec::new();
                        for (ci, ck) in children.iter().enumerate() {
                            for p in self.read_node(ck)? {
                                all_pts.push((ci, p));
                            }
                        }
                        if all_pts.is_empty() {
                            return Ok(());
                        }
                        let (parent_pts, per_child) =
                            self.grid_sample(parent, all_pts, children.len());
                        // Rewrite each child with its remaining points.
                        for (ci, ck) in children.iter().enumerate() {
                            self.write_node_to_temp(ck, &per_child[ci])?;
                        }
                        if !parent_pts.is_empty() {
                            self.write_node_to_temp(parent, &parent_pts)?;
                        }
                        Ok(())
                    })
                    .collect::<Result<Vec<_>>>()?;

                for (parent, _, _) in &small_parents[batch_start..batch_end] {
                    all_new_parents.push(*parent);
                    keys_by_level.entry(d).or_default().insert(*parent);
                }
                batch_start = batch_end;
            }

            // Report progress: one tick per merged level (the caller knows
            // the total via max_chunk_level).
            config.report(crate::ProgressEvent::StageProgress {
                done: (max_chunk_level - d) as u64,
            });
        }

        Ok(all_new_parents)
    }

    /// Chunked build. Per-chunk in-memory build, then merge across chunks
    /// at coarse levels up to the global root.
    pub fn build_node_map(&self, config: &crate::PipelineConfig) -> Result<Vec<(VoxelKey, usize)>> {
        let plan = self
            .chunked_plan
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("build_node_map_chunked called without a chunk plan"))?;

        let n_chunks = plan.chunks.len();
        if n_chunks == 0 {
            info!("Chunked build: no chunks (empty input?)");
            return Ok(Vec::new());
        }

        info!(
            "Chunked build: {} chunks, max_chunk_level={}",
            n_chunks,
            plan.chunks.iter().map(|c| c.level).max().unwrap_or(0)
        );

        // ---- Phase 1: per-chunk in-memory build ----
        //
        // Process chunks in batches sized by memory budget. Within each batch,
        // chunks run fully in parallel via rayon. Per-chunk peak working set
        // estimate: ~200 B/pt (input vec + leaves HashMap + grid_sample
        // input/output + scratch). Conservative; matches design §9.4.
        const PER_CHUNK_BYTES_PER_POINT: u64 = 200;

        // Sort chunks by descending point count so the greedy batching stays
        // balanced (largest chunks first, smaller ones fill the gaps).
        let mut chunks_indexed: Vec<(u32, &crate::chunking::PlannedChunk)> = plan
            .chunks
            .iter()
            .enumerate()
            .map(|(i, c)| (i as u32, c))
            .collect();
        chunks_indexed.sort_by(|a, b| b.1.point_count.cmp(&a.1.point_count));

        config.report(crate::ProgressEvent::StageStart {
            name: "Building",
            total: n_chunks as u64,
        });

        let chunks_done = AtomicU64::new(0);
        let mut all_chunk_node_keys: Vec<VoxelKey> = Vec::new();
        let mut chunk_root_keys: Vec<VoxelKey> = Vec::new();

        let mut batch_start = 0;
        while batch_start < chunks_indexed.len() {
            let mut batch_mem: u64 = 0;
            let mut batch_end = batch_start;
            while batch_end < chunks_indexed.len() {
                let est_mem = chunks_indexed[batch_end].1.point_count * PER_CHUNK_BYTES_PER_POINT;
                // Always include at least one chunk per batch, even if it
                // exceeds the budget on its own (better to OOM honestly than
                // to hang forever in an empty batch).
                if batch_end > batch_start && batch_mem + est_mem > config.memory_budget {
                    break;
                }
                batch_mem += est_mem;
                batch_end += 1;
            }

            let batch = &chunks_indexed[batch_start..batch_end];
            debug!(
                "Chunked build batch: {} chunks, est {} MB",
                batch.len(),
                batch_mem / 1_048_576
            );

            // Process this batch in parallel.
            let batch_results: Vec<Vec<(VoxelKey, usize)>> = batch
                .par_iter()
                .map(|(chunk_idx, chunk)| -> Result<Vec<(VoxelKey, usize)>> {
                    let nodes = self.build_chunk_in_memory(chunk, *chunk_idx, config)?;
                    let done = chunks_done.fetch_add(1, Ordering::Relaxed) + 1;
                    config.report(crate::ProgressEvent::StageProgress { done });
                    Ok(nodes)
                })
                .collect::<Result<_>>()?;

            // Collect node keys produced by this batch.
            for ((_chunk_idx, chunk), nodes) in batch.iter().zip(batch_results.into_iter()) {
                let chunk_level_i32 = chunk.level as i32;
                for (k, _) in &nodes {
                    all_chunk_node_keys.push(*k);
                    if k.level == chunk_level_i32 {
                        chunk_root_keys.push(*k);
                    }
                }
            }

            batch_start = batch_end;
        }

        debug!(
            "Per-chunk build done: {} chunk nodes, {} chunk roots",
            all_chunk_node_keys.len(),
            chunk_root_keys.len()
        );

        // ---- Phase 2: merge chunk tops up to global root ----
        let merge_parents = self.merge_chunk_tops(&chunk_root_keys, config)?;
        debug!("Merge step produced {} parent nodes", merge_parents.len());

        config.report(crate::ProgressEvent::StageDone);

        // ---- Phase 3: assemble the result list ----
        //
        // The result is the union of per-chunk nodes + merge parents.
        // After the merge step, chunk root files have been rewritten with
        // their post-merge point counts (the points that did NOT get
        // promoted), so we re-read counts from disk for *every* node to
        // ensure we report the post-merge state, not the per-chunk-build
        // intermediate state.
        let mut all_keys: Vec<VoxelKey> = all_chunk_node_keys;
        all_keys.extend(merge_parents);
        // Deduplicate (a chunk root that's also at level 0 would be counted
        // twice — should not occur in practice but defensive).
        all_keys.sort_unstable_by_key(|k| (k.level, k.x, k.y, k.z));
        all_keys.dedup();

        let mut result: Vec<(VoxelKey, usize)> = all_keys
            .par_iter()
            .map(|k| -> Result<(VoxelKey, usize)> {
                let n = count_temp_file_points(&self.node_path(k), self.temp_compression)? as usize;
                Ok((*k, n))
            })
            .collect::<Result<Vec<_>>>()?;

        // Drop empty nodes (e.g. chunk roots whose points were all promoted
        // by the merge and never had any remaining points written back).
        result.retain(|(_, count)| *count > 0);

        let total_pts: usize = result.iter().map(|(_, c)| *c).sum();
        info!(
            "Chunked build: {} nodes, {} total points (input: {})",
            result.len(),
            total_pts,
            self.total_points
        );
        if total_pts as u64 != self.total_points {
            debug!(
                "COPC contains {} points vs {} from input headers (diff {}). \
                 Input LAZ headers sometimes report inaccurate point counts.",
                total_pts,
                self.total_points,
                self.total_points as i64 - total_pts as i64
            );
        }

        // Ensure every ancestor of every data node is present (the writer
        // requires zero-point ancestors so validators can traverse top-down).
        // Same logic as the per-leaf path.
        let mut present: HashSet<VoxelKey> = result.iter().map(|(k, _)| *k).collect();
        let mut extra: Vec<VoxelKey> = Vec::new();
        for (key, _) in &result {
            let mut k = *key;
            while let Some(parent) = k.parent() {
                if present.insert(parent) {
                    extra.push(parent);
                }
                k = parent;
            }
        }
        for k in extra {
            result.push((k, 0));
        }
        result.sort_by_key(|(k, _)| k.level);

        Ok(result)
    }
}

impl Drop for OctreeBuilder {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.tmp_dir);
    }
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
    fn rawpoint_roundtrip_single() {
        let p = sample_point();
        let mut buf = Vec::new();
        p.write(&mut buf).unwrap();
        assert_eq!(buf.len(), RawPoint::BYTE_SIZE);
        let p2 = RawPoint::read(&mut &buf[..]).unwrap();
        assert_eq!(p.x, p2.x);
        assert_eq!(p.y, p2.y);
        assert_eq!(p.z, p2.z);
        assert_eq!(p.intensity, p2.intensity);
        assert_eq!(p.return_number, p2.return_number);
        assert_eq!(p.number_of_returns, p2.number_of_returns);
        assert_eq!(p.classification, p2.classification);
        assert_eq!(p.scan_angle, p2.scan_angle);
        assert_eq!(p.user_data, p2.user_data);
        assert_eq!(p.point_source_id, p2.point_source_id);
        assert_eq!(p.gps_time, p2.gps_time);
        assert_eq!(p.red, p2.red);
        assert_eq!(p.green, p2.green);
        assert_eq!(p.blue, p2.blue);
        assert_eq!(p.nir, p2.nir);
    }

    #[test]
    fn rawpoint_roundtrip_bulk() {
        let points = vec![
            sample_point(),
            RawPoint {
                x: 0,
                y: 0,
                z: 0,
                intensity: 0,
                return_number: 0,
                number_of_returns: 0,
                classification: 0,
                scan_angle: 0,
                user_data: 0,
                point_source_id: 0,
                gps_time: 0.0,
                red: 0,
                green: 0,
                blue: 0,
                nir: 0,
            },
            sample_point(),
        ];

        let mut buf = Vec::new();
        RawPoint::write_bulk(&points, &mut buf).unwrap();
        assert_eq!(buf.len(), RawPoint::BYTE_SIZE * 3);

        // Read them back one at a time
        let mut cursor = std::io::Cursor::new(&buf[..]);
        for orig in &points {
            let p = RawPoint::read(&mut cursor).unwrap();
            assert_eq!(orig.x, p.x);
            assert_eq!(orig.gps_time, p.gps_time);
            assert_eq!(orig.nir, p.nir);
        }
    }

    /// Regression test: multi-batch LZ4 files must round-trip every point.
    ///
    /// `FrameDecoder` returns `Ok(0)` at each frame boundary, which naïve
    /// readers mistake for EOF and truncate the stream after the first
    /// frame. `MultiFrameReader` + `read_temp_batches` must transparently
    /// walk every concatenated frame.
    #[test]
    fn lz4_multi_batch_round_trip() {
        let codec = crate::TempCompression::Lz4;
        // Four batches of varying sizes, written as four separate frames
        // into one buffer — mirrors what `ChunkWriterCache::append` does
        // under LRU pressure.
        let batches: Vec<Vec<RawPoint>> = vec![
            (0..10).map(|_| sample_point()).collect(),
            (0..1).map(|_| sample_point()).collect(),
            (0..1000).map(|_| sample_point()).collect(),
            (0..42).map(|_| sample_point()).collect(),
        ];
        let expected: usize = batches.iter().map(|b| b.len()).sum();

        let mut buf: Vec<u8> = Vec::new();
        for batch in &batches {
            write_temp_batch(&mut buf, batch, codec).unwrap();
        }

        let decoded = read_temp_batches(std::io::Cursor::new(&buf[..]), codec).unwrap();
        assert_eq!(
            decoded.len(),
            expected,
            "multi-frame LZ4 decode must recover every point"
        );
    }

    /// Same round-trip at the count helper: `count_temp_file_points` under
    /// LZ4 must also walk every frame rather than stopping at the first
    /// boundary.
    #[test]
    fn lz4_multi_batch_count_temp_file_points() {
        let codec = crate::TempCompression::Lz4;
        let batches: Vec<Vec<RawPoint>> = vec![
            (0..7).map(|_| sample_point()).collect(),
            (0..13).map(|_| sample_point()).collect(),
            (0..21).map(|_| sample_point()).collect(),
        ];
        let expected: u64 = batches.iter().map(|b| b.len() as u64).sum();

        let tmp = std::env::temp_dir().join(format!("copc_lz4_mf_count_{}", std::process::id()));
        {
            let mut f = BufWriter::new(File::create(&tmp).unwrap());
            for batch in &batches {
                write_temp_batch(&mut f, batch, codec).unwrap();
            }
            f.flush().unwrap();
        }
        let got = count_temp_file_points(&tmp, codec).unwrap();
        let _ = std::fs::remove_file(&tmp);
        assert_eq!(got, expected, "multi-frame count must match total");
    }

    #[test]
    fn rawpoint_bulk_matches_single() {
        let p = sample_point();
        let mut single_buf = Vec::new();
        p.write(&mut single_buf).unwrap();

        let mut bulk_buf = Vec::new();
        RawPoint::write_bulk(std::slice::from_ref(&p), &mut bulk_buf).unwrap();

        assert_eq!(
            single_buf, bulk_buf,
            "bulk write must produce identical bytes to single write"
        );
    }

    #[test]
    fn input_to_copc_format_mapping() {
        // No color, no NIR → format 6
        assert_eq!(input_to_copc_format(0), 6);
        assert_eq!(input_to_copc_format(1), 6);
        assert_eq!(input_to_copc_format(4), 6);
        assert_eq!(input_to_copc_format(6), 6);
        assert_eq!(input_to_copc_format(9), 6);

        // Has color, no NIR → format 7
        assert_eq!(input_to_copc_format(2), 7);
        assert_eq!(input_to_copc_format(3), 7);
        assert_eq!(input_to_copc_format(5), 7);
        assert_eq!(input_to_copc_format(7), 7);

        // Has color + NIR → format 8
        assert_eq!(input_to_copc_format(8), 8);
        assert_eq!(input_to_copc_format(10), 8);
    }

    // ----- Chunked-build helpers -----

    use crate::chunking::{ChunkPlan, PlannedChunk};

    fn make_plan(grid_size: u32, chunks: Vec<PlannedChunk>) -> ChunkPlan {
        let total_points = chunks.iter().map(|c| c.point_count).sum();
        ChunkPlan {
            grid_size,
            grid_depth: grid_size.trailing_zeros(),
            chunk_target: 1_000_000,
            total_points,
            chunks,
        }
    }

    #[test]
    fn build_chunk_lut_single_root_chunk() {
        // 4³ grid, one chunk at level 0 (the root) covering everything.
        let plan = make_plan(
            4,
            vec![PlannedChunk {
                level: 0,
                gx: 0,
                gy: 0,
                gz: 0,
                point_count: 100,
            }],
        );
        let lut = build_chunk_lut(&plan);
        assert_eq!(lut.len(), 64);
        assert!(lut.iter().all(|&c| c == 0));
    }

    #[test]
    fn build_chunk_lut_eight_finest_chunks() {
        // 2³ grid (depth 1). Eight chunks at level 1, one per fine cell.
        let chunks: Vec<PlannedChunk> = (0i32..8)
            .map(|i| PlannedChunk {
                level: 1,
                gx: i % 2,
                gy: (i / 2) % 2,
                gz: i / 4,
                point_count: 10,
            })
            .collect();
        let plan = make_plan(2, chunks);
        let lut = build_chunk_lut(&plan);
        assert_eq!(lut.len(), 8);
        // Each cell should map to a distinct chunk index 0..8.
        let mut seen: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for &c in &lut {
            assert!(c < 8, "got out-of-range chunk idx {c}");
            seen.insert(c);
        }
        assert_eq!(seen.len(), 8);
    }

    #[test]
    fn build_chunk_lut_mixed_levels() {
        // 4³ grid (depth 2). Two chunks:
        //   - chunk 0 at level 1, position (0,0,0) — covers the 2³ subcube at the origin (8 cells)
        //   - chunk 1 at level 2, position (3,3,3) — covers exactly 1 fine cell at (3,3,3)
        // The remaining 64 - 8 - 1 = 55 cells should be u32::MAX (uncovered).
        let plan = make_plan(
            4,
            vec![
                PlannedChunk {
                    level: 1,
                    gx: 0,
                    gy: 0,
                    gz: 0,
                    point_count: 50,
                },
                PlannedChunk {
                    level: 2,
                    gx: 3,
                    gy: 3,
                    gz: 3,
                    point_count: 5,
                },
            ],
        );
        let lut = build_chunk_lut(&plan);
        assert_eq!(lut.len(), 64);

        // Cells (gx, gy, gz) with all coords in 0..2 should map to chunk 0.
        for z in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    let idx = x + y * 4 + z * 16;
                    assert_eq!(lut[idx], 0, "cell ({x},{y},{z}) should map to chunk 0");
                }
            }
        }

        // Cell (3, 3, 3) → chunk 1.
        assert_eq!(lut[3 + 3 * 4 + 3 * 16], 1);

        // Count uncovered cells: should be 64 - 8 - 1 = 55.
        let uncovered = lut.iter().filter(|&&c| c == u32::MAX).count();
        assert_eq!(uncovered, 55);
    }

    #[test]
    fn chunk_local_extra_levels_basic() {
        // ≤ MAX_LEAF_POINTS → no subdivision needed
        assert_eq!(chunk_local_extra_levels(0), 0);
        assert_eq!(chunk_local_extra_levels(1), 0);
        assert_eq!(chunk_local_extra_levels(MAX_LEAF_POINTS), 0);

        // Just above MAX_LEAF_POINTS → 1 extra level (8 leaves)
        assert_eq!(chunk_local_extra_levels(MAX_LEAF_POINTS + 1), 1);
        assert_eq!(chunk_local_extra_levels(8 * MAX_LEAF_POINTS), 1);

        // 8x leaf budget → still 1 extra level (avg = MAX_LEAF_POINTS)
        // 8x + 1 → needs 2 extra levels
        assert_eq!(chunk_local_extra_levels(8 * MAX_LEAF_POINTS + 1), 2);

        // 36M points → ceil(log8(360)) = 3
        assert_eq!(chunk_local_extra_levels(36_000_000), 3);

        // 100M points → ceil(log8(1000)) = 4
        assert_eq!(chunk_local_extra_levels(100_000_000), 4);
    }

    #[test]
    fn build_chunk_lut_chunk_at_corner() {
        // 8³ grid (depth 3). One chunk at level 2 position (3, 3, 3) — covers
        // 2³ cells at the far corner of the grid: (6..8, 6..8, 6..8).
        let plan = make_plan(
            8,
            vec![PlannedChunk {
                level: 2,
                gx: 3,
                gy: 3,
                gz: 3,
                point_count: 10,
            }],
        );
        let lut = build_chunk_lut(&plan);
        assert_eq!(lut.len(), 512);
        let g = 8;
        for z in 6..8 {
            for y in 6..8 {
                for x in 6..8 {
                    let idx = x + y * g + z * g * g;
                    assert_eq!(lut[idx], 0, "cell ({x},{y},{z}) should be in the chunk");
                }
            }
        }
        // No other cell should be in the chunk.
        let in_chunk = lut.iter().filter(|&&c| c == 0).count();
        assert_eq!(in_chunk, 8);
    }
}
