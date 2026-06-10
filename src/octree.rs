use crate::PipelineConfig;
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
use crate::copc_types::VoxelKey;
use crate::node_store::{FileNodeStore, NodeStore, PackedNodeStore};
use anyhow::{Context, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use rayon::prelude::*;
// Internal scratch maps are keyed by voxel coordinates / chunk indices and
// hashed once or more per *point* in the hot paths (leaf classification,
// distribute grouping, spill routing). FxHash trades SipHash's HashDoS
// resistance — irrelevant for internal temp state derived from point
// geometry — for a hash that is a few cycles per key instead of dozens.
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::{debug, info};

// ---------------------------------------------------------------------------
// Distribute constants and helper types
// ---------------------------------------------------------------------------

/// Maximum number of leaf chunk files kept open simultaneously across all
/// distribute workers. Divided evenly between workers.
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
struct ChunkWriterCache {
    writers: FxHashMap<u32, BufWriter<File>>,
    /// Insertion / access order. Front = least recently used.
    order: VecDeque<u32>,
    capacity: usize,
    num_extra_bytes: u16,
    codec: TempCompression,
}

impl ChunkWriterCache {
    fn new(capacity: usize, num_extra_bytes: u16, codec: TempCompression) -> Self {
        ChunkWriterCache {
            writers: FxHashMap::default(),
            order: VecDeque::new(),
            capacity,
            num_extra_bytes,
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
        write_temp_batch(w, points, self.num_extra_bytes, self.codec)?;
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

/// ~2/3 of cores, leaving headroom for the LAZ parallel decoder. Capped
/// by input file count.
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
/// `MAX_LEAF_POINTS` per leaf; `grid_sample`'s natural cascading handles
/// the imbalance during the bottom-up build.
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

/// Sentinel `cells` entry meaning "this fine cell was sub-split into deeper
/// octree sub-chunks; consult `sub` for the split descriptor and route points
/// with a deeper `point_to_key` descent." Distinct from `u32::MAX`, which
/// still means "no chunk covers this cell" (a bug for a well-formed plan).
const SUBSPLIT_MARKER: u32 = u32::MAX - 1;

/// Descriptor for a sub-split fine cell: how many extra octree levels deep
/// its sub-chunks live (`k`) and the chunk index of its first sub-octant.
///
/// Sub-chunk indices are contiguous from `base` in z-major, then y, then x
/// order — `base + sx + sy*2^k + sz*4^k` for local octant `(sx, sy, sz)` —
/// matching the emission order in `chunking::emit_chunk_root`.
#[derive(Debug, Clone, Copy)]
struct SubSplit {
    k: u32,
    base: u32,
}

/// Cell → chunk routing table built from a chunk plan.
///
/// `cells[gx + gy*G + gz*G²]` holds the chunk index for that fine cell, or
/// [`SUBSPLIT_MARKER`] when the cell was sub-split (look it up in `sub`), or
/// `u32::MAX` when no chunk covers it (a malformed plan; treated as a bug).
struct ChunkLut {
    cells: Vec<u32>,
    sub: FxHashMap<usize, SubSplit>,
}

/// Local octant index of a deeper sub-chunk voxel within its `grid_depth`
/// ancestor cell, given the split depth `k`. Inverse of the placement in
/// `chunking::emit_chunk_root`.
fn sub_octant_index(gx: i32, gy: i32, gz: i32, k: u32) -> u32 {
    let mask = (1i32 << k) - 1;
    let sx = (gx & mask) as u32;
    let sy = (gy & mask) as u32;
    let sz = (gz & mask) as u32;
    sx + (sy << k) + (sz << (2 * k))
}

/// Build the cell → chunk routing table from a chunk plan.
///
/// The grid has `grid_size³` fine cells. Chunks at or above `grid_depth`
/// cover a `2^(grid_depth - level)` cube of fine cells and fill each covered
/// cell with their index. Sub-split chunks (`level > grid_depth`) all share
/// one fine cell; that cell is marked [`SUBSPLIT_MARKER`] and a [`SubSplit`]
/// descriptor records the split depth and base index so distribute can route
/// points into the right sub-chunk with a deeper descent.
fn build_chunk_lut(plan: &crate::chunking::ChunkPlan) -> ChunkLut {
    let g = plan.grid_size as usize;
    let n_cells = g * g * g;
    let mut cells = vec![u32::MAX; n_cells];
    let mut sub: FxHashMap<usize, SubSplit> = FxHashMap::default();

    for (chunk_idx, chunk) in plan.chunks.iter().enumerate() {
        let chunk_idx = chunk_idx as u32;
        let level_diff = plan.grid_depth as i32 - chunk.level as i32;

        if level_diff < 0 {
            // Sub-split chunk: deeper than the grid. Map it back to its
            // ancestor fine cell and record (or check) the split descriptor.
            let k = chunk.level - plan.grid_depth;
            // Ancestor cell coords at grid_depth: drop the k low octant bits.
            let cell_gx = (chunk.gx >> k) as usize;
            let cell_gy = (chunk.gy >> k) as usize;
            let cell_gz = (chunk.gz >> k) as usize;
            debug_assert!(
                cell_gx < g && cell_gy < g && cell_gz < g,
                "sub-chunk cell out of grid"
            );
            let cell_idx = cell_gx + cell_gy * g + cell_gz * g * g;
            let octant = sub_octant_index(chunk.gx, chunk.gy, chunk.gz, k);
            let base = chunk_idx - octant;
            cells[cell_idx] = SUBSPLIT_MARKER;
            match sub.get(&cell_idx) {
                Some(existing) => debug_assert!(
                    existing.k == k && existing.base == base,
                    "inconsistent sub-split for cell {cell_idx}: {existing:?} vs {{k:{k}, base:{base}}}"
                ),
                None => {
                    sub.insert(cell_idx, SubSplit { k, base });
                }
            }
            continue;
        }

        // Chunk at or above grid depth: fill its covered cube of fine cells.
        let stride: usize = 1usize << level_diff as u32;
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
                for cell in &mut cells[row_start..row_end] {
                    *cell = chunk_idx;
                }
            }
        }
    }

    ChunkLut { cells, sub }
}

/// Move per-worker shard files into the canonical `tmp_dir/chunks/chunk_N`
/// location. Runs in parallel over chunk indices.
///
/// A chunk covers a contiguous spatial region while each worker streams a
/// contiguous slice of the input files, so most chunks receive points from
/// few workers — frequently exactly one. The single-shard case is a
/// `rename` (no data movement at all; shards and chunks live under the
/// same temp directory). With multiple shards, the first is renamed and
/// the rest are appended via plain `File`-to-`File` `io::copy`, which on
/// Linux specializes to in-kernel `copy_file_range` — the destination is
/// deliberately *not* opened in append mode and not wrapped in a
/// `BufWriter`, since either would force the userspace fallback path.
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
            use std::io::{Seek, SeekFrom};

            let canonical = chunk_canonical_path(chunks_root, chunk_idx);
            let shard_paths: Vec<PathBuf> = (0..num_workers)
                .map(|w| {
                    shards_root
                        .join(w.to_string())
                        .join(format!("chunk_{chunk_idx}"))
                })
                .filter(|p| p.exists())
                .collect();

            match shard_paths.as_slice() {
                [] => {
                    // No worker produced points for this chunk (the planner
                    // emits all 8^k sub-octants of a sub-split cell, some of
                    // which are empty). The build phase streams every
                    // planned chunk's canonical file, so it must exist.
                    File::create(&canonical)
                        .with_context(|| format!("creating empty chunk file {:?}", canonical))?;
                }
                [single] => {
                    std::fs::rename(single, &canonical).with_context(|| {
                        format!("renaming shard {:?} to {:?}", single, canonical)
                    })?;
                }
                [first, rest @ ..] => {
                    std::fs::rename(first, &canonical).with_context(|| {
                        format!("renaming shard {:?} to {:?}", first, canonical)
                    })?;
                    let mut out = OpenOptions::new()
                        .write(true)
                        .open(&canonical)
                        .with_context(|| format!("opening canonical chunk file {:?}", canonical))?;
                    out.seek(SeekFrom::End(0))
                        .context("seeking to end of canonical chunk file")?;
                    for shard in rest {
                        let mut input = File::open(shard)
                            .with_context(|| format!("opening shard {:?}", shard))?;
                        std::io::copy(&mut input, &mut out).with_context(|| {
                            format!("copying shard {:?} into {:?}", shard, canonical)
                        })?;
                    }
                }
            }
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

/// Estimated peak working-set bytes per point during a single chunk's
/// in-memory build. The peak sums the leaf map (~48 B/pt), the `all_pts`
/// sort buffer that lives alongside the leaves (~56 B/pt), grid_sample's
/// outputs (~56 B/pt), plus HashMap growth and allocator fragmentation;
/// 600 B/pt leaves genuine headroom over the raw ~160 B/pt sum. Used both
/// to size build batches and to decide when a chunk must take the spilled
/// (sub-divided) build path. Mirrors `chunking::PER_CHUNK_PEAK_BYTES_PER_POINT`.
const PER_CHUNK_BYTES_PER_POINT_BUILD: u64 = 600;

/// Fraction of the memory budget that phase 1's live build working set may
/// occupy. The remaining headroom covers the process baseline still resident
/// when the build starts (decoder buffers and allocations retained from scan /
/// count / distribute), the writer phase's later peak, and allocator slack.
/// Kept below half so that baseline + build stays well under the budget.
const BUILD_WORKING_SET_FRACTION: f64 = 0.4;

/// Smallest per-chunk in-memory budget we will split the working-set pool
/// into. Concurrency grows by handing each new build slot at least this much,
/// so slots stay large enough to build typical chunks fully in memory rather
/// than forcing the (more memory-hungry) spill path. Larger budgets therefore
/// buy *more concurrency* at a healthy per-slot size, not ever-tinier slots.
const MIN_BUILD_SLOT_BYTES: u64 = 512 * 1024 * 1024;

/// Hard cap on concurrent build slots, regardless of cores or budget. Past
/// this, extra parallelism yields diminishing returns while multiplying peak
/// memory and the number of spill temp files open at once. Big machines still
/// get large per-slot budgets; they just don't fan out beyond this width.
const MAX_BUILD_CONCURRENCY: usize = 16;

/// A merge parent must be grid-sampled fully in memory (there is no spill for
/// the merge step), so it is only rejected as "too large" when its children's
/// combined points exceed `max(memory_budget, this floor)`. The floor lets the
/// merge proceed for parents that are trivially holdable even when the
/// configured budget is artificially tiny — merge parents hold LOD-thinned
/// node data, which is inherently small, so anything under a couple of GB is
/// always safe to grid-sample regardless of `--memory-limit`.
const MERGE_PARENT_OOM_FLOOR_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// Aggregate counters for the spill path, accumulated across all chunks during
/// phase 1 and logged once afterwards. Avoids per-chunk / per-sub-octant log
/// lines (which flood at scale) while still surfacing whether — and how hard —
/// the spill engaged: the number of chunks that spilled, the deepest extra
/// split level any of them needed, and the total sub-octant files produced.
#[derive(Default)]
struct SpillStats {
    chunks_spilled: AtomicU64,
    max_split_extra: AtomicU64,
    sub_octant_files: AtomicU64,
}

/// Compute the phase-1 build concurrency and the per-chunk in-memory budget.
///
/// The working-set pool is `BUILD_WORKING_SET_FRACTION × memory_budget`.
/// Concurrency is the largest count (≥ 1) that still gives each slot at least
/// `MIN_BUILD_SLOT_BYTES`, capped by `cores` and `MAX_BUILD_CONCURRENCY`; the
/// per-chunk budget is then the pool divided evenly among the slots. Peak live
/// build memory is therefore ≈ the pool (`concurrency × per_chunk`), bounded
/// by `BUILD_WORKING_SET_FRACTION × memory_budget`.
///
/// This scales across the whole budget range: a 1 GB budget yields one slot
/// taking the whole pool (so a chunk builds in memory rather than spilling),
/// while a 256 GB budget fans out to `min(cores, MAX_BUILD_CONCURRENCY)` slots
/// each with a large budget. The per-chunk budget is deliberately kept *large*
/// — a chunk only spills when it genuinely cannot fit one slot, because the
/// spill path costs more peak memory than an in-memory build, so it is a
/// correctness safety-net for oversized chunks, never a memory-saving tactic.
///
/// Returns `(concurrency, per_chunk_budget_bytes)`.
fn build_concurrency_and_budget(memory_budget: u64, cores: usize) -> (usize, u64) {
    let pool = (memory_budget as f64 * BUILD_WORKING_SET_FRACTION) as u64;
    // Grow concurrency only while each slot still clears the minimum slot size,
    // so slots stay large (spill-avoiding) rather than splitting into tiny
    // pieces. At least one slot always.
    let by_memory = (pool / MIN_BUILD_SLOT_BYTES).max(1) as usize;
    let concurrency = by_memory.min(cores.max(1)).min(MAX_BUILD_CONCURRENCY);
    // Each slot's share of the pool is its in-memory budget; a chunk whose
    // actual working set exceeds this spills. Floored at one MAX_LEAF_POINTS
    // leaf set so even a tiny budget can make progress one chunk at a time.
    let min_per_chunk = MAX_LEAF_POINTS.saturating_mul(PER_CHUNK_BYTES_PER_POINT_BUILD);
    let per_chunk = (pool / concurrency as u64).max(min_per_chunk);
    (concurrency, per_chunk)
}

/// Grid cells per axis for LOD thinning. Matches untwine's CellCount = 128.
/// Higher values keep more points at coarse LOD levels (better progressive rendering).
pub(crate) const GRID_CELLS_PER_AXIS: i64 = 128;

// ---------------------------------------------------------------------------
// Raw point storage
// ---------------------------------------------------------------------------

/// A raw point stored as scaled integer coordinates plus attributes.
/// Scaled ints allow exact LAS round-trip without floating-point loss.
///
/// `extras` carries the trailing per-point bytes from any LAS Extra Bytes
/// VLR present in the input. When the input declares no extras the slice
/// is empty and adds no heap pressure (empty `Box<[u8]>` is one word).
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
    pub extras: Box<[u8]>,
}

impl RawPoint {
    /// Size of the fixed (non-extras) portion of the on-disk record.
    pub const BASE_BYTE_SIZE: usize = 4 + 4 + 4 + 2 + 1 + 1 + 1 + 2 + 1 + 2 + 8 + 2 + 2 + 2 + 2; // 38

    /// Total on-disk record size for the given extras width.
    pub const fn record_size(num_extra_bytes: u16) -> usize {
        Self::BASE_BYTE_SIZE + num_extra_bytes as usize
    }

    fn write_base_into(&self, c: &mut std::io::Cursor<&mut [u8]>) -> Result<()> {
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
        c.write_all(&self.extras)?;
        Ok(())
    }

    /// Decode one on-disk record. `rec` must be exactly
    /// `record_size(num_extra_bytes)` bytes (offsets mirror
    /// [`Self::write_base_into`]). The only allocation is the extras
    /// `Box<[u8]>`, and only when `num_extra_bytes > 0`.
    fn from_record(rec: &[u8], num_extra_bytes: u16) -> Self {
        debug_assert_eq!(rec.len(), Self::record_size(num_extra_bytes));
        let i32le = |o: usize| i32::from_le_bytes(rec[o..o + 4].try_into().unwrap());
        let u16le = |o: usize| u16::from_le_bytes(rec[o..o + 2].try_into().unwrap());
        RawPoint {
            x: i32le(0),
            y: i32le(4),
            z: i32le(8),
            intensity: u16le(12),
            return_number: rec[14],
            number_of_returns: rec[15],
            classification: rec[16],
            scan_angle: i16::from_le_bytes(rec[17..19].try_into().unwrap()),
            user_data: rec[19],
            point_source_id: u16le(20),
            gps_time: f64::from_le_bytes(rec[22..30].try_into().unwrap()),
            red: u16le(30),
            green: u16le(32),
            blue: u16le(34),
            nir: u16le(36),
            extras: rec[Self::BASE_BYTE_SIZE..].to_vec().into_boxed_slice(),
        }
    }

    /// Read a single record from a reader. Production code decodes points
    /// in bulk via [`for_each_point_in_batches`]; this single-point form
    /// is kept for round-trip tests.
    #[cfg(test)]
    pub fn read<R: std::io::Read>(r: &mut R, num_extra_bytes: u16) -> Result<Self> {
        let record_size = Self::record_size(num_extra_bytes);
        let mut buf = vec![0u8; record_size];
        r.read_exact(&mut buf)?;
        Ok(Self::from_record(&buf, num_extra_bytes))
    }

    /// Write multiple points to a writer in a single bulk operation.
    /// All points must carry exactly `num_extra_bytes` of extras.
    pub fn write_bulk<W: std::io::Write>(
        points: &[RawPoint],
        num_extra_bytes: u16,
        w: &mut W,
    ) -> Result<()> {
        let record_size = Self::record_size(num_extra_bytes);
        let mut buf = vec![0u8; record_size * points.len()];
        {
            let mut c = std::io::Cursor::new(&mut buf[..]);
            for p in points {
                debug_assert_eq!(
                    p.extras.len(),
                    num_extra_bytes as usize,
                    "RawPoint::write_bulk: extras length mismatch"
                );
                p.write_base_into(&mut c)?;
            }
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
//     point_count × RawPoint  (record_size = 38 + num_extra_bytes per point)
//
// The record size is parameterised by `num_extra_bytes` (carried by the
// `OctreeBuilder`) so per-point Extra Bytes payloads can flow through the
// pipeline without per-point allocation overhead beyond a single Box<[u8]>.
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
/// `points.len() × record_size` payload bytes. If `codec` is `Lz4`, the
/// entire batch is encapsulated in a self-contained LZ4 frame.
pub(crate) fn write_temp_batch<W: Write>(
    w: &mut W,
    points: &[RawPoint],
    num_extra_bytes: u16,
    codec: TempCompression,
) -> Result<()> {
    match codec {
        TempCompression::None => {
            w.write_u32::<LittleEndian>(points.len() as u32)?;
            RawPoint::write_bulk(points, num_extra_bytes, w)?;
        }
        TempCompression::Lz4 => {
            let mut enc = lz4_flex::frame::FrameEncoder::new(w);
            enc.write_u32::<LittleEndian>(points.len() as u32)?;
            RawPoint::write_bulk(points, num_extra_bytes, &mut enc)?;
            enc.finish().context("finishing lz4 frame")?;
        }
    }
    Ok(())
}

/// Read every batch in a reader to EOF, concatenating their points.
/// For `Lz4` the reader is wrapped in a `FrameDecoder` + `MultiFrameReader`,
/// which together walk multi-frame streams transparently. An empty reader
/// produces an empty Vec.
pub(crate) fn read_temp_batches<R: std::io::Read>(
    r: R,
    num_extra_bytes: u16,
    codec: TempCompression,
) -> Result<Vec<RawPoint>> {
    match codec {
        TempCompression::None => read_batches_loop(&mut BufReader::new(r), num_extra_bytes),
        TempCompression::Lz4 => {
            let dec = lz4_flex::frame::FrameDecoder::new(BufReader::new(r));
            let mut mf = MultiFrameReader { inner: dec };
            read_batches_loop(&mut mf, num_extra_bytes)
        }
    }
}

fn read_batches_loop<R: std::io::Read>(r: &mut R, num_extra_bytes: u16) -> Result<Vec<RawPoint>> {
    let mut out: Vec<RawPoint> = Vec::new();
    for_each_point_in_batches(r, num_extra_bytes, |p| {
        out.push(p);
        Ok(())
    })?;
    Ok(out)
}

/// Upper bound on points decoded per contiguous read in
/// [`for_each_point_in_batches`]. Caps the reused payload buffer at
/// `65_536 × record_size` (~2.5 MB at the 38-byte base record) so streaming
/// over a file whose batches are huge (e.g. a whole node written as one
/// batch) never materialises the full payload at once.
const MAX_POINTS_PER_READ: usize = 65_536;

/// Stream every point in a batch reader, invoking `f` on each one.
///
/// The chunk build path uses this to classify points into their leaf
/// voxels as they are decoded, avoiding an intermediate `Vec<RawPoint>`
/// that would hold every point in the chunk resident at once.
///
/// Points are decoded from contiguous sub-batch reads (capped at
/// [`MAX_POINTS_PER_READ`]) into a single reused buffer, rather than one
/// tiny read + heap allocation per point — temp files are re-read several
/// times across build, merge, and write, so the per-point path used to
/// dominate allocator traffic.
fn for_each_point_in_batches<R: std::io::Read, F: FnMut(RawPoint) -> Result<()>>(
    r: &mut R,
    num_extra_bytes: u16,
    mut f: F,
) -> Result<()> {
    let record_size = RawPoint::record_size(num_extra_bytes);
    // Grown lazily so reading a small node file doesn't pay the full cap.
    let mut buf: Vec<u8> = Vec::new();
    loop {
        let count = match r.read_u32::<LittleEndian>() {
            Ok(n) => n as usize,
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        };
        let mut remaining = count;
        while remaining > 0 {
            let take = remaining.min(MAX_POINTS_PER_READ);
            let need = take * record_size;
            if buf.len() < need {
                buf.resize(need, 0);
            }
            r.read_exact(&mut buf[..need])?;
            for rec in buf[..need].chunks_exact(record_size) {
                f(RawPoint::from_record(rec, num_extra_bytes))?;
            }
            remaining -= take;
        }
    }
    Ok(())
}

/// Streaming counterpart to `read_temp_batches`. Invokes `f` on every
/// decoded `RawPoint` without ever materialising the full Vec.
fn stream_temp_batches<R: std::io::Read, F: FnMut(RawPoint) -> Result<()>>(
    r: R,
    num_extra_bytes: u16,
    codec: TempCompression,
    f: F,
) -> Result<()> {
    match codec {
        TempCompression::None => {
            for_each_point_in_batches(&mut BufReader::new(r), num_extra_bytes, f)
        }
        TempCompression::Lz4 => {
            let dec = lz4_flex::frame::FrameDecoder::new(BufReader::new(r));
            let mut mf = MultiFrameReader { inner: dec };
            for_each_point_in_batches(&mut mf, num_extra_bytes, f)
        }
    }
}

/// Count the points in a temp file by walking batch headers. For `None` mode
/// this seeks past payloads without decoding. For `Lz4` mode there is no
/// cheap way to skip a compressed payload, so we run the decoder and count
/// headers (still cheaper than materialising all points — payload bytes are
/// streamed through a throwaway sink). Returns 0 if the file does not exist.
pub(crate) fn count_temp_file_points(
    path: &Path,
    num_extra_bytes: u16,
    codec: TempCompression,
) -> Result<u64> {
    let f = match File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(e) => return Err(e.into()),
    };
    let record_size = RawPoint::record_size(num_extra_bytes) as u64;
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
                let payload = count * record_size;
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
                let mut remaining = count * record_size;
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

    /// Cube that contains this AABB, padded by one scale unit per axis.
    ///
    /// The COPC spec reconstructs each node's per-axis cube bound via
    /// `cx − halfsize + (vx + 1) × (2 × halfsize / 2^depth)`. That float
    /// multiplication chain loses ~1 ULP at every depth, so a point sitting
    /// exactly on a cell boundary can land 1 ULP outside the
    /// spec-reconstructed bound while still being "the same point" at the
    /// file's stored precision. Padding halfsize by one scale unit gives
    /// every face slack far larger than any ULP drift.
    pub fn to_cube(&self, scale_x: f64, scale_y: f64, scale_z: f64) -> (f64, f64, f64, f64) {
        let cx = (self.min_x + self.max_x) / 2.0;
        let cy = (self.min_y + self.max_y) / 2.0;
        let cz = (self.min_z + self.max_z) / 2.0;
        let scale_pad = scale_x.max(scale_y).max(scale_z);
        let half = ((self.max_x - self.min_x)
            .max(self.max_y - self.min_y)
            .max(self.max_z - self.min_z))
            / 2.0
            + scale_pad;
        let half = half * 1.0001; // tiny relative epsilon on top
        (cx, cy, cz, half)
    }
}

/// Headroom kept below `i32::MAX` when checking whether
/// `(world − offset) / scale` is representable. Leaves room for rounding,
/// the half-scale-unit pad added by [`Bounds::to_cube`], and any tiny
/// expansion (e.g. an extra point one ULP past the header bounds).
const I32_HEADROOM: f64 = 2_147_400_000.0;

/// Ensure `min..=max` is representable as `i32 = (world − offset) / scale`.
///
/// First re-centers `offset` on the midpoint of `min..max` when the existing
/// offset would put either endpoint outside the headroom. Then doubles
/// `scale` until the half-extent fits. Returns `true` iff either value was
/// changed, so the caller can log the adjustment.
fn fit_offset_scale(offset: &mut f64, scale: &mut f64, min: f64, max: f64) -> bool {
    let fits = |o: f64, s: f64| -> bool {
        ((min - o) / s).abs() <= I32_HEADROOM && ((max - o) / s).abs() <= I32_HEADROOM
    };
    let original = (*offset, *scale);
    if !fits(*offset, *scale) {
        *offset = 0.5 * (min + max);
    }
    while !fits(*offset, *scale) {
        *scale *= 2.0;
    }
    (*offset, *scale) != original
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
#[derive(Debug, Clone)]
pub struct ScanResult {
    pub bounds: Bounds,
    pub point_count: u64,
    pub scale_x: f64,
    pub scale_y: f64,
    pub scale_z: f64,
    pub offset_x: f64,
    pub offset_y: f64,
    pub offset_z: f64,
    /// CRS kind found in the file (if any). For WKT CRS we store only a
    /// 64-bit hash so `ScanResult` size stays constant in the number of
    /// files — the canonical WKT bytes are kept once in `ScanOutput`. For
    /// GeoTIFF CRS we store the EPSG code(s) directly since they are tiny.
    pub crs: Option<CrsKind>,
    /// Parsed Extra Bytes VLR — structural fields + per-file stats.
    /// `None` when the file declares no extra bytes. Keeping the parsed
    /// form per file (a few hundred bytes per file) costs more than a
    /// single hash but lets validate produce rich per-field diff
    /// messages on mismatch and lets the writer merge stats honestly
    /// across all inputs without re-reading the VLR.
    pub(crate) extra_bytes_parsed: Option<crate::extra_bytes::ParsedExtraBytes>,
    /// Hash of the file's Extra Bytes VLR *schema* (structural fields
    /// only, with per-file stats excluded). Used as the fast equality
    /// check across files; identical hashes mean the per-point bytes
    /// can be interpreted with a single canonical schema.
    pub extra_bytes_schema_hash: Option<u64>,
    /// Trailing extra-byte width declared by this file's point format.
    /// Must match across all files (enforced in validate).
    pub num_extra_bytes: u16,
    pub point_format_id: u8,
}

/// Per-file CRS identity: small enough to hold once per `ScanResult` even
/// at 100k+ inputs. Full WKT bytes live elsewhere (see `ScanOutput`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrsKind {
    /// Hash of the WKT CRS VLR. Used for fast file-vs-file equality
    /// without retaining per-file WKT bytes.
    WktHash(u64),
    /// GeoTIFF EPSG codes: (horizontal, optional vertical).
    GeoTiffEpsg(u16, Option<u16>),
}

/// Output of the scan pass: per-file results plus the one canonical WKT
/// payload from the first file that had one. Downstream stages use this
/// single copy instead of rehydrating one per file.
pub struct ScanOutput {
    pub results: Vec<ScanResult>,
    pub canonical_wkt: Option<Vec<u8>>,
    /// Canonical LAS Extra Bytes VLR payload from the first file that had
    /// one. `None` when no input declares extra bytes. Validated to be
    /// byte-identical across all input files.
    pub canonical_extra_bytes_vlr: Option<Vec<u8>>,
}

/// Stable 64-bit hash of a byte payload. SipHash-1-3 with a fixed seed
/// so digests are identical across files in the same process. With 64
/// bits and typically a single VLR per kind per run, collision risk is
/// negligible. Used for both WKT CRS and Extra Bytes VLR identity.
pub(crate) fn bytes_hash(bytes: &[u8]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{BuildHasher, BuildHasherDefault, Hasher};
    let mut h = BuildHasherDefault::<DefaultHasher>::default().build_hasher();
    h.write(bytes);
    h.finish()
}

/// Best-effort EPSG code parse from WKT-CRS bytes.
///
/// Walks the trailing identifier digits at the end of the WKT (and at the
/// end of any `VERTCRS` / `VERTICALCRS` / `VERT_CS` sub-string for the
/// vertical component). This is not a real WKT parser; it returns `None`
/// when no plausible EPSG code is found, leaving callers to treat the
/// inputs as inequivalent rather than erroring out.
pub fn get_epsg_from_wkt_crs_bytes(bytes: &[u8]) -> Option<(u16, Option<u16>)> {
    const EPSG_RANGE: std::ops::Range<u16> = 1024..(i16::MAX as u16);
    let wkt = String::from_utf8_lossy(bytes);

    fn trailing_code(bytes: &[u8]) -> Option<u16> {
        // EPSG codes sit at the end of the substring (e.g. `…AUTHORITY["EPSG","4326"]]`).
        // Walk backwards collecting digits; bail if we never find any in the
        // last ~10 bytes (codes are 4–5 digits starting 2–3 bytes from the back).
        let mut code: u16 = 0;
        let mut started = false;
        let mut power: u16 = 1;
        for byte in bytes.trim_ascii_end().iter().rev().take(10) {
            if byte.is_ascii_digit() {
                started = true;
                code = code.checked_add(power.checked_mul((byte - b'0') as u16)?)?;
                power = power.checked_mul(10)?;
            } else if started {
                break;
            }
        }
        if started { Some(code) } else { None }
    }

    let (horizontal_part, vertical_part) = if let Some(split) = wkt.split_once("VERTCRS") {
        (split.0, Some(split.1))
    } else if let Some(split) = wkt.split_once("VERTICALCRS") {
        (split.0, Some(split.1))
    } else if let Some(split) = wkt.split_once("VERT_CS") {
        (split.0, Some(split.1))
    } else {
        (wkt.as_ref(), None)
    };

    let horizontal = trailing_code(horizontal_part.as_bytes())?;
    if !EPSG_RANGE.contains(&horizontal) {
        return None;
    }
    let vertical = vertical_part
        .and_then(|s| trailing_code(s.as_bytes()))
        .filter(|v| EPSG_RANGE.contains(v));
    Some((horizontal, vertical))
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
    /// X scale factor. Inherited from the first input file when the combined
    /// bounds fit in i32 around `offset_x`; otherwise doubled until they do.
    pub scale_x: f64,
    /// Y scale factor; see [`Self::scale_x`].
    pub scale_y: f64,
    /// Z scale factor; see [`Self::scale_x`].
    pub scale_z: f64,
    /// X offset. Inherited from the first input file when its value puts the
    /// combined min/max within i32 range at `scale_x`; otherwise re-centered
    /// on the combined-bounds midpoint.
    pub offset_x: f64,
    /// Y offset; see [`Self::offset_x`].
    pub offset_y: f64,
    /// Z offset; see [`Self::offset_x`].
    pub offset_z: f64,
    /// Temp directory where node files are written.
    pub tmp_dir: PathBuf,
    /// WKT CRS payload from the first input file (if present).
    pub wkt_crs: Option<Vec<u8>>,
    /// LAS Extra Bytes VLR payload (user_id=LASF_Spec, record_id=4) from
    /// the first input file (if present). All input files share this VLR
    /// (validated upstream); a single canonical copy lives here and is
    /// re-emitted into the output COPC unchanged.
    pub extra_bytes_vlr: Option<Vec<u8>>,
    /// Trailing extra-byte width of every point record. Zero when the
    /// inputs declare no extras. Validated to be uniform across input files.
    pub num_extra_bytes: u16,
    /// COPC output point format (6, 7, or 8), derived from input files.
    pub point_format: u8,
    /// Chunk plan computed by `distribute` and consumed by `build_node_map`.
    pub(crate) chunked_plan: Option<crate::chunking::ChunkPlan>,
    /// Compression codec applied to scratch temp files.
    pub(crate) temp_compression: crate::TempCompression,
    /// Storage backend for per-node point data. `Arc` so rayon workers
    /// share a single instance.
    pub(crate) node_store: Arc<dyn NodeStore>,
}

impl OctreeBuilder {
    /// Pass 1: scan all files in parallel to get bounds and total point count.
    ///
    /// Each file's WKT CRS bytes and Extra Bytes VLR bytes are hashed on
    /// the fly; only the first file's canonical payloads are kept in full.
    /// Downstream validate compares hashes and uses the canonical payloads
    /// for the output COPC.
    pub fn scan(input_files: &[PathBuf], config: &PipelineConfig) -> Result<ScanOutput> {
        // (ScanResult, wkt_bytes, extra_bytes_vlr) — the trailing Options
        // carry raw VLR bytes so we can pick the first non-None ones after
        // the parallel scan. Post-aggregation we drop every per-file Vec
        // and keep only one canonical copy of each.
        type PerFileEntry = (ScanResult, Option<Vec<u8>>, Option<Vec<u8>>);

        let done = std::sync::atomic::AtomicU64::new(0);
        let per_file: Result<Vec<PerFileEntry>> = input_files
            .par_iter()
            .map(|path| -> Result<PerFileEntry> {
                let reader = las::Reader::from_path(path)
                    .with_context(|| format!("Cannot open {:?}", path))?;
                let header = reader.header();
                let b = header.bounds();
                let mut bounds = Bounds::empty();
                bounds.expand_with(b.min.x, b.min.y, b.min.z);
                bounds.expand_with(b.max.x, b.max.y, b.max.z);
                let t = header.transforms();
                let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                config.report(crate::ProgressEvent::StageProgress { done: n });
                // CRS: support both WKT (LAS 1.4) and GeoTIFF (LAS 1.4 and earlier).
                let (crs, wkt_bytes): (Option<CrsKind>, Option<Vec<u8>>) =
                    if let Some(wkt) = header.get_wkt_crs_bytes() {
                        let bytes = wkt.to_vec();
                        let kind = CrsKind::WktHash(bytes_hash(&bytes));
                        (Some(kind), Some(bytes))
                    } else if let Ok(Some(geotiff)) = header.get_geotiff_crs() {
                        let horizontal = match geotiff.get_gt_model_type_geo_key_value() {
                            Some(1) => geotiff.get_projected_crs_geo_key_value(),
                            Some(2) | Some(3) => geotiff.get_geodetic_crs_geo_key_value(),
                            _ => None,
                        };
                        let vertical = geotiff.get_vertical_crs_geo_key_value();
                        let kind = horizontal.map(|h| CrsKind::GeoTiffEpsg(h, vertical));
                        (kind, None)
                    } else {
                        (None, None)
                    };
                // Extra Bytes: parse once per file.
                let extra_bytes_vlr: Option<Vec<u8>> = header
                    .all_vlrs()
                    .find(|v| v.user_id.trim_end_matches('\0') == "LASF_Spec" && v.record_id == 4)
                    .map(|v| v.data.clone());
                let extra_bytes_parsed = match &extra_bytes_vlr {
                    Some(bytes) => Some(
                        crate::extra_bytes::ParsedExtraBytes::parse(bytes)
                            .with_context(|| format!("parsing Extra Bytes VLR in {path:?}"))?,
                    ),
                    None => None,
                };
                let extra_bytes_schema_hash = extra_bytes_parsed
                    .as_ref()
                    .map(crate::extra_bytes::schema_hash);
                let num_extra_bytes = header.point_format().extra_bytes;
                Ok((
                    ScanResult {
                        bounds,
                        point_count: header.number_of_points(),
                        scale_x: t.x.scale,
                        scale_y: t.y.scale,
                        scale_z: t.z.scale,
                        offset_x: t.x.offset,
                        offset_y: t.y.offset,
                        offset_z: t.z.offset,
                        crs,
                        extra_bytes_parsed,
                        extra_bytes_schema_hash,
                        num_extra_bytes,
                        point_format_id: header.point_format().to_u8().unwrap_or(0),
                    },
                    wkt_bytes,
                    extra_bytes_vlr,
                ))
            })
            .collect();

        let per_file = per_file?;
        let mut canonical_wkt: Option<Vec<u8>> = None;
        let mut canonical_extra_bytes_vlr: Option<Vec<u8>> = None;
        let mut results: Vec<ScanResult> = Vec::with_capacity(per_file.len());
        for (sr, wkt, eb) in per_file {
            if canonical_wkt.is_none()
                && let Some(bytes) = wkt
            {
                canonical_wkt = Some(bytes);
            }
            if canonical_extra_bytes_vlr.is_none()
                && let Some(bytes) = eb
            {
                canonical_extra_bytes_vlr = Some(bytes);
            }
            results.push(sr);
        }
        Ok(ScanOutput {
            results,
            canonical_wkt,
            canonical_extra_bytes_vlr,
        })
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
        let (mut scale_x, mut scale_y, mut scale_z) = (first.scale_x, first.scale_y, first.scale_z);
        let (mut offset_x, mut offset_y, mut offset_z) =
            (first.offset_x, first.offset_y, first.offset_z);

        // Points are encoded as `i32 = round((world - offset) / scale)`. The
        // first input's offset/scale are sized for that file's bounds; when
        // multiple files have disjoint extents, points from outlying files
        // can fall outside the i32 representable range. Rust's saturating
        // `f64 as i32` cast then clamps every overflowing point to
        // `i32::MIN`/`MAX`, collapsing them onto the same wrong coordinate.
        // Re-center the offset and, if necessary, coarsen the scale so the
        // combined bounds fit. Scale only ever grows (precision is never
        // tightened beyond what the first file claimed).
        let adjusted_x = fit_offset_scale(&mut offset_x, &mut scale_x, bounds.min_x, bounds.max_x);
        let adjusted_y = fit_offset_scale(&mut offset_y, &mut scale_y, bounds.min_y, bounds.max_y);
        let adjusted_z = fit_offset_scale(&mut offset_z, &mut scale_z, bounds.min_z, bounds.max_z);
        if adjusted_x || adjusted_y || adjusted_z {
            info!(
                "Adjusted output offset/scale to fit combined extents in i32: \
                 offset=({offset_x:.6}, {offset_y:.6}, {offset_z:.6}) \
                 scale=({scale_x:e}, {scale_y:e}, {scale_z:e})",
            );
        }

        let (cx, cy, cz, halfsize) = bounds.to_cube(scale_x, scale_y, scale_z);

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
            debug!("Removing stale temp dir {:?}", tmp_dir);
            std::fs::remove_dir_all(&tmp_dir)?;
        }
        std::fs::create_dir_all(&tmp_dir)?;

        let num_extra_bytes = validated.num_extra_bytes;
        let node_store: Arc<dyn NodeStore> = match config.node_storage {
            crate::NodeStorage::Files => Arc::new(FileNodeStore::new(
                tmp_dir.clone(),
                num_extra_bytes,
                config.temp_compression,
            )),
            crate::NodeStorage::Packed => Arc::new(PackedNodeStore::new(
                &tmp_dir,
                num_extra_bytes,
                config.temp_compression,
                rayon::current_num_threads().max(1),
            )?),
        };

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
            extra_bytes_vlr: validated.extra_bytes_vlr.clone(),
            num_extra_bytes,
            point_format: validated.point_format,
            chunked_plan: None,
            temp_compression: config.temp_compression,
            node_store,
        })
    }

    /// Point count for a given node key, consulting the active node store.
    /// Returns 0 when the key has never been written.
    pub(crate) fn count_node(&self, key: &VoxelKey) -> Result<u64> {
        self.node_store.count(key)
    }

    /// Stream every point in a `PointData` byte slab as a `RawPoint`,
    /// re-encoding coordinates against the builder's scale/offset.
    ///
    /// Walks the slab via per-field column iterators instead of
    /// materialising `las::Point` values — no per-point allocation (the
    /// `extra_bytes` Vec in particular) and no `Option`/enum decoding for
    /// fields the format doesn't carry. Field semantics match what
    /// `las::Point` would have produced: legacy classifications have their
    /// flag bits masked off, scan angles are normalised to degrees and
    /// re-encoded as extended `i16` units, and absent gps/rgb/nir default
    /// to zero.
    fn for_each_raw_point<F: FnMut(RawPoint)>(&self, pd: &las::PointData, mut f: F) {
        let mut xs = pd.x();
        let mut ys = pd.y();
        let mut zs = pd.z();
        let mut intensities = pd.intensity();
        let mut return_numbers = pd.return_number();
        let mut numbers_of_returns = pd.number_of_returns();
        let mut classifications = pd.classification();
        let mut scan_angles = pd.scan_angle_degrees();
        let mut user_datas = pd.user_data();
        let mut point_source_ids = pd.point_source_id();
        let mut gps_times = pd.gps_time();
        let mut rgbs = pd.rgb();
        let mut nirs = pd.nir();
        // Extra bytes are the trailing `num_extra_bytes` of each record.
        // Validate enforces a uniform extras width across inputs, and the
        // slab's record length comes from the same header, so the slice
        // width is exact by construction.
        let record_len = pd.record_len();
        let nxb = self.num_extra_bytes as usize;
        let mut records = pd.raw_bytes().chunks_exact(record_len);

        for _ in 0..pd.len() {
            let x = xs.next().expect("x column ends early");
            let y = ys.next().expect("y column ends early");
            let z = zs.next().expect("z column ends early");
            let extras = if nxb == 0 {
                Box::<[u8]>::default()
            } else {
                let rec = records.next().expect("record column ends early");
                rec[record_len - nxb..].to_vec().into_boxed_slice()
            };
            let (red, green, blue) = rgbs.as_mut().and_then(|it| it.next()).unwrap_or((0, 0, 0));
            f(RawPoint {
                x: ((x - self.offset_x) / self.scale_x).round() as i32,
                y: ((y - self.offset_y) / self.scale_y).round() as i32,
                z: ((z - self.offset_z) / self.scale_z).round() as i32,
                intensity: intensities.next().expect("intensity column ends early"),
                return_number: return_numbers.next().expect("rn column ends early"),
                number_of_returns: numbers_of_returns.next().expect("nor column ends early"),
                classification: classifications.next().expect("class column ends early"),
                scan_angle: (scan_angles.next().expect("angle column ends early") / 0.006).round()
                    as i16,
                user_data: user_datas.next().expect("user_data column ends early"),
                point_source_id: point_source_ids.next().expect("psid column ends early"),
                gps_time: gps_times.as_mut().and_then(|it| it.next()).unwrap_or(0.0),
                red,
                green,
                blue,
                nir: nirs.as_mut().and_then(|it| it.next()).unwrap_or(0),
                extras,
            });
        }
    }

    /// Read all raw points for a given node key from the active node store.
    /// Returns an empty Vec when the key has never been written.
    pub fn read_node(&self, key: &VoxelKey) -> Result<Vec<RawPoint>> {
        self.node_store.read(key)
    }

    /// Write points for the given node key (overwrites any prior content).
    pub fn write_node_to_temp(&self, key: &VoxelKey, points: &[RawPoint]) -> Result<()> {
        self.node_store.write(key, points)
    }

    /// Bottom-up grid-sample loop over an in-memory `nodes` HashMap.
    ///
    /// Walks levels `(min_level..actual_max_depth)` in reverse, grouping
    /// children of each parent and running `grid_sample` to produce that
    /// parent. After the loop completes, every node in the resulting map is
    /// written to its canonical temp file location.
    ///
    /// With `min_level = 0` the loop walks all the way to the global root.
    /// With a positive `min_level` the loop stops once it has produced
    /// parents at `min_level`, leaving any coarser ancestors to a later
    /// merge step — used by per-chunk builds that stop at the chunk's
    /// root level.
    ///
    /// `report_progress` controls whether per-level `StageProgress` events
    /// are emitted. Set `false` when the caller's outer stage tracks a
    /// different unit of progress (e.g. chunks-done).
    fn bottom_up_levels(
        &self,
        mut nodes: FxHashMap<VoxelKey, Vec<RawPoint>>,
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
            // Group children at level d+1 by parent (iterate keys, no disk I/O).
            let mut parent_children: FxHashMap<VoxelKey, Vec<VoxelKey>> = FxHashMap::default();
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
                for (ck, rem) in children.into_iter().zip(remaining) {
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

        // Collapse: if everything gathered here fits a single leaf, keep it all
        // in the parent and return empty children. Applied bottom-up, this
        // lifts an under-full subtree into one leaf at the shallowest level it
        // fits — so output node depth tracks point *density*, not the build's
        // (memory-driven) spill depth. Without this, a region the spill had to
        // split deep stays fragmented into many tiny deep nodes even though its
        // points would comfortably fit one leaf. `pts` here is the full subtree
        // total because collapse runs deepest-first: a collapsed child folds
        // all its points into the parent, so the next level up sees the true
        // total; a non-collapsed (oversized) child correctly stays subdivided.
        if pts.len() as u64 <= MAX_LEAF_POINTS {
            let parent_pts = pts.into_iter().map(|(_, p)| p).collect();
            return (parent_pts, vec![vec![]; n_children]);
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

        // Sort by Morton code within the parent voxel for spatially coherent
        // traversal. `sort_by_cached_key` computes one Morton code per point
        // (instead of two per comparison with `sort_unstable_by_key`) and
        // moves the 64-byte elements once in a final permutation pass rather
        // than swapping them throughout the sort. The transient key+index
        // table costs ~16 B/pt, well inside the build's 600 B/pt estimate.
        pts.sort_by_cached_key(|(_, p)| {
            let dx = (p.x as i64 - origin_x).max(0) as u32;
            let dy = (p.y as i64 - origin_y).max(0) as u32;
            let dz = (p.z as i64 - origin_z).max(0) as u32;
            morton3(dx, dy, dz)
        });

        // Track which children actually have points so we can protect them.
        let mut child_has_pts = vec![false; n_children];
        for (ci, _) in &pts {
            child_has_pts[*ci] = true;
        }

        // Occupancy bitmap over the fixed 128³ sampling grid (256 KiB),
        // replacing a hashed per-point set insert with a load+OR. This
        // path only runs for parents holding > MAX_LEAF_POINTS, so the
        // fixed allocation amortizes to under 3 bytes per thousand points.
        // Cell coordinates are clamped to the grid, so a boundary point
        // that rounds one cell out (the origins are rounded to integer
        // coordinates) shares the face cell instead of occupying a
        // phantom out-of-grid cell.
        let g = GRID_CELLS_PER_AXIS;
        let n_cells = (g * g * g) as usize;
        let mut occupied = vec![0u64; n_cells / 64];
        let grid_idx = |p: &RawPoint| -> usize {
            let gx = ((p.x as i64 - origin_x) / cell).clamp(0, g - 1);
            let gy = ((p.y as i64 - origin_y) / cell).clamp(0, g - 1);
            let gz = ((p.z as i64 - origin_z) / cell).clamp(0, g - 1);
            (gx + gy * g + gz * g * g) as usize
        };

        // Partition: accepted for parent vs remaining for children. No
        // cloning. Accepted points are structurally capped at one per grid
        // cell (n_cells total) by the bitmap, so no explicit cap check is
        // needed; capacity is sized to the input, not the theoretical max.
        let mut parent_pts: Vec<(usize, RawPoint)> = Vec::with_capacity(pts.len().min(n_cells));
        let mut remaining: Vec<Vec<RawPoint>> = vec![Vec::new(); n_children];

        for (ci, p) in pts {
            let idx = grid_idx(&p);
            let (word, mask) = (idx >> 6, 1u64 << (idx & 63));
            if occupied[word] & mask == 0 {
                occupied[word] |= mask;
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

    /// Distribute points to per-chunk temp files. Computes the chunk plan
    /// via the counting-sort analyzer, builds a fast cell→chunk lookup
    /// table, then streams each input file in parallel and appends each
    /// point to its chunk's temp file via a bounded LRU writer cache.
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
        debug!(
            "Distribute: {} chunks, target {} points each, grid {}³",
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
        // peak per point is the `PointData` byte slab (on-disk record
        // layout, ≤ ~67 B + extras) plus the grouped `RawPoint` awaiting
        // flush (~56 B + a boxed extras payload) plus the cache's BufWriter
        // overhead (negligible). Extras are resident twice — once in the
        // slab record, once in the RawPoint box. Size for 1/8 of the
        // per-worker budget so there's headroom for the LAZ decoder.
        const BASE_BYTES_PER_POINT_TRANSIENT: u64 = 120;
        let bytes_per_point_transient =
            BASE_BYTES_PER_POINT_TRANSIENT + 2 * self.num_extra_bytes as u64;
        let per_worker_budget = config.memory_budget / num_workers as u64;
        let read_chunk_size =
            ((per_worker_budget / 8) / bytes_per_point_transient).clamp(50_000, 2_000_000) as usize;

        debug!(
            "Distribute: budget={} MB, workers={}, read_chunk_size={}, open-file cap={} (per worker)",
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

                let mut cache = ChunkWriterCache::new(
                    per_worker_cap,
                    self.num_extra_bytes,
                    self.temp_compression,
                );
                // Group buffer reused across batches so capacity amortizes.
                // Drained at the end of each batch so retained Vec capacity
                // stays bounded by the working set, not by all-time peak.
                let mut groups: FxHashMap<u32, Vec<RawPoint>> = FxHashMap::default();

                for path in &input_files[start..end] {
                    let mut reader = las::Reader::from_path(path)
                        .with_context(|| format!("Cannot open {:?}", path))?;
                    // Per-file byte slab: `fill_points` reuses the buffer
                    // across batches but only re-checks the *format* on
                    // reuse, while the coordinate transforms (scale/offset)
                    // can legitimately differ per file — so build against
                    // each file's header.
                    let mut pd = las::PointDataBuilder::new()
                        .for_header(reader.header())
                        .build();

                    loop {
                        let n = reader.fill_points(read_chunk_size as u64, &mut pd)?;
                        if n == 0 {
                            break;
                        }

                        // Classify points into chunks via the LUT and group
                        // them in memory before writing. Grouping batches
                        // many points per writer call → fewer LRU touches
                        // and better amortization than per-point appends.
                        self.for_each_raw_point(&pd, |raw| {
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
                            let chunk_idx = match lut.cells[cell_idx] {
                                SUBSPLIT_MARKER => {
                                    // Over-target cell: descend `k` levels
                                    // deeper to pick the sub-chunk. Same
                                    // `point_to_key` refinement the planner
                                    // and per-chunk build use, so the point's
                                    // sub-chunk root is its leaf's ancestor.
                                    let s = lut.sub[&cell_idx];
                                    let deep = point_to_key(
                                        wx,
                                        wy,
                                        wz,
                                        self.cx,
                                        self.cy,
                                        self.cz,
                                        self.halfsize,
                                        grid_depth + s.k,
                                    );
                                    // Clamp the deeper voxel into the (already
                                    // clamped) ancestor cell's sub-range so a
                                    // boundary point that rounds one past the
                                    // cell can't wrap into a sibling's octant.
                                    let span = 1i32 << s.k;
                                    let local = |deep_c: i32, cell_c: usize| -> u32 {
                                        let lo = cell_c as i32 * span;
                                        (deep_c - lo).clamp(0, span - 1) as u32
                                    };
                                    let lx = local(deep.x, gx);
                                    let ly = local(deep.y, gy);
                                    let lz = local(deep.z, gz);
                                    s.base + lx + (ly << s.k) + (lz << (2 * s.k))
                                }
                                idx => idx,
                            };
                            groups.entry(chunk_idx).or_default().push(raw);
                        });

                        // Flush each group to the cache. Drain so vec capacity
                        // doesn't accumulate across batches.
                        for (chunk_idx, pts) in groups.drain() {
                            cache.append(chunk_idx, shard_dir, &pts)?;
                        }

                        let done = progress.fetch_add(n, Ordering::Relaxed) + n;
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

    /// Open a chunk file and stream every point through `f`.
    ///
    /// The build path streams points straight into its leaves HashMap
    /// instead of materialising a full intermediate Vec, so peak memory
    /// scales with the leaves map alone rather than 2× the chunk size.
    fn stream_chunk_file<F: FnMut(RawPoint) -> Result<()>>(
        &self,
        chunk_idx: u32,
        f: F,
    ) -> Result<()> {
        let path = chunk_canonical_path(&self.tmp_dir.join("chunks"), chunk_idx);
        let file = File::open(&path).with_context(|| format!("opening chunk file {:?}", path))?;
        stream_temp_batches(file, self.num_extra_bytes, self.temp_compression, f)
    }

    /// Build a single chunk's sub-octree and write its node files to canonical
    /// temp paths.
    ///
    /// Returns the list of `(VoxelKey, point_count)` for nodes at levels in
    /// `[chunk.level, chunk_leaf_depth]` that the chunk produced. The chunk
    /// root at `chunk.level` is included; coarser ancestors are left to the
    /// merge step.
    ///
    /// Dispatches on the chunk's estimated working set: chunks that fit
    /// `chunk_mem_budget` build fully in memory; larger chunks take the
    /// spilled path ([`Self::build_chunk_spilled`]), which sub-divides into
    /// bounded subtrees so peak memory stays under budget regardless of how
    /// the points are distributed inside the chunk.
    ///
    /// The dispatch reads the chunk file's **actual** point count rather than
    /// trusting the planner's `point_count`, which is only an estimate (the
    /// planner sees occupancy at grid resolution and, for a sub-split cell,
    /// divides a cell's count evenly across sub-octants). A chunk whose real
    /// distribution is far denser than the estimate — e.g. a road corridor
    /// concentrated in one sub-octant — must still take the spilled path, so
    /// the gate has to be the truth on disk. The count is one cheap streaming
    /// pass (seek-based for uncompressed temp files).
    fn build_chunk_in_memory(
        &self,
        chunk: &crate::chunking::PlannedChunk,
        chunk_idx: u32,
        chunk_mem_budget: u64,
        spill_stats: &SpillStats,
        config: &crate::PipelineConfig,
    ) -> Result<Vec<(VoxelKey, usize)>> {
        let chunk_root = VoxelKey {
            level: chunk.level as i32,
            x: chunk.gx,
            y: chunk.gy,
            z: chunk.gz,
        };
        let actual_points = self.count_chunk_file(chunk_idx)?;
        let est_bytes = actual_points.saturating_mul(PER_CHUNK_BYTES_PER_POINT_BUILD);
        if est_bytes <= chunk_mem_budget {
            self.build_subtree_in_memory(chunk_root, chunk_idx, actual_points, config)
        } else {
            spill_stats.chunks_spilled.fetch_add(1, Ordering::Relaxed);
            self.build_chunk_spilled(chunk, chunk_idx, chunk_mem_budget, spill_stats, config)
        }
    }

    /// Actual point count in a chunk's canonical temp file (0 if absent).
    /// One streaming pass over batch headers — see [`count_temp_file_points`].
    fn count_chunk_file(&self, chunk_idx: u32) -> Result<u64> {
        let path = chunk_canonical_path(&self.tmp_dir.join("chunks"), chunk_idx);
        count_temp_file_points(&path, self.num_extra_bytes, self.temp_compression)
    }

    /// Build the sub-octree rooted at `root` from every point in chunk file
    /// `chunk_idx`, keeping the whole subtree resident in memory. Used by the
    /// common (non-spilled) whole-chunk path, where every point in the file
    /// belongs under `root` (the chunk root).
    ///
    /// `est_points` seeds the initial leaf depth; the adaptive subdivision in
    /// [`Self::finish_subtree`] corrects any residual per-leaf overflow.
    fn build_subtree_in_memory(
        &self,
        root: VoxelKey,
        chunk_idx: u32,
        est_points: u64,
        config: &crate::PipelineConfig,
    ) -> Result<Vec<(VoxelKey, usize)>> {
        let leaf_depth = root.level as u32 + chunk_local_extra_levels(est_points);
        let mut leaves: FxHashMap<VoxelKey, Vec<RawPoint>> = FxHashMap::default();
        self.stream_chunk_file(chunk_idx, |raw| {
            self.push_into_leaf(&mut leaves, raw, root, leaf_depth);
            Ok(())
        })?;
        self.finish_subtree(root, leaves, leaf_depth, config)
    }

    /// Build the sub-octree rooted at `root` from an in-memory point vector.
    /// Used by the spilled path, where each sub-octant's points have already
    /// been routed into their own (budget-sized) temp file and read back
    /// whole. Consumes `points`.
    fn build_subtree_from_points(
        &self,
        root: VoxelKey,
        points: Vec<RawPoint>,
        config: &crate::PipelineConfig,
    ) -> Result<Vec<(VoxelKey, usize)>> {
        let leaf_depth = root.level as u32 + chunk_local_extra_levels(points.len() as u64);
        let mut leaves: FxHashMap<VoxelKey, Vec<RawPoint>> = FxHashMap::default();
        for raw in points {
            self.push_into_leaf(&mut leaves, raw, root, leaf_depth);
        }
        self.finish_subtree(root, leaves, leaf_depth, config)
    }

    /// Classify one point into its leaf voxel at `leaf_depth` and push it into
    /// `leaves`. In debug builds, assert the leaf's ancestor at `root.level`
    /// is `root` — the consistency contract between distribute's chunk
    /// routing and the build's leaf derivation (both use `point_to_key`).
    fn push_into_leaf(
        &self,
        leaves: &mut FxHashMap<VoxelKey, Vec<RawPoint>>,
        raw: RawPoint,
        // Only read by the debug-assertion below; unused in release builds.
        #[cfg_attr(not(debug_assertions), allow(unused_variables))] root: VoxelKey,
        leaf_depth: u32,
    ) {
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
        #[cfg(debug_assertions)]
        {
            let mut ancestor = key;
            while ancestor.level > root.level {
                ancestor = ancestor.parent().expect("leaf above root level");
            }
            debug_assert!(
                ancestor == root,
                "leaf {:?} does not belong to root {:?}: ancestor at L{} = ({},{},{})",
                key,
                root,
                ancestor.level,
                ancestor.x,
                ancestor.y,
                ancestor.z,
            );
        }
        leaves.entry(key).or_default().push(raw);
    }

    /// Shared subtree finisher: adaptive leaf subdivision + bottom-up
    /// grid-sample down to `root.level`. Writes every produced node to its
    /// canonical temp path and returns `(VoxelKey, point_count)` for each.
    fn finish_subtree(
        &self,
        root: VoxelKey,
        mut leaves: FxHashMap<VoxelKey, Vec<RawPoint>>,
        leaf_depth: u32,
        config: &crate::PipelineConfig,
    ) -> Result<Vec<(VoxelKey, usize)>> {
        let root_level = root.level as u32;
        if leaves.is_empty() {
            // Empty subtree → no nodes produced. Expected for empty sub-octant
            // slots in the spilled path; defensive otherwise.
            return Ok(Vec::new());
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
                    "Subtree (L{} {},{},{}): stopping subdivision at depth cap {} \
                     ({} leaves still oversized)",
                    root.level,
                    root.x,
                    root.y,
                    root.z,
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

        // Run the bottom-up grid-sample loop, stopping at root.level.
        // bottom_up_levels writes every produced node to its canonical temp
        // file path, so the subtree is on disk after this returns. Suppress
        // per-level progress events: the outer build stage tracks
        // chunks-done, not levels.
        self.bottom_up_levels(leaves, effective_max_depth, root_level, false, config)
    }

    /// Build an oversized chunk by sub-dividing it into bounded subtrees.
    ///
    /// The planner can only estimate a chunk's size from grid-resolution
    /// occupancy, so a chunk that fits the budget on paper may concentrate
    /// most of its points in one sub-region. This path guarantees bounded
    /// peak memory regardless of the in-chunk distribution:
    ///
    /// 1. Choose a `split_level` whose densest sub-octant fits
    ///    `chunk_mem_budget` ([`Self::choose_split_level`], one streaming
    ///    count pass — no per-level re-reads).
    /// 2. Stream the chunk file **once**, routing every point into a temp file
    ///    for its `split_level` sub-octant via a bounded append-writer cache.
    /// 3. Build each sub-octant subtree from its own (budget-sized) temp file
    ///    and merge the sub-octant roots up to the chunk root.
    ///
    /// The chunk file is read exactly twice — once to choose the split level,
    /// once to route — and each sub-octant build then reads only its own small
    /// spill file. So the whole operation is `O(chunk_size)` regardless of how
    /// deep the split goes or how many sub-octants result.
    fn build_chunk_spilled(
        &self,
        chunk: &crate::chunking::PlannedChunk,
        chunk_idx: u32,
        chunk_mem_budget: u64,
        spill_stats: &SpillStats,
        config: &crate::PipelineConfig,
    ) -> Result<Vec<(VoxelKey, usize)>> {
        let chunk_root = VoxelKey {
            level: chunk.level as i32,
            x: chunk.gx,
            y: chunk.gy,
            z: chunk.gz,
        };

        // Step 1: in ONE streaming pass, find the shallowest split depth at
        // which the densest sub-octant fits budget. `choose_split_level`
        // counts occupancy at the deepest candidate level and rolls the counts
        // up to every shallower level in memory, so the chunk file is read
        // exactly once regardless of how deep the data forces the split.
        let max_points_in_budget = (chunk_mem_budget / PER_CHUNK_BYTES_PER_POINT_BUILD).max(1);
        let (split_level, _densest) =
            self.choose_split_level(chunk_idx, chunk_root, max_points_in_budget)?;
        // Track the deepest extra split any chunk needed (aggregate; logged
        // once after phase 1). `densest > budget` only at the depth cap with
        // pathological coincident-point density — finish_subtree's leaf-depth
        // cap absorbs the residual, so no separate warning per chunk.
        spill_stats
            .max_split_extra
            .fetch_max((split_level - chunk.level) as u64, Ordering::Relaxed);

        // Step 2: route every point into its sub-octant temp file in ONE pass.
        // Sub-octant `VoxelKey`s are mapped to dense `u32` indices so the
        // existing bounded append-writer cache (`ChunkWriterCache`) can shard
        // them; `sub_roots[idx]` recovers the key. Files live under a
        // per-chunk spill dir that is removed once the chunk is built.
        let spill_dir = self.tmp_dir.join(format!("spill_{chunk_idx}"));
        if spill_dir.exists() {
            std::fs::remove_dir_all(&spill_dir)
                .with_context(|| format!("removing stale spill dir {spill_dir:?}"))?;
        }
        std::fs::create_dir_all(&spill_dir)
            .with_context(|| format!("creating spill dir {spill_dir:?}"))?;

        let mut sub_index: FxHashMap<VoxelKey, u32> = FxHashMap::default();
        let mut sub_roots: Vec<VoxelKey> = Vec::new();
        let per_worker_cap = CHUNKED_OPEN_FILES_CAP.max(MIN_PER_WORKER_CHUNK_FILES);
        let mut cache =
            ChunkWriterCache::new(per_worker_cap, self.num_extra_bytes, self.temp_compression);
        // Group points per sub-octant before appending so each cache touch
        // writes a batch rather than a single point.
        let mut pending: FxHashMap<u32, Vec<RawPoint>> = FxHashMap::default();
        let mut pending_points: usize = 0;
        // Flush threshold in points: a fraction of budget so the grouping
        // buffer never dominates memory during routing.
        let flush_threshold = ((chunk_mem_budget / 4)
            / RawPoint::record_size(self.num_extra_bytes) as u64)
            .max(1) as usize;

        self.stream_chunk_file(chunk_idx, |raw| {
            let wx = raw.x as f64 * self.scale_x + self.offset_x;
            let wy = raw.y as f64 * self.scale_y + self.offset_y;
            let wz = raw.z as f64 * self.scale_z + self.offset_z;
            let sub = point_to_key(
                wx,
                wy,
                wz,
                self.cx,
                self.cy,
                self.cz,
                self.halfsize,
                split_level,
            );
            let idx = match sub_index.get(&sub) {
                Some(i) => *i,
                None => {
                    let i = sub_roots.len() as u32;
                    sub_index.insert(sub, i);
                    sub_roots.push(sub);
                    i
                }
            };
            pending.entry(idx).or_default().push(raw);
            pending_points += 1;
            if pending_points >= flush_threshold {
                for (idx, pts) in pending.drain() {
                    cache.append(idx, &spill_dir, &pts)?;
                }
                pending_points = 0;
            }
            Ok(())
        })?;
        for (idx, pts) in pending.drain() {
            cache.append(idx, &spill_dir, &pts)?;
        }
        cache.flush_all()?;

        spill_stats
            .sub_octant_files
            .fetch_add(sub_roots.len() as u64, Ordering::Relaxed);

        // Step 3: build each sub-octant from its own temp file. Each build
        // writes its whole subtree (including the sub-octant root) to disk and
        // returns only node *keys* — we deliberately do NOT carry any points
        // back into memory, so this loop's resident set is bounded by one
        // sub-octant build at a time (the spill runs inside the bounded build
        // pool) plus the accumulating key list (tiny: ~32 B per node).
        let codec = self.temp_compression;
        let nxb = self.num_extra_bytes;
        let built: Vec<Vec<(VoxelKey, usize)>> = sub_roots
            .par_iter()
            .enumerate()
            .map(|(idx, sub_root)| -> Result<Vec<(VoxelKey, usize)>> {
                let path = chunk_shard_path(&spill_dir, idx as u32);
                let f = match File::open(&path) {
                    Ok(f) => f,
                    // A sub-octant index always has a file (we created it on
                    // first point), but be defensive.
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                        return Ok(Vec::new());
                    }
                    Err(e) => return Err(e.into()),
                };
                let points = read_temp_batches(f, nxb, codec)?;
                self.build_subtree_from_points(*sub_root, points, config)
            })
            .collect::<Result<_>>()?;

        let mut all_nodes: Vec<(VoxelKey, usize)> = Vec::new();
        let mut sub_root_keys: Vec<VoxelKey> = Vec::new();
        let split_level_i32 = split_level as i32;
        for nodes in built {
            for (k, _) in &nodes {
                if k.level == split_level_i32 {
                    sub_root_keys.push(*k);
                }
            }
            all_nodes.extend(nodes);
        }

        // The spill files are no longer needed.
        let _ = std::fs::remove_dir_all(&spill_dir);

        if sub_root_keys.is_empty() {
            return Ok(all_nodes);
        }

        // Merge the sub-octant roots (at split_level) up to the chunk root
        // (at chunk.level) using the disk-based, bounded merge — reading each
        // parent's children from disk rather than holding every sub-octant's
        // points resident. Concurrency 1: this spill is already running inside
        // one slot of the outer build pool, so it must not fan out further.
        let merged = self.merge_chunk_tops(
            &sub_root_keys,
            chunk.level as i32,
            1,
            chunk_mem_budget,
            true, // quiet: one local merge per spilled chunk — don't flood
            config,
        )?;

        // Combine deeper per-sub-octant nodes with the merged ancestors
        // (split_level down to chunk.level). `merge_chunk_tops` returns keys
        // only; the caller (`build_node_map`) re-reads every node's count from
        // disk after the global merge, so the count carried here is a
        // placeholder.
        all_nodes.extend(merged.into_iter().map(|k| (k, 0usize)));
        Ok(all_nodes)
    }

    /// Choose the shallowest sub-octant split level whose densest sub-octant
    /// holds at most `max_points` points, in a SINGLE streaming pass over the
    /// chunk file.
    ///
    /// Returns `(split_level, densest_count_at_that_level)`. The chunk file is
    /// read exactly once: every point is classified at the deepest candidate
    /// level (`chunk_root.level + SPILL_DEPTH_CAP`) and accumulated into a
    /// counting map. Because `point_to_key` is a strict refinement and
    /// `VoxelKey::parent` is coordinate-halving, the per-sub-octant count at
    /// any shallower level is recovered by right-shifting the deep coordinates
    /// — so the densest count at every candidate depth is computed in memory
    /// from that one map, with no extra file reads.
    ///
    /// Picks the shallowest level whose densest sub-octant fits `max_points`;
    /// if even the deepest level doesn't (pathological coincident-point
    /// density), returns the deepest level and lets the caller proceed (the
    /// per-subtree leaf-depth cap absorbs the residual).
    fn choose_split_level(
        &self,
        chunk_idx: u32,
        chunk_root: VoxelKey,
        max_points: u64,
    ) -> Result<(u32, u64)> {
        // Cap the extra depth below the chunk root. 8^cap is the maximum
        // sub-octant fan-out; the deep counting map is bounded by the number
        // of populated cells at this depth (≤ point count, and far smaller for
        // any real surface data, which clusters).
        const SPILL_DEPTH_CAP: u32 = 12;
        let deepest = chunk_root.level as u32 + SPILL_DEPTH_CAP;

        // Count occupancy at the deepest candidate level in one pass.
        let mut deep_counts: FxHashMap<VoxelKey, u64> = FxHashMap::default();
        self.stream_chunk_file(chunk_idx, |raw| {
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
                deepest,
            );
            *deep_counts.entry(key).or_insert(0) += 1;
            Ok(())
        })?;

        if deep_counts.is_empty() {
            // Empty chunk — nothing to split; one level deep is harmless.
            return Ok((chunk_root.level as u32 + 1, 0));
        }

        // Roll the deep counts up to each shallower level and find, for each,
        // the densest sub-octant. A key at `deepest` maps to its ancestor at
        // level `L = chunk_root.level + s` by right-shifting each coordinate by
        // `deepest - L`. Walk shallow→deep and stop at the first level that
        // fits `max_points`.
        let root_level = chunk_root.level as u32;
        let mut best: Option<(u32, u64)> = None;
        for s in 1..=SPILL_DEPTH_CAP {
            let level = root_level + s;
            let shift = deepest - level;
            let mut level_counts: FxHashMap<VoxelKey, u64> = FxHashMap::default();
            for (k, c) in &deep_counts {
                let anc = VoxelKey {
                    level: level as i32,
                    x: k.x >> shift,
                    y: k.y >> shift,
                    z: k.z >> shift,
                };
                *level_counts.entry(anc).or_insert(0) += c;
            }
            let densest = level_counts.values().copied().max().unwrap_or(0);
            best = Some((level, densest));
            if densest <= max_points {
                return Ok((level, densest));
            }
        }
        // None fit; return the deepest level probed.
        Ok(best.expect("loop ran at least once"))
    }

    /// Merge a set of root keys upward, level by level, down to `stop_level`.
    ///
    /// Starting from `root_keys` (nodes at varying levels), we walk levels in
    /// reverse and at each level group nodes by their parent and run
    /// `grid_sample` to produce the parent, reading children from disk and
    /// writing parents + per-child remainders back to disk. Nothing is held
    /// resident beyond one bounded batch of parents at a time, so peak memory
    /// is `≈ build_concurrency × per_parent_budget` regardless of how many
    /// roots there are.
    ///
    /// **Variable-level roots are handled naturally**: when level `d`'s
    /// children are processed, the set may include both original roots that
    /// happened to be at level `d+1` AND parents the merge produced from level
    /// `d+2`. They're treated identically.
    ///
    /// With `stop_level = 0` this merges all the way to the global root (used
    /// for the top-level chunk-tops merge). With a positive `stop_level` it
    /// stops once parents at `stop_level` are produced — used by the spill
    /// path to merge a chunk's deeper sub-octant roots up to the chunk root
    /// without materialising every sub-octant's points at once.
    ///
    /// Returns the merge-produced parent keys at every level from `stop_level`
    /// up to `max(root level) - 1`.
    fn merge_chunk_tops(
        &self,
        chunk_root_keys: &[VoxelKey],
        stop_level: i32,
        build_concurrency: usize,
        per_parent_budget: u64,
        // When true, suppress the per-level / per-batch debug lines. Set for
        // the spill's local sub-octant merge, which runs once per spilled chunk
        // and would otherwise flood the log; the single global merge logs.
        quiet: bool,
        config: &crate::PipelineConfig,
    ) -> Result<Vec<VoxelKey>> {
        if chunk_root_keys.is_empty() {
            return Ok(Vec::new());
        }

        // Bound merge memory the same way phase 1 does: at most
        // `build_concurrency` parents merge concurrently, each holding its
        // children's (LOD-thinned) points — capped at `per_parent_budget`. A
        // bounded pool enforces the concurrency regardless of core count. For
        // concurrency 1 (e.g. the spill's local merge, already running inside
        // one outer build slot) skip the pool and run merges inline.
        let merge_pool = if build_concurrency > 1 {
            Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(build_concurrency)
                    .build()
                    .context("building bounded merge thread pool")?,
            )
        } else {
            None
        };

        // Organise chunk roots by level. Like bottom_up_on_disk, we use a
        // HashSet per level so we never get duplicate keys (which can happen
        // if two chunks somehow share the same root key — should not occur
        // for a well-formed plan, but defensive).
        let mut keys_by_level: FxHashMap<i32, FxHashSet<VoxelKey>> = FxHashMap::default();
        for k in chunk_root_keys {
            keys_by_level.entry(k.level).or_default().insert(*k);
        }

        let max_chunk_level: i32 = keys_by_level.keys().copied().max().unwrap_or(0);
        if !quiet {
            debug!(
                "Merge chunk tops: {} chunk roots, max_chunk_level={}",
                chunk_root_keys.len(),
                max_chunk_level
            );
        }

        // Reuse the same per-point cost estimate as bottom_up_on_disk's
        // small-parent batching: input vec + per-child remaining. The
        // `2 * num_extra_bytes` term accounts for the heap-allocated
        // extras payloads (`Box<[u8]>`) — `size_of::<RawPoint>` only
        // counts the box header, not the bytes it points at, so without
        // this term the budget gate under-reports by
        // `num_extra_bytes` × points for each of the two `RawPoint`
        // instances per point.
        let mem_per_point: u64 = (std::mem::size_of::<(usize, RawPoint)>()
            + std::mem::size_of::<RawPoint>()) as u64
            + 2 * self.num_extra_bytes as u64;

        let mut all_new_parents: Vec<VoxelKey> = Vec::new();

        // Walk from the deepest level down to `stop_level`. d is the parent
        // level we're producing in this iteration.
        for d in (stop_level..max_chunk_level).rev() {
            let child_level = d + 1;
            let child_keys: Vec<VoxelKey> = match keys_by_level.get(&child_level) {
                Some(v) => v.iter().copied().collect(),
                None => continue,
            };
            if child_keys.is_empty() {
                continue;
            }

            // Group children at level d+1 by their parent at level d.
            let mut parent_children: FxHashMap<VoxelKey, Vec<VoxelKey>> = FxHashMap::default();
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

            // A merge parent has no spill path — it must be grid-sampled in
            // memory — so it is only "too large" when its children exceed what
            // is physically safe to hold: max(budget, floor). The floor keeps
            // trivially-small merges (LOD-thinned node data) working even under
            // an artificially tiny `--memory-limit`.
            let oom_threshold = config.memory_budget.max(MERGE_PARENT_OOM_FLOOR_BYTES);
            for (parent, children) in parent_children {
                let est_points: u64 = children
                    .iter()
                    .map(|ck| self.count_node(ck).unwrap_or(0))
                    .sum();
                let est_mem = est_points.saturating_mul(mem_per_point);
                if est_mem > oom_threshold {
                    large_parents.push((parent, children, est_mem));
                } else {
                    small_parents.push((parent, children, est_mem));
                }
            }

            if !quiet {
                debug!(
                    "Merge level {}→{}: {} parents ({} small, {} large), per-parent budget={} MB, concurrency={}",
                    child_level,
                    d,
                    small_parents.len() + large_parents.len(),
                    small_parents.len(),
                    large_parents.len(),
                    per_parent_budget / 1_048_576,
                    build_concurrency,
                );
            }

            if let Some((parent, children, est_mem)) = large_parents.first() {
                // A parent whose combined children exceed what is physically
                // safe to hold can't be grid-sampled in memory at all — it
                // signals either a pathological chunk plan or a budget set too
                // low for the input. Bail out with a message the user can act on.
                return Err(anyhow::anyhow!(
                    "merge parent {:?} has {} children with combined estimate {} MB, \
                     exceeding the {} MB in-memory merge limit. Raise --memory-limit \
                     or investigate the chunk plan.",
                    parent,
                    children.len(),
                    est_mem / 1_048_576,
                    oom_threshold / 1_048_576,
                ));
            }

            // Sort small parents by descending estimated memory so the
            // batching greedy stays balanced.
            small_parents.sort_by_key(|p| std::cmp::Reverse(p.2));

            // Cap each batch's combined working set at
            // `build_concurrency × per_parent_budget` so that, with the merge
            // pool capping concurrency, peak resident merge memory stays under
            // `BUILD_PEAK_FRACTION × memory_budget` — the same invariant as
            // phase 1.
            let batch_cap = (build_concurrency as u64).saturating_mul(per_parent_budget);
            let mut batch_start = 0;
            while batch_start < small_parents.len() {
                let mut batch_mem: u64 = 0;
                let mut batch_end = batch_start;
                while batch_end < small_parents.len() {
                    if batch_end > batch_start && batch_mem + small_parents[batch_end].2 > batch_cap
                    {
                        break;
                    }
                    batch_mem += small_parents[batch_end].2;
                    batch_end += 1;
                }

                let batch = &small_parents[batch_start..batch_end];
                if !quiet {
                    debug!(
                        "  Merge batch: {} parents, est {} MB",
                        batch.len(),
                        batch_mem / 1_048_576,
                    );
                }
                let process_batch = || -> Result<Vec<()>> {
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
                        .collect::<Result<Vec<_>>>()
                };
                match &merge_pool {
                    Some(pool) => pool.install(process_batch)?,
                    None => process_batch()?,
                };

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

    /// Build the node map: per-chunk in-memory build, then merge across
    /// chunks at coarse levels up to the global root.
    pub fn build_node_map(&self, config: &crate::PipelineConfig) -> Result<Vec<(VoxelKey, usize)>> {
        let plan = self
            .chunked_plan
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("build_node_map called without a chunk plan"))?;

        let n_chunks = plan.chunks.len();
        if n_chunks == 0 {
            info!("Build: no chunks (empty input?)");
            return Ok(Vec::new());
        }

        debug!(
            "Build: {} chunks, max_chunk_level={}",
            n_chunks,
            plan.chunks.iter().map(|c| c.level).max().unwrap_or(0)
        );

        // ---- Phase 1: per-chunk in-memory build ----
        //
        // Peak build memory is `(concurrent subtree builds) × (per-subtree
        // peak)`. Two things make the concurrency hard to see locally: rayon's
        // global pool runs one task per core, and a spilling chunk itself
        // fans out a nested `par_iter` over its sub-octants. Both draw from the
        // same pool, so a naive build could hold `cores × per_chunk_peak`
        // resident — or worse with nesting — and blow the budget.
        //
        // We bound it with a dedicated build pool of `build_concurrency`
        // threads and run *all* build work (top-level chunks and the spill's
        // sub-octant builds) inside it. At most `build_concurrency` subtrees
        // are ever resident, each capped at `chunk_mem_budget`, so peak ≈
        // `build_concurrency × chunk_mem_budget ≤ BUILD_PEAK_FRACTION ×
        // memory_budget`. Chunks (and sub-octants) larger than the per-subtree
        // budget take the spilled path, keeping each one bounded.
        let cores = rayon::current_num_threads();
        let (build_concurrency, chunk_mem_budget) =
            build_concurrency_and_budget(config.memory_budget, cores);
        debug!(
            "Build: concurrency={}, per-chunk budget={} MB (build working set ≈ {} MB of {} MB budget)",
            build_concurrency,
            chunk_mem_budget / 1_048_576,
            (build_concurrency as u64 * chunk_mem_budget) / 1_048_576,
            config.memory_budget / 1_048_576,
        );

        let chunks_indexed: Vec<(u32, &crate::chunking::PlannedChunk)> = plan
            .chunks
            .iter()
            .enumerate()
            .map(|(i, c)| (i as u32, c))
            .collect();

        config.report(crate::ProgressEvent::StageStart {
            name: "Building",
            total: n_chunks as u64,
        });

        let chunks_done = AtomicU64::new(0);
        let spill_stats = SpillStats::default();

        // Build pool with bounded threads. Running phase 1 inside it caps the
        // total number of concurrently-resident subtree builds — including the
        // nested sub-octant builds inside a spilling chunk, which inherit this
        // pool rather than the global one.
        let build_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(build_concurrency)
            .build()
            .context("building bounded build thread pool")?;

        let batch_results: Vec<Vec<(VoxelKey, usize)>> = build_pool.install(|| {
            chunks_indexed
                .par_iter()
                .map(|(chunk_idx, chunk)| -> Result<Vec<(VoxelKey, usize)>> {
                    let nodes = self.build_chunk_in_memory(
                        chunk,
                        *chunk_idx,
                        chunk_mem_budget,
                        &spill_stats,
                        config,
                    )?;
                    let done = chunks_done.fetch_add(1, Ordering::Relaxed) + 1;
                    config.report(crate::ProgressEvent::StageProgress { done });
                    Ok(nodes)
                })
                .collect::<Result<_>>()
        })?;

        // One aggregate line on whether/how hard the spill engaged, instead of
        // per-chunk spill logs. `chunks_spilled == 0` means every chunk fit its
        // in-memory budget; a non-zero count with a modest max split depth is
        // the spill working as intended.
        let spilled = spill_stats.chunks_spilled.load(Ordering::Relaxed);
        if spilled > 0 {
            debug!(
                "Build: {} of {} chunks spilled (max +{} levels, {} sub-octant files total)",
                spilled,
                n_chunks,
                spill_stats.max_split_extra.load(Ordering::Relaxed),
                spill_stats.sub_octant_files.load(Ordering::Relaxed),
            );
        }

        let mut all_chunk_node_keys: Vec<VoxelKey> = Vec::new();
        let mut chunk_root_keys: Vec<VoxelKey> = Vec::new();
        for ((_chunk_idx, chunk), nodes) in chunks_indexed.iter().zip(batch_results) {
            let chunk_level_i32 = chunk.level as i32;
            for (k, _) in &nodes {
                all_chunk_node_keys.push(*k);
                if k.level == chunk_level_i32 {
                    chunk_root_keys.push(*k);
                }
            }
        }

        debug!(
            "Per-chunk build done: {} chunk nodes, {} chunk roots",
            all_chunk_node_keys.len(),
            chunk_root_keys.len()
        );

        // ---- Phase 2: merge chunk tops up to global root ----
        let merge_parents = self.merge_chunk_tops(
            &chunk_root_keys,
            0, // stop at the global root
            build_concurrency,
            chunk_mem_budget,
            false, // global merge: log per-level progress
            config,
        )?;
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
                let n = self.count_node(k)? as usize;
                Ok((*k, n))
            })
            .collect::<Result<Vec<_>>>()?;

        // Drop empty nodes (e.g. chunk roots whose points were all promoted
        // by the merge and never had any remaining points written back).
        result.retain(|(_, count)| *count > 0);

        let total_pts: usize = result.iter().map(|(_, c)| *c).sum();
        debug!(
            "Build: {} nodes, {} total points (input: {}), avg {} pts/node",
            result.len(),
            total_pts,
            self.total_points,
            if result.is_empty() {
                0
            } else {
                total_pts / result.len()
            },
        );
        // Per-level node-count + point distribution, to surface
        // over-fragmentation (many tiny deep nodes) at a glance.
        {
            let mut by_level: std::collections::BTreeMap<i32, (usize, usize)> =
                std::collections::BTreeMap::new();
            for (k, c) in &result {
                let e = by_level.entry(k.level).or_insert((0, 0));
                e.0 += 1;
                e.1 += *c;
            }
            for (lvl, (n, pts)) in &by_level {
                debug!(
                    "  level {}: {} nodes, {} pts ({} avg)",
                    lvl,
                    n,
                    pts,
                    if *n == 0 { 0 } else { pts / n }
                );
            }
        }
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
        let mut present: FxHashSet<VoxelKey> = result.iter().map(|(k, _)| *k).collect();
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

    #[test]
    fn fit_offset_scale_keeps_first_file_offset_when_bounds_fit() {
        // Single-file or homogeneous-extent inputs should be bit-identical
        // to prior versions: untouched offset and scale.
        let mut offset = 122_243.6;
        let mut scale = 0.001;
        let changed = fit_offset_scale(&mut offset, &mut scale, 122_200.0, 122_287.146);
        assert!(!changed);
        assert_eq!(offset, 122_243.6);
        assert_eq!(scale, 0.001);
    }

    #[test]
    fn fit_offset_scale_recenters_when_first_file_offset_is_far() {
        // Reproduces the disjoint-multi-file Y-clumping bug: the first
        // file's offset sits at the top of the combined Y range, so the
        // bottom of the range encodes to ~-4.5e9, well past i32::MIN.
        // The fix re-centers offset on the combined midpoint; the half
        // extent then fits comfortably in i32 at the original scale.
        let mut offset = 455_444.84;
        let mut scale = 1e-5;
        let changed = fit_offset_scale(&mut offset, &mut scale, 410_823.89, 455_651.89);
        assert!(changed);
        // Re-centered.
        assert!((offset - 433_237.89).abs() < 0.01, "offset = {offset}");
        // Scale doubled once: half-extent 22414m / 1e-5 = 2.24e9 > headroom,
        // so we need 2e-5 (half-extent / 2e-5 = 1.12e9, fits).
        assert_eq!(scale, 2e-5);
        // Sanity: both endpoints now fit in i32 with the chosen offset/scale.
        let lo = ((410_823.89 - offset) / scale).round();
        let hi = ((455_651.89 - offset) / scale).round();
        assert!(lo >= i32::MIN as f64 && hi <= i32::MAX as f64);
    }

    #[test]
    fn fit_offset_scale_grows_scale_for_extreme_spans() {
        // Span so wide even a midpoint offset can't fit in i32 at the
        // input scale; helper should double until it does.
        let mut offset = 0.0;
        let mut scale = 1e-5;
        let half_extent = 100_000.0; // 100km half-span → 200km total
        let changed = fit_offset_scale(&mut offset, &mut scale, -half_extent, half_extent);
        assert!(changed);
        assert_eq!(offset, 0.0); // midpoint
        assert!(half_extent / scale <= I32_HEADROOM);
        // Should be a power-of-two multiple of the original scale.
        let ratio = (scale / 1e-5).round() as u32;
        assert!(ratio.is_power_of_two(), "scale ratio = {ratio}");
    }

    #[test]
    fn to_cube_pads_halfsize_by_one_scale_unit() {
        // Halfsize must grow by at least one scale unit to absorb the
        // per-depth ULP drift in the spec's node-bound reconstruction.
        let mut b = Bounds::empty();
        b.expand_with(184000.0, 426417.72, -13.11);
        b.expand_with(184100.0, 426500.0, 44.23);
        let scale = 0.01;
        let (_, _, _cz, halfsize) = b.to_cube(scale, scale, scale);
        assert!(
            halfsize >= 50.0 + scale,
            "halfsize {halfsize} must add scale pad"
        );
        assert!(
            halfsize <= 50.05,
            "halfsize {halfsize} must stay close to half x-range"
        );
    }

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
            extras: Box::<[u8]>::default(),
        }
    }

    fn sample_point_with_extras(extras: &[u8]) -> RawPoint {
        let mut p = sample_point();
        p.extras = extras.to_vec().into_boxed_slice();
        p
    }

    /// Round-trip a single point through write_bulk + read.
    #[test]
    fn rawpoint_roundtrip_single() {
        let p = sample_point();
        let mut buf = Vec::new();
        RawPoint::write_bulk(std::slice::from_ref(&p), 0, &mut buf).unwrap();
        assert_eq!(buf.len(), RawPoint::record_size(0));
        let p2 = RawPoint::read(&mut &buf[..], 0).unwrap();
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
        assert_eq!(&*p.extras, &*p2.extras);
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
                extras: Box::<[u8]>::default(),
            },
            sample_point(),
        ];

        let mut buf = Vec::new();
        RawPoint::write_bulk(&points, 0, &mut buf).unwrap();
        assert_eq!(buf.len(), RawPoint::record_size(0) * 3);

        // Read them back one at a time
        let mut cursor = std::io::Cursor::new(&buf[..]);
        for orig in &points {
            let p = RawPoint::read(&mut cursor, 0).unwrap();
            assert_eq!(orig.x, p.x);
            assert_eq!(orig.gps_time, p.gps_time);
            assert_eq!(orig.nir, p.nir);
        }
    }

    /// Round-trip points with trailing extras: every byte must survive
    /// the bulk write + per-point read path.
    #[test]
    fn rawpoint_roundtrip_with_extras() {
        let extras_a = b"\xAA\xBB\xCC\xDD\xEE\xFF\x00\x11\x22\x33\x44\x55";
        let extras_b = b"\xDE\xAD\xBE\xEF\xCA\xFE\xBA\xBE\x01\x02\x03\x04";
        let extras_c = b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
        let num = 12u16;

        let points = vec![
            sample_point_with_extras(extras_a),
            sample_point_with_extras(extras_b),
            sample_point_with_extras(extras_c),
        ];

        let mut buf = Vec::new();
        RawPoint::write_bulk(&points, num, &mut buf).unwrap();
        assert_eq!(buf.len(), RawPoint::record_size(num) * 3);

        let mut cursor = std::io::Cursor::new(&buf[..]);
        let decoded: Vec<RawPoint> = (0..3)
            .map(|_| RawPoint::read(&mut cursor, num).unwrap())
            .collect();
        assert_eq!(&*decoded[0].extras, extras_a);
        assert_eq!(&*decoded[1].extras, extras_b);
        assert_eq!(&*decoded[2].extras, extras_c);
    }

    /// Stream round-trip with extras through write_temp_batch +
    /// read_temp_batches under both compression codecs.
    #[test]
    fn temp_batch_roundtrip_with_extras() {
        let num = 8u16;
        let points: Vec<RawPoint> = (0..5)
            .map(|i| sample_point_with_extras(&[i as u8; 8]))
            .collect();

        for codec in [crate::TempCompression::None, crate::TempCompression::Lz4] {
            let mut buf = Vec::new();
            write_temp_batch(&mut buf, &points, num, codec).unwrap();
            let decoded = read_temp_batches(std::io::Cursor::new(&buf[..]), num, codec).unwrap();
            assert_eq!(decoded.len(), points.len());
            for (orig, got) in points.iter().zip(decoded.iter()) {
                assert_eq!(&*orig.extras, &*got.extras, "extras must survive {codec:?}");
            }
        }
    }

    /// A single batch larger than `MAX_POINTS_PER_READ` must decode across
    /// multiple sub-batch reads without losing or corrupting points.
    #[test]
    fn batch_larger_than_subread_cap_round_trips() {
        let n = MAX_POINTS_PER_READ + 1_000;
        let points: Vec<RawPoint> = (0..n)
            .map(|i| {
                let mut p = sample_point();
                p.x = i as i32;
                p.gps_time = i as f64;
                p
            })
            .collect();

        for codec in [crate::TempCompression::None, crate::TempCompression::Lz4] {
            let mut buf = Vec::new();
            write_temp_batch(&mut buf, &points, 0, codec).unwrap();
            let decoded = read_temp_batches(std::io::Cursor::new(&buf[..]), 0, codec).unwrap();
            assert_eq!(decoded.len(), n, "point count must survive {codec:?}");
            // Spot-check fields either side of the sub-read boundary.
            for i in [0, MAX_POINTS_PER_READ - 1, MAX_POINTS_PER_READ, n - 1] {
                assert_eq!(decoded[i].x, i as i32, "x at {i} ({codec:?})");
                assert_eq!(decoded[i].gps_time, i as f64, "gps_time at {i} ({codec:?})");
                assert_eq!(decoded[i].nir, 32768, "nir at {i} ({codec:?})");
            }
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
            write_temp_batch(&mut buf, batch, 0, codec).unwrap();
        }

        let decoded = read_temp_batches(std::io::Cursor::new(&buf[..]), 0, codec).unwrap();
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
                write_temp_batch(&mut f, batch, 0, codec).unwrap();
            }
            f.flush().unwrap();
        }
        let got = count_temp_file_points(&tmp, 0, codec).unwrap();
        let _ = std::fs::remove_file(&tmp);
        assert_eq!(got, expected, "multi-frame count must match total");
    }

    #[test]
    fn rawpoint_record_size_arithmetic() {
        // Base record is 38 bytes; extras shift the total by exactly the
        // declared width.
        assert_eq!(RawPoint::record_size(0), 38);
        assert_eq!(RawPoint::record_size(12), 50);
        assert_eq!(RawPoint::record_size(255), 38 + 255);
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
            header_mismatch: None,
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
        assert_eq!(lut.cells.len(), 64);
        assert!(lut.cells.iter().all(|&c| c == 0));
        assert!(lut.sub.is_empty());
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
        assert_eq!(lut.cells.len(), 8);
        assert!(lut.sub.is_empty());
        // Each cell should map to a distinct chunk index 0..8.
        let mut seen: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for &c in &lut.cells {
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
        assert_eq!(lut.cells.len(), 64);
        assert!(lut.sub.is_empty());

        // Cells (gx, gy, gz) with all coords in 0..2 should map to chunk 0.
        for z in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    let idx = x + y * 4 + z * 16;
                    assert_eq!(
                        lut.cells[idx], 0,
                        "cell ({x},{y},{z}) should map to chunk 0"
                    );
                }
            }
        }

        // Cell (3, 3, 3) → chunk 1.
        assert_eq!(lut.cells[3 + 3 * 4 + 3 * 16], 1);

        // Count uncovered cells: should be 64 - 8 - 1 = 55.
        let uncovered = lut.cells.iter().filter(|&&c| c == u32::MAX).count();
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
    fn build_concurrency_and_budget_scales() {
        let gib = 1024u64 * 1024 * 1024;
        let min_per_chunk = MAX_LEAF_POINTS * PER_CHUNK_BYTES_PER_POINT_BUILD;
        for &budget in &[gib, 2 * gib, 9 * gib, 32 * gib, 64 * gib, 256 * gib] {
            for &cores in &[1usize, 2, 8, 10, 64] {
                let (c, per) = build_concurrency_and_budget(budget, cores);

                // Concurrency in [1, min(cores, MAX_BUILD_CONCURRENCY)].
                assert!(
                    c >= 1,
                    "concurrency must be ≥ 1 (budget={budget}, cores={cores})"
                );
                assert!(
                    c <= cores.min(MAX_BUILD_CONCURRENCY),
                    "concurrency {c} exceeds min(cores={cores}, cap={MAX_BUILD_CONCURRENCY})"
                );

                // Live build working set = concurrency × per-chunk budget. It
                // must fit BUILD_WORKING_SET_FRACTION × budget — except when a
                // budget is too small to fund even one minimum slot, where we
                // accept one slot at the floor (a chunk can't build in less).
                let working_set = c as u64 * per;
                let pool = (budget as f64 * BUILD_WORKING_SET_FRACTION) as u64;
                assert!(
                    working_set <= pool || (c == 1 && per == min_per_chunk),
                    "working set {working_set} exceeds pool {pool} \
                     (budget={budget}, cores={cores}, c={c}, per={per})"
                );

                // Each slot is at least one usable leaf set.
                assert!(
                    per >= min_per_chunk,
                    "per-chunk {per} below floor {min_per_chunk}"
                );
            }
        }

        // Tight budget → a single large slot (avoids spilling), not many tiny.
        let (c1, per1) = build_concurrency_and_budget(gib, 64);
        assert_eq!(c1, 1, "1 GB budget should use a single build slot");
        assert!(
            per1 >= MIN_BUILD_SLOT_BYTES / 2,
            "single slot should still be sizeable"
        );

        // Abundant budget + cores → fans out to the cap with large slots.
        let (c2, per2) = build_concurrency_and_budget(256 * gib, 64);
        assert_eq!(
            c2, MAX_BUILD_CONCURRENCY,
            "huge budget should reach the concurrency cap"
        );
        assert!(
            per2 > MIN_BUILD_SLOT_BYTES * 4,
            "huge budget should give large per-slot budgets, got {per2}"
        );

        // More budget never reduces concurrency (monotonic) at fixed cores.
        let mut prev = 0;
        for &g in &[1u64, 2, 4, 8, 16, 32, 64, 128, 256] {
            let (c, _) = build_concurrency_and_budget(g * gib, 64);
            assert!(
                c >= prev,
                "concurrency dropped as budget grew ({g} GB → {c} < {prev})"
            );
            prev = c;
        }
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
        assert_eq!(lut.cells.len(), 512);
        assert!(lut.sub.is_empty());
        let g = 8;
        for z in 6..8 {
            for y in 6..8 {
                for x in 6..8 {
                    let idx = x + y * g + z * g * g;
                    assert_eq!(
                        lut.cells[idx], 0,
                        "cell ({x},{y},{z}) should be in the chunk"
                    );
                }
            }
        }
        // No other cell should be in the chunk.
        let in_chunk = lut.cells.iter().filter(|&&c| c == 0).count();
        assert_eq!(in_chunk, 8);
    }

    #[test]
    fn build_chunk_lut_subsplit_cell() {
        // 4³ grid (depth 2). Cell (1,2,3) was sub-split into k=1 → 8 deeper
        // sub-chunks at level 3, coords (2..4, 4..6, 6..8) in z,y,x order
        // (matching chunking::emit_chunk_root). All other cells map to a
        // single normal chunk so the grid is fully covered.
        let k = 1u32;
        let (cgx, cgy, cgz) = (1i32, 2, 3);
        let mut chunks = vec![PlannedChunk {
            level: 2,
            gx: 0,
            gy: 0,
            gz: 0,
            point_count: 10,
        }];
        // Sub-chunks emitted in z-major, then y, then x — same as the planner.
        for sz in 0..2 {
            for sy in 0..2 {
                for sx in 0..2 {
                    chunks.push(PlannedChunk {
                        level: 2 + k,
                        gx: (cgx << k) | sx,
                        gy: (cgy << k) | sy,
                        gz: (cgz << k) | sz,
                        point_count: 1_000_000,
                    });
                }
            }
        }
        let plan = make_plan(4, chunks);
        let lut = build_chunk_lut(&plan);

        // The sub-split cell is marked and has a descriptor.
        let cell_idx = cgx as usize + cgy as usize * 4 + cgz as usize * 16;
        assert_eq!(lut.cells[cell_idx], SUBSPLIT_MARKER);
        let s = lut
            .sub
            .get(&cell_idx)
            .expect("sub-split descriptor present");
        assert_eq!(s.k, k);
        // The 8 sub-chunks occupy plan indices 1..9 (chunk 0 is the normal
        // one), and the base is the index of local octant (0,0,0) = 1.
        assert_eq!(s.base, 1);
        assert_eq!(lut.sub.len(), 1);

        // Routing: for each local octant, base + sub_octant_index reproduces
        // the global chunk index assigned by enumerate() order.
        for sz in 0..2i32 {
            for sy in 0..2i32 {
                for sx in 0..2i32 {
                    let gx = (cgx << k) | sx;
                    let gy = (cgy << k) | sy;
                    let gz = (cgz << k) | sz;
                    let octant = sub_octant_index(gx, gy, gz, k);
                    let global = s.base + octant;
                    // Expected: emission order index + 1 (chunk 0 is normal).
                    let expected = 1 + (sx + sy * 2 + sz * 4) as u32;
                    assert_eq!(global, expected, "octant ({sx},{sy},{sz}) routes wrong");
                }
            }
        }
    }
}
