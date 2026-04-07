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
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::{debug, info};

/// Maximum number of leaf temp files kept open simultaneously **across all
/// distribute workers**. Divided evenly between workers, with a small floor.
///
/// Each entry holds a `BufWriter<File>` (~8 KiB internal buffer) plus an OS
/// file descriptor. Capping this lets us process datasets with millions of
/// leaves without exhausting either memory or the FD ulimit (default 256 on
/// macOS, 1024 on most Linux containers).
const DISTRIBUTE_OPEN_FILES_CAP: usize = 512;

/// Minimum per-worker LRU capacity. Below this the cache thrashes.
const MIN_PER_WORKER_OPEN_FILES: usize = 32;

/// LRU cache of append-mode `BufWriter`s for leaf temp files.
///
/// On eviction the writer is flushed and dropped, releasing both its buffer
/// and its file descriptor. Subsequent writes to the same key reopen the file
/// in append mode.
struct LeafWriterCache {
    writers: HashMap<VoxelKey, BufWriter<File>>,
    /// Insertion / access order. Front = least recently used.
    order: VecDeque<VoxelKey>,
    capacity: usize,
}

impl LeafWriterCache {
    fn new(capacity: usize) -> Self {
        LeafWriterCache {
            writers: HashMap::new(),
            order: VecDeque::new(),
            capacity,
        }
    }

    /// Append the given points to the temp file for `key`, opening (and
    /// possibly evicting another entry) as needed.
    fn append(&mut self, key: VoxelKey, tmp_dir: &Path, points: &[RawPoint]) -> Result<()> {
        if points.is_empty() {
            return Ok(());
        }

        if self.writers.contains_key(&key) {
            // Move to back (most-recently-used).
            if let Some(pos) = self.order.iter().position(|k| *k == key) {
                self.order.remove(pos);
            }
            self.order.push_back(key);
        } else {
            // Evict if at capacity.
            while self.writers.len() >= self.capacity {
                if let Some(victim) = self.order.pop_front() {
                    if let Some(mut w) = self.writers.remove(&victim) {
                        w.flush().context("flush evicted distribute writer")?;
                    }
                } else {
                    break;
                }
            }
            let path = tmp_dir.join(format!("{}_{}_{}_{}", key.level, key.x, key.y, key.z));
            let f = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .with_context(|| format!("Cannot open leaf file {:?}", path))?;
            self.writers.insert(key, BufWriter::new(f));
            self.order.push_back(key);
        }

        let w = self.writers.get_mut(&key).expect("just inserted");
        RawPoint::write_bulk(points, w)?;
        Ok(())
    }

    /// Flush and drop every cached writer.
    fn flush_all(&mut self) -> Result<()> {
        for (_, mut w) in self.writers.drain() {
            w.flush().context("flush distribute writer")?;
        }
        self.order.clear();
        Ok(())
    }
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
    /// Initial octree depth before normalization.
    pub depth: u32,
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
            depth,
            scale_x,
            scale_y,
            scale_z,
            offset_x,
            offset_y,
            offset_z,
            tmp_dir,
            wkt_crs: validated.wkt_crs.clone(),
            point_format: validated.point_format,
        })
    }

    /// Path for a node's temp file.
    fn node_path(&self, key: &VoxelKey) -> PathBuf {
        self.tmp_dir
            .join(format!("{}_{}_{}_{}", key.level, key.x, key.y, key.z))
    }

    /// Convert a `las::Point` to a `RawPoint` using the builder's scale/offset.
    fn convert_point(&self, p: &las::Point) -> RawPoint {
        let ix = ((p.x - self.offset_x) / self.scale_x).round() as i32;
        let iy = ((p.y - self.offset_y) / self.scale_y).round() as i32;
        let iz = ((p.z - self.offset_z) / self.scale_z).round() as i32;
        RawPoint {
            x: ix,
            y: iy,
            z: iz,
            intensity: p.intensity,
            return_number: p.return_number,
            number_of_returns: p.number_of_returns,
            classification: p.classification.into(),
            scan_angle: (p.scan_angle / 0.006).round() as i16,
            user_data: p.user_data,
            point_source_id: p.point_source_id,
            gps_time: p.gps_time.unwrap_or(0.0),
            red: p.color.as_ref().map(|c| c.red).unwrap_or(0),
            green: p.color.as_ref().map(|c| c.green).unwrap_or(0),
            blue: p.color.as_ref().map(|c| c.blue).unwrap_or(0),
            nir: p.nir.unwrap_or(0),
        }
    }

    /// Sequential key assignment + coordinate conversion for a batch of
    /// points. Used by the parallel distribute workers; outer parallelism
    /// across files provides the cores, so this loop stays single-threaded
    /// to avoid nested rayon scheduling overhead.
    fn classify_points(&self, points: &[las::Point]) -> Vec<(VoxelKey, RawPoint)> {
        points
            .iter()
            .map(|p| {
                let raw = self.convert_point(p);
                let rx = raw.x as f64 * self.scale_x + self.offset_x;
                let ry = raw.y as f64 * self.scale_y + self.offset_y;
                let rz = raw.z as f64 * self.scale_z + self.offset_z;
                let key = point_to_key(
                    rx,
                    ry,
                    rz,
                    self.cx,
                    self.cy,
                    self.cz,
                    self.halfsize,
                    self.depth,
                );
                (key, raw)
            })
            .collect()
    }

    /// Group a classified batch by voxel key and write each group to its
    /// cached writer in one bulk call. The per-batch HashMap is dropped on
    /// return so no `Vec` capacity accumulates across batches.
    fn write_classified_batch(
        classified: Vec<(VoxelKey, RawPoint)>,
        cache: &mut LeafWriterCache,
        shard_dir: &Path,
    ) -> Result<()> {
        let mut groups: HashMap<VoxelKey, Vec<RawPoint>> = HashMap::new();
        for (key, raw) in classified {
            groups.entry(key).or_default().push(raw);
        }
        // Drain so the inner Vecs are moved out and freed at end of iteration,
        // and the HashMap itself is dropped at end of function.
        for (key, pts) in groups.drain() {
            cache.append(key, shard_dir, &pts)?;
        }
        Ok(())
    }

    /// Compute the number of parallel distribute workers based on available
    /// cores and the input file count.
    ///
    /// We use ~2/3 of the cores so the `las` crate's parallel LAZ decoder
    /// (enabled via the `laz-parallel` feature) still has headroom for
    /// in-file decompression without oversubscribing the CPU. Empirically,
    /// `cores / 2` left ~30% of cores idle on an 8-core pod, while `cores`
    /// risks oversubscription. `cores * 2 / 3` is the sweet spot.
    fn distribute_worker_count(input_file_count: usize) -> usize {
        let cores = rayon::current_num_threads();
        let target = ((cores * 2) / 3).max(2);
        target.min(input_file_count).max(1)
    }

    /// Pass 2: assign all points to leaf temp files.
    ///
    /// Input files are processed in parallel by N workers, each with its own
    /// `LeafWriterCache` writing into a private shard subdirectory. After all
    /// workers finish, `merge_distribute_shards()` concatenates the per-worker
    /// shard files into the canonical `tmp_dir/{level}_{x}_{y}_{z}` files
    /// expected by `normalize_leaves` and `build_node_map`.
    ///
    /// Each worker reads its files in fixed-size chunks sized from the memory
    /// budget, so peak memory is bounded regardless of input file size. The
    /// number of open leaf temp files is bounded by `DISTRIBUTE_OPEN_FILES_CAP`
    /// across all workers combined.
    pub fn distribute(&self, input_files: &[PathBuf], config: &PipelineConfig) -> Result<()> {
        let num_workers = Self::distribute_worker_count(input_files.len());

        // Size each chunk so one worker's transient memory
        //   `Vec<las::Point>` (~120 B/pt) +
        //   `Vec<(VoxelKey, RawPoint)>` (~50 B/pt) +
        //   per-voxel grouping HashMap (~50 B/pt)
        // stays within (budget / 8 / num_workers). The /8 leaves headroom
        // for the writer caches, rayon thread-local allocations, and the
        // decompression buffers held inside `las::Reader`.
        const BYTES_PER_POINT_TRANSIENT: u64 = 120 + 50 + 50;
        let per_worker_budget = config.memory_budget / num_workers as u64;
        let chunk_size =
            ((per_worker_budget / 8) / BYTES_PER_POINT_TRANSIENT).clamp(50_000, 2_000_000) as usize;

        let per_worker_cap =
            (DISTRIBUTE_OPEN_FILES_CAP / num_workers).max(MIN_PER_WORKER_OPEN_FILES);

        debug!(
            "Distribute: budget={} MB, workers={}, chunk_size={} points, open-file cap={} (per worker)",
            config.memory_budget / 1_048_576,
            num_workers,
            chunk_size,
            per_worker_cap
        );

        // Create shard subdirectories under the temp dir.
        let shards_root = self.tmp_dir.join("shards");
        if shards_root.exists() {
            std::fs::remove_dir_all(&shards_root)
                .with_context(|| format!("removing stale shards dir {:?}", shards_root))?;
        }
        std::fs::create_dir_all(&shards_root)?;
        let shard_dirs: Vec<PathBuf> = (0..num_workers)
            .map(|w| {
                let d = shards_root.join(w.to_string());
                std::fs::create_dir_all(&d)?;
                Ok(d)
            })
            .collect::<Result<Vec<_>>>()?;

        // Split the input file list into one slice per worker. Round-robin
        // would balance file-size variance better, but contiguous slices keep
        // logging readable and the difference is small in practice because
        // each worker processes many files.
        let files_per_worker = input_files.len().div_ceil(num_workers);
        let progress = AtomicU64::new(0);

        shard_dirs
            .par_iter()
            .enumerate()
            .try_for_each(|(worker_id, shard_dir)| -> Result<()> {
                let start = worker_id * files_per_worker;
                let end = (start + files_per_worker).min(input_files.len());
                if start >= end {
                    return Ok(());
                }
                let mut cache = LeafWriterCache::new(per_worker_cap);
                let mut points: Vec<las::Point> = Vec::with_capacity(chunk_size);

                for path in &input_files[start..end] {
                    let mut reader = las::Reader::from_path(path)
                        .with_context(|| format!("Cannot open {:?}", path))?;

                    let file_point_count = reader.header().number_of_points();
                    debug!(
                        "Distribute worker {} file {:?}: {} points",
                        worker_id,
                        path.file_name().unwrap_or_default(),
                        file_point_count
                    );

                    loop {
                        points.clear();
                        let n = reader.read_points_into(chunk_size as u64, &mut points)?;
                        if n == 0 {
                            break;
                        }
                        let classified = self.classify_points(&points);
                        // Free the las::Point chunk before building the per-voxel
                        // grouping map so the two large allocations don't coexist.
                        points.clear();
                        Self::write_classified_batch(classified, &mut cache, shard_dir)?;
                        let done = progress.fetch_add(n, Ordering::Relaxed) + n;
                        config.report(crate::ProgressEvent::StageProgress { done });
                    }
                }

                cache.flush_all()
            })?;

        self.merge_distribute_shards(&shards_root, num_workers)?;
        Ok(())
    }

    /// Concatenate per-worker shard files into the canonical
    /// `tmp_dir/{level}_{x}_{y}_{z}` location and remove the shards directory.
    ///
    /// Runs in parallel over the union of voxel keys present in any shard.
    /// For each key, opens an append-mode writer on the canonical path and
    /// streams every shard's contribution into it via `std::io::copy`.
    fn merge_distribute_shards(&self, shards_root: &Path, num_workers: usize) -> Result<()> {
        // Collect the union of voxel keys across all shards. Each filename is
        // `{level}_{x}_{y}_{z}` (parsed by `all_node_keys`).
        let mut keys: HashSet<VoxelKey> = HashSet::new();
        for w in 0..num_workers {
            let shard_dir = shards_root.join(w.to_string());
            let dir_iter = match std::fs::read_dir(&shard_dir) {
                Ok(it) => it,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
                Err(e) => return Err(e.into()),
            };
            for entry in dir_iter {
                let entry = entry?;
                let name = entry.file_name().into_string().unwrap_or_default();
                let parts: Vec<&str> = name.split('_').collect();
                if parts.len() == 4
                    && let (Ok(l), Ok(x), Ok(y), Ok(z)) = (
                        parts[0].parse::<i32>(),
                        parts[1].parse::<i32>(),
                        parts[2].parse::<i32>(),
                        parts[3].parse::<i32>(),
                    )
                {
                    keys.insert(VoxelKey { level: l, x, y, z });
                }
            }
        }

        debug!(
            "Distribute merge: {} unique voxel keys across {} shards",
            keys.len(),
            num_workers
        );

        let keys: Vec<VoxelKey> = keys.into_iter().collect();
        keys.par_iter().try_for_each(|key| -> Result<()> {
            let canonical = self.node_path(key);
            let f = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&canonical)
                .with_context(|| format!("opening canonical leaf file {:?}", canonical))?;
            let mut out = BufWriter::new(f);
            for w in 0..num_workers {
                let shard_path = shards_root
                    .join(w.to_string())
                    .join(format!("{}_{}_{}_{}", key.level, key.x, key.y, key.z));
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
            out.flush().context("flush merged leaf file")?;
            Ok(())
        })?;

        std::fs::remove_dir_all(shards_root)
            .with_context(|| format!("removing shards root {:?}", shards_root))?;
        Ok(())
    }

    /// Read all raw points for a given node key from disk.
    pub fn read_node(&self, key: &VoxelKey) -> Result<Vec<RawPoint>> {
        let path = self.node_path(key);
        let f = match File::open(&path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(vec![]),
            Err(e) => return Err(e.into()),
        };
        let file_len = f.metadata()?.len();
        let count = file_len as usize / RawPoint::BYTE_SIZE;
        let mut r = BufReader::new(f);
        let mut pts = Vec::with_capacity(count);
        for _ in 0..count {
            pts.push(RawPoint::read(&mut r)?);
        }
        Ok(pts)
    }

    /// Stream points for a node key directly into an existing
    /// `Vec<(usize, RawPoint)>`, tagging each with `ci`.
    ///
    /// Unlike `read_node` this avoids allocating an intermediate
    /// `Vec<RawPoint>`, so the caller's accumulator and the file's points
    /// never coexist as two separate allocations.
    pub fn read_node_into_indexed(
        &self,
        key: &VoxelKey,
        ci: usize,
        sink: &mut Vec<(usize, RawPoint)>,
    ) -> Result<()> {
        let path = self.node_path(key);
        let f = match File::open(&path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(e.into()),
        };
        let file_len = f.metadata()?.len();
        let count = file_len as usize / RawPoint::BYTE_SIZE;
        sink.reserve(count);
        let mut r = BufReader::new(f);
        for _ in 0..count {
            sink.push((ci, RawPoint::read(&mut r)?));
        }
        Ok(())
    }

    /// Write points to a temp file for the given node key (overwrites if exists).
    pub fn write_node_to_temp(&self, key: &VoxelKey, points: &[RawPoint]) -> Result<()> {
        use std::io::Write;
        let path = self.node_path(key);
        let f = File::create(&path)?;
        let mut w = BufWriter::new(f);
        RawPoint::write_bulk(points, &mut w)?;
        w.flush().context("flush node temp file")?;
        Ok(())
    }

    /// Enumerate all node keys that have a non-empty temp file.
    fn all_node_keys(&self) -> Result<Vec<VoxelKey>> {
        let mut keys = Vec::new();
        for entry in std::fs::read_dir(&self.tmp_dir)? {
            let entry = entry?;
            let name = entry.file_name().into_string().unwrap_or_default();
            let parts: Vec<&str> = name.split('_').collect();
            if parts.len() == 4
                && let (Ok(l), Ok(x), Ok(y), Ok(z)) = (
                    parts[0].parse::<i32>(),
                    parts[1].parse::<i32>(),
                    parts[2].parse::<i32>(),
                    parts[3].parse::<i32>(),
                )
            {
                keys.push(VoxelKey { level: l, x, y, z });
            }
        }
        Ok(keys)
    }

    /// Phase 0 of build_node_map: split any leaf file that exceeds MAX_LEAF_POINTS.
    ///
    /// All nodes being split in one round occupy disjoint voxels, so their child
    /// files never conflict — the entire round is processed in parallel via rayon.
    /// Rounds repeat until no oversized nodes remain.
    fn normalize_leaves(&self, memory_budget: u64, config: &crate::PipelineConfig) -> Result<()> {
        let oversized = |k: &VoxelKey| -> bool {
            self.node_path(k)
                .metadata()
                .is_ok_and(|m| m.len() / RawPoint::BYTE_SIZE as u64 > MAX_LEAF_POINTS)
        };

        let mut to_split: Vec<VoxelKey> = self
            .all_node_keys()?
            .into_iter()
            .filter(|k| oversized(k))
            .collect();

        if to_split.is_empty() {
            return Ok(());
        }

        // Report normalize as its own stage. We don't know the total upfront
        // (each round can produce new oversized children), so use 0 = spinner.
        config.report(crate::ProgressEvent::StageStart {
            name: "Normalizing",
            total: 0,
        });

        // Points per chunk when streaming a split. Each rayon thread holds one
        // chunk (~40 bytes/point) plus a HashMap of classified output (~40 bytes/
        // point + hash overhead ≈ 1.3x), so peak per thread ≈ 2.5x chunk size.
        // Divide the budget by (threads × 2.5) to size chunks so all threads
        // can be active simultaneously without exceeding the budget.
        let num_threads = rayon::current_num_threads() as u64;
        let per_thread_budget = memory_budget / num_threads.max(1);
        // 2.5x overhead → divide by 3 for safety margin
        let chunk_points = ((per_thread_budget / 3) as usize / RawPoint::BYTE_SIZE).max(100_000);

        let mut round = 0u32;
        while !to_split.is_empty() {
            round += 1;
            let split_count = to_split.len();
            let total_split_bytes: u64 = to_split
                .iter()
                .map(|k| self.node_path(k).metadata().map_or(0, |m| m.len()))
                .sum();
            info!(
                "normalize round {round}: splitting {} nodes ({} MB on disk)",
                split_count,
                total_split_bytes / 1_048_576,
            );

            // All nodes in a round occupy disjoint voxels → their children
            // are also disjoint → safe to split in parallel. Each node is
            // streamed in chunks so memory stays bounded regardless of node size.
            let new_children: Vec<VoxelKey> = to_split
                .par_iter()
                .map(|key| -> Result<Vec<VoxelKey>> {
                    let path = self.node_path(key);
                    let f = File::open(&path)?;
                    let file_len = f.metadata()?.len();
                    let total_count = file_len as usize / RawPoint::BYTE_SIZE;
                    let mut reader = BufReader::new(f);

                    // Open child writers in append mode so successive chunks
                    // accumulate into the same files.
                    let child_level = key.level + 1;
                    let mut child_writers: HashMap<VoxelKey, BufWriter<File>> = HashMap::new();
                    let mut child_keys_set: HashSet<VoxelKey> = HashSet::new();

                    let mut read_so_far = 0usize;
                    while read_so_far < total_count {
                        let n = chunk_points.min(total_count - read_so_far);
                        let mut chunk = Vec::with_capacity(n);
                        for _ in 0..n {
                            chunk.push(RawPoint::read(&mut reader)?);
                        }
                        read_so_far += n;

                        // Classify and append to child files.
                        let mut per_child: HashMap<VoxelKey, Vec<RawPoint>> = HashMap::new();
                        for p in chunk {
                            let wx = p.x as f64 * self.scale_x + self.offset_x;
                            let wy = p.y as f64 * self.scale_y + self.offset_y;
                            let wz = p.z as f64 * self.scale_z + self.offset_z;
                            let ck = point_to_key(
                                wx,
                                wy,
                                wz,
                                self.cx,
                                self.cy,
                                self.cz,
                                self.halfsize,
                                child_level as u32,
                            );
                            per_child.entry(ck).or_default().push(p);
                        }

                        for (ck, pts) in per_child {
                            child_keys_set.insert(ck);
                            let w = child_writers.entry(ck).or_insert_with(|| {
                                let child_path = self.node_path(&ck);
                                let f = OpenOptions::new()
                                    .create(true)
                                    .append(true)
                                    .open(&child_path)
                                    .expect("Cannot open child temp file");
                                BufWriter::new(f)
                            });
                            RawPoint::write_bulk(&pts, w)?;
                        }
                    }

                    // Flush all child writers.
                    for (_, mut w) in child_writers {
                        std::io::Write::flush(&mut w).context("flush child temp file")?;
                    }

                    // Remove the parent file now that all data has been written to children.
                    std::fs::remove_file(&path)?;

                    Ok(child_keys_set.into_iter().collect())
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect();

            to_split = new_children.into_iter().filter(|k| oversized(k)).collect();
        }
        config.report(crate::ProgressEvent::StageDone);
        Ok(())
    }

    /// Build the full node hierarchy.
    ///
    /// Phase 0: normalize leaves in parallel.
    /// Phase 1: determine actual max depth.
    /// Phase 2: bottom-up ancestor building.
    ///   Fast path  – if all leaf data fits in the memory budget, everything is
    ///                processed in RAM (one disk read + one disk write total).
    ///   Slow path  – level-by-level with disk I/O, but without redundant directory
    ///                scans (keys are tracked in a per-level map).
    ///
    /// Returns (VoxelKey, point_count) for every node in the hierarchy.
    pub fn build_node_map(&self, config: &crate::PipelineConfig) -> Result<Vec<(VoxelKey, usize)>> {
        // Phase 0
        self.normalize_leaves(config.memory_budget, config)?;

        // Phase 1
        let leaf_keys = self.all_node_keys()?;
        let actual_max_depth = leaf_keys.iter().map(|k| k.level as u32).max().unwrap_or(0);
        info!(
            "Leaf nodes after normalization: {}, max depth: {actual_max_depth}",
            leaf_keys.len()
        );

        // Estimate memory required to hold all leaf data. The in-memory path
        // keeps all points in a HashMap<VoxelKey, Vec<RawPoint>> and during
        // grid_sample temporarily holds both input and output, so the peak is
        // roughly 2x the raw data size.
        let total_bytes: u64 = leaf_keys
            .iter()
            .map(|k| self.node_path(k).metadata().map_or(0, |m| m.len()))
            .sum();
        let estimated_peak = total_bytes * 2;

        config.report(crate::ProgressEvent::StageStart {
            name: "Building",
            total: actual_max_depth as u64,
        });

        // Phase 2
        let result = if estimated_peak <= config.memory_budget {
            info!(
                "Build path: IN-MEMORY (leaf data={} MB, est peak={} MB, budget={} MB)",
                total_bytes / 1_048_576,
                estimated_peak / 1_048_576,
                config.memory_budget / 1_048_576,
            );
            self.bottom_up_in_memory(&leaf_keys, actual_max_depth, config)?
        } else {
            info!(
                "Build path: OUT-OF-CORE (leaf data={} MB, est peak={} MB > budget={} MB)",
                total_bytes / 1_048_576,
                estimated_peak / 1_048_576,
                config.memory_budget / 1_048_576,
            );
            self.bottom_up_on_disk(leaf_keys, actual_max_depth, config.memory_budget, config)?
        };
        config.report(crate::ProgressEvent::StageDone);

        let total_pts: usize = result.iter().map(|(_, c)| *c).sum();
        info!(
            "Total octree nodes: {}, total points: {} (original: {})",
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

        // Ensure every ancestor of every data node is present in the hierarchy
        // (empty ancestors allow validators to traverse the tree top-down).
        let mut result = result;
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

    /// Bottom-up pass — in-memory fast path.
    ///
    /// Loads all leaf data into a HashMap, runs grid_sample at every level
    /// without any intermediate disk I/O, then writes the final node files once.
    fn bottom_up_in_memory(
        &self,
        leaf_keys: &[VoxelKey],
        actual_max_depth: u32,
        config: &crate::PipelineConfig,
    ) -> Result<Vec<(VoxelKey, usize)>> {
        debug!(
            "In-memory build: loading {} leaf nodes in parallel",
            leaf_keys.len()
        );
        // Load all leaf nodes in parallel.
        let mut nodes: HashMap<VoxelKey, Vec<RawPoint>> = leaf_keys
            .par_iter()
            .map(|k| -> Result<(VoxelKey, Vec<RawPoint>)> { Ok((*k, self.read_node(k)?)) })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .filter(|(_, pts)| !pts.is_empty())
            .collect();

        let total_in_mem: usize = nodes.values().map(|v| v.len()).sum();
        debug!(
            "In-memory build: loaded {} nodes, {} points ({} MB)",
            nodes.len(),
            total_in_mem,
            (total_in_mem * std::mem::size_of::<RawPoint>()) / 1_048_576,
        );

        for d in (0..actual_max_depth).rev() {
            config.report(crate::ProgressEvent::StageProgress {
                done: (actual_max_depth - d) as u64,
            });
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

    /// Bottom-up pass — disk-based slow path.
    ///
    /// Processes one level at a time with disk I/O.  Avoids redundant directory
    /// scans by maintaining a per-level key map updated after each level.
    /// Parents are batched so that the estimated peak memory of all concurrently
    /// loaded points stays within the configured memory budget.
    fn bottom_up_on_disk(
        &self,
        leaf_keys: Vec<VoxelKey>,
        actual_max_depth: u32,
        memory_budget: u64,
        config: &crate::PipelineConfig,
    ) -> Result<Vec<(VoxelKey, usize)>> {
        // Organise known keys by level — updated as parent nodes are created.
        // Use a set per level to prevent duplicate keys (a leaf and a parent
        // created by the bottom-up pass can share the same key when
        // normalize_leaves produces leaves at mixed depths).
        let mut keys_by_level: HashMap<i32, HashSet<VoxelKey>> = HashMap::new();
        for k in leaf_keys {
            keys_by_level.entry(k.level).or_default().insert(k);
        }

        // In-memory cost per point during a small-parent task. We hold:
        //   - the `all_pts: Vec<(usize, RawPoint)>` input  (~56 B/pt)
        //   - the `parent_pts` + per-child `remaining` outputs (~38 B/pt total,
        //     since each input point ends up in exactly one output vec)
        //   - rayon/HashSet/sort-scratch overhead (~20 B/pt safety margin)
        // Total ≈ 114 B/pt; round up to 128 for headroom. The previous value
        // (56) under-counted the outputs and could push batches ~40% over the
        // configured budget at peak.
        const MEM_PER_POINT: u64 = 128;

        for d in (0..actual_max_depth).rev() {
            config.report(crate::ProgressEvent::StageProgress {
                done: (actual_max_depth - d) as u64,
            });
            debug!("Building ancestor level {d}");
            let child_keys: Vec<VoxelKey> = match keys_by_level.get(&(d as i32 + 1)) {
                Some(v) => v.iter().copied().collect(),
                None => continue,
            };

            let mut parent_children: HashMap<VoxelKey, Vec<VoxelKey>> = HashMap::new();
            for ck in &child_keys {
                if let Some(parent) = ck.parent() {
                    parent_children.entry(parent).or_default().push(*ck);
                }
            }
            if parent_children.is_empty() {
                continue;
            }

            // Estimate each parent's memory cost from its children's file sizes,
            // then batch parents so the total stays within the memory budget.
            // Parents that exceed the budget on their own use a streaming path.
            let mut small_parents: Vec<(VoxelKey, Vec<VoxelKey>, u64)> = Vec::new();
            let mut large_parents: Vec<(VoxelKey, Vec<VoxelKey>)> = Vec::new();

            for (parent, children) in parent_children {
                let child_bytes: u64 = children
                    .iter()
                    .map(|ck| self.node_path(ck).metadata().map_or(0, |m| m.len()))
                    .sum();
                let est_points = child_bytes / RawPoint::BYTE_SIZE as u64;
                let est_mem = est_points * MEM_PER_POINT;
                if est_mem > memory_budget {
                    large_parents.push((parent, children));
                } else {
                    small_parents.push((parent, children, est_mem));
                }
            }

            debug!(
                "Level {d}: {} parents ({} small, {} large/streaming), budget={} MB",
                small_parents.len() + large_parents.len(),
                small_parents.len(),
                large_parents.len(),
                memory_budget / 1_048_576,
            );

            let mut new_parent_keys: Vec<VoxelKey> = Vec::new();

            // Process oversized parents sequentially with streaming grid_sample.
            // Only one child is in memory at a time.
            for (parent, children) in &large_parents {
                let child_bytes: u64 = children
                    .iter()
                    .map(|ck| self.node_path(ck).metadata().map_or(0, |m| m.len()))
                    .sum();
                debug!(
                    "  STREAMING parent {:?}: {} children, {} MB on disk",
                    parent,
                    children.len(),
                    child_bytes / 1_048_576,
                );
                self.grid_sample_streaming(parent, children)?;
                new_parent_keys.push(*parent);
            }

            // Batch small parents for parallel processing.
            small_parents.sort_by(|a, b| b.2.cmp(&a.2));

            let mut batch_start = 0;
            while batch_start < small_parents.len() {
                let mut batch_mem: u64 = 0;
                let mut batch_end = batch_start;
                while batch_end < small_parents.len() {
                    if batch_end > batch_start
                        && batch_mem + small_parents[batch_end].2 > memory_budget
                    {
                        break;
                    }
                    batch_mem += small_parents[batch_end].2;
                    batch_end += 1;
                }

                let batch = &small_parents[batch_start..batch_end];
                debug!(
                    "  Batch: {} parents, est {} MB",
                    batch.len(),
                    batch_mem / 1_048_576,
                );
                batch
                    .par_iter()
                    .map(|(parent, children, _)| -> Result<()> {
                        // Stream every child directly into a single tagged vec
                        // — no transient per-child Vec<RawPoint>, so the peak
                        // matches the budget estimate (MEM_PER_POINT).
                        let mut all_pts: Vec<(usize, RawPoint)> = Vec::new();
                        for (ci, ck) in children.iter().enumerate() {
                            self.read_node_into_indexed(ck, ci, &mut all_pts)?;
                        }
                        if all_pts.is_empty() {
                            return Ok(());
                        }
                        let (parent_pts, per_child) =
                            self.grid_sample(parent, all_pts, children.len());
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
                    new_parent_keys.push(*parent);
                }
                batch_start = batch_end;
            }

            keys_by_level
                .entry(d as i32)
                .or_default()
                .extend(new_parent_keys.into_iter());
        }

        // Enumerate result from the in-memory key map (no directory scan needed).
        let mut result = Vec::new();
        for level_keys in keys_by_level.values() {
            for key in level_keys {
                let file_len = self.node_path(key).metadata().map_or(0, |m| m.len());
                let count = file_len as usize / RawPoint::BYTE_SIZE;
                if count > 0 {
                    result.push((*key, count));
                }
            }
        }

        // Verify point conservation — flag if total differs from input.
        let total: u64 = result.iter().map(|(_, c)| *c as u64).sum();
        if total != self.total_points {
            info!(
                "Point conservation check: octree has {} points vs {} input (diff {})",
                total,
                self.total_points,
                total as i64 - self.total_points as i64
            );
        }

        Ok(result)
    }

    /// Streaming grid-sample for a single oversized parent.
    ///
    /// Processes one child at a time so that only one child's points are ever in
    /// memory. The grid occupancy set (max 128³ entries ≈ 48 MB) and the parent
    /// accumulator (max 128³ points ≈ 80 MB) stay resident; everything else is
    /// streamed from/to disk.
    ///
    /// Tradeoff: points are not Morton-sorted across children, so spatial
    /// coherence of the parent is slightly worse than the in-memory path.
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
                let file_len = self.node_path(ck).metadata().map_or(0, |m| m.len());
                if file_len == 0 && !parent_pts.is_empty() {
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
}
