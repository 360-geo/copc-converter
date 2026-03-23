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
/// 4. Normalize leaves: any leaf with > MAX_NODE_POINTS is split into children on disk.
/// 5. Build the tree bottom-up in parallel: each parent node gets a thinned sample
///    of its children's points written back to disk.
/// 6. Produce the list of (VoxelKey, point_count) for the writer, which reads from disk.
///
/// Memory usage is bounded by the configurable memory budget.
use crate::PipelineConfig;
use crate::copc_types::VoxelKey;
use anyhow::{Context, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use log::info;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};

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

/// Maximum points to keep per non-leaf node (thinned sample for overview).
const MAX_NODE_POINTS: usize = 100_000;

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

pub struct OctreeBuilder {
    pub bounds: Bounds,
    pub total_points: u64,
    pub cx: f64,
    pub cy: f64,
    pub cz: f64,
    pub halfsize: f64,
    pub depth: u32,
    /// Scale / offset from the first input file – used for all raw-integer conversions.
    pub scale_x: f64,
    pub scale_y: f64,
    pub scale_z: f64,
    pub offset_x: f64,
    pub offset_y: f64,
    pub offset_z: f64,
    /// Temp directory where node files are written.
    pub tmp_dir: PathBuf,
    /// WKT CRS payload from the first input file (if present).
    pub wkt_crs: Option<Vec<u8>>,
}

impl OctreeBuilder {
    /// Pass 1: scan all files in parallel to get bounds and total point count.
    pub fn scan(input_files: &[PathBuf], config: &PipelineConfig) -> Result<Self> {
        type Transforms = (f64, f64, f64, f64, f64, f64);
        let results: Vec<(Bounds, u64, Transforms, Option<Vec<u8>>)> = input_files
            .par_iter()
            .map(|path| -> Result<_> {
                info!("Scanning {:?}", path);
                let reader = las::Reader::from_path(path)
                    .with_context(|| format!("Cannot open {:?}", path))?;
                let hdr = reader.header();
                let b = hdr.bounds();
                let mut bounds = Bounds::empty();
                bounds.expand_with(b.min.x, b.min.y, b.min.z);
                bounds.expand_with(b.max.x, b.max.y, b.max.z);
                let point_count = hdr.number_of_points();
                let t = hdr.transforms();
                let transforms = (
                    t.x.scale, t.y.scale, t.z.scale, t.x.offset, t.y.offset, t.z.offset,
                );
                let wkt = hdr
                    .all_vlrs()
                    .find(|v| v.is_wkt_crs())
                    .map(|v| v.data.clone());
                Ok((bounds, point_count, transforms, wkt))
            })
            .collect::<Result<Vec<_>>>()?;

        let mut bounds = Bounds::empty();
        let mut total_points = 0u64;
        for (b, count, _, _) in &results {
            bounds.merge(b);
            total_points += count;
        }

        let (scale_x, scale_y, scale_z, offset_x, offset_y, offset_z) = results
            .first()
            .map(|(_, _, t, _)| *t)
            .unwrap_or((0.001, 0.001, 0.001, 0.0, 0.0, 0.0));

        // Extract WKT CRS from the first file and verify all files match.
        let wkt_crs = results[0].3.clone();
        for (i, (_, _, _, wkt)) in results.iter().enumerate().skip(1) {
            if *wkt != wkt_crs {
                anyhow::bail!(
                    "CRS mismatch: file {:?} has a different WKT CRS than {:?}",
                    input_files[i],
                    input_files[0],
                );
            }
        }

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
        info!("Octree depth = {depth}, total points = {total_points}");

        let sys_tmp = std::env::temp_dir();
        let base_tmp = config.temp_dir.as_deref().unwrap_or(&sys_tmp);
        let tmp_dir = base_tmp.join(format!("copc_{}", std::process::id()));
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
            wkt_crs,
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
            nir: 0,
        }
    }

    /// Parallel key assignment + coordinate conversion for a batch of points.
    ///
    /// Key assignment uses the *reconstructed* world coordinates (integer → world)
    /// rather than the original floating-point coordinates.  This guarantees that
    /// what the validator computes from the stored integers always falls inside the
    /// assigned voxel, even when input files use different scales/offsets.
    fn classify_points_parallel(&self, points: &[las::Point]) -> Vec<(VoxelKey, RawPoint)> {
        points
            .par_iter()
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

    /// Merge classified points into per-key buffers and flush periodically.
    fn merge_into_buffers(
        classified: Vec<(VoxelKey, RawPoint)>,
        buffers: &mut HashMap<VoxelKey, Vec<RawPoint>>,
        writers: &mut HashMap<VoxelKey, BufWriter<File>>,
        tmp_dir: &Path,
        point_idx: &mut u64,
        flush_every: usize,
    ) -> Result<()> {
        for (key, raw) in classified {
            buffers.entry(key).or_default().push(raw);
            *point_idx += 1;
            if (*point_idx).is_multiple_of(flush_every as u64) {
                Self::flush_buffers(buffers, writers, tmp_dir)?;
            }
        }
        Ok(())
    }

    /// Pass 2: assign all points to leaf temp files.
    ///
    /// Uses `read_all_points_into` (fast parallel decompression) when the file
    /// fits within half the memory budget; otherwise falls back to batched reads.
    /// Key assignment and coordinate conversion are always parallelized via rayon.
    pub fn distribute(&self, input_files: &[PathBuf], config: &PipelineConfig) -> Result<()> {
        let flush_every =
            ((config.memory_budget / 4) as usize / RawPoint::BYTE_SIZE).clamp(10_000, 500_000);
        info!("Flush interval: {} points", flush_every);

        let mut buffers: HashMap<VoxelKey, Vec<RawPoint>> = HashMap::new();
        let mut writers: HashMap<VoxelKey, BufWriter<File>> = HashMap::new();
        let mut point_idx = 0u64;

        let half_budget = config.memory_budget / 2;

        for path in input_files {
            info!("Distributing {:?}", path);
            let mut reader =
                las::Reader::from_path(path).with_context(|| format!("Cannot open {:?}", path))?;

            let file_point_count = reader.header().number_of_points();
            // Estimated memory per las::Point (~120 bytes)
            let estimated_mem = file_point_count * 120;

            if estimated_mem <= half_budget {
                // Fast path: load entire file with parallel decompression
                let mut points: Vec<las::Point> = Vec::new();
                reader.read_all_points_into(&mut points)?;
                let classified = self.classify_points_parallel(&points);
                drop(points);
                Self::merge_into_buffers(
                    classified,
                    &mut buffers,
                    &mut writers,
                    &self.tmp_dir,
                    &mut point_idx,
                    flush_every,
                )?;
            } else {
                // Batched path: read in chunks to stay within budget
                let batch_size = (half_budget / 120).max(10_000) as usize;
                info!(
                    "File too large (~{} MB), using batched reads of {} points",
                    estimated_mem / (1024 * 1024),
                    batch_size
                );
                let mut points: Vec<las::Point> = Vec::new();
                loop {
                    points.clear();
                    let n = reader.read_points_into(batch_size as u64, &mut points)?;
                    if n == 0 {
                        break;
                    }
                    let classified = self.classify_points_parallel(&points);
                    Self::merge_into_buffers(
                        classified,
                        &mut buffers,
                        &mut writers,
                        &self.tmp_dir,
                        &mut point_idx,
                        flush_every,
                    )?;
                }
            }
        }

        Self::flush_buffers(&mut buffers, &mut writers, &self.tmp_dir)?;
        // Explicitly flush all BufWriters so no data is lost on drop.
        for (_, w) in writers.iter_mut() {
            std::io::Write::flush(w).context("flush distribute writer")?;
        }
        Ok(())
    }

    fn flush_buffers(
        buffers: &mut HashMap<VoxelKey, Vec<RawPoint>>,
        writers: &mut HashMap<VoxelKey, BufWriter<File>>,
        tmp_dir: &Path,
    ) -> Result<()> {
        for (key, pts) in buffers.iter_mut() {
            if pts.is_empty() {
                continue;
            }
            let w = writers.entry(*key).or_insert_with(|| {
                let path = tmp_dir.join(format!("{}_{}_{}_{}", key.level, key.x, key.y, key.z));
                let f = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path)
                    .expect("Cannot open leaf file");
                BufWriter::new(f)
            });
            RawPoint::write_bulk(pts, w)?;
            pts.clear();
        }
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

    /// Phase 0 of build_node_map: split any leaf file that exceeds MAX_NODE_POINTS.
    ///
    /// All nodes being split in one round occupy disjoint voxels, so their child
    /// files never conflict — the entire round is processed in parallel via rayon.
    /// Rounds repeat until no oversized nodes remain.
    fn normalize_leaves(&self) -> Result<()> {
        // Allow at most 4 extra levels beyond the initial depth.  This caps the
        // total LOD count at depth+5 (levels 0…depth+4) regardless of local
        // density spikes.
        let max_level = self.depth as i32 + 4;

        let oversized = |k: &VoxelKey| -> bool {
            k.level < max_level
                && self
                    .node_path(k)
                    .metadata()
                    .is_ok_and(|m| m.len() as usize / RawPoint::BYTE_SIZE > MAX_NODE_POINTS)
        };

        let mut to_split: Vec<VoxelKey> = self
            .all_node_keys()?
            .into_iter()
            .filter(|k| oversized(k))
            .collect();

        while !to_split.is_empty() {
            // All nodes in `to_split` are in disjoint voxels → safe to split in parallel.
            let new_children: Vec<VoxelKey> = to_split
                .into_par_iter()
                .map(|key| -> Result<Vec<VoxelKey>> {
                    let pts = self.read_node(&key)?;
                    std::fs::remove_file(self.node_path(&key))?;

                    let child_level = key.level + 1;
                    let mut children: HashMap<VoxelKey, Vec<RawPoint>> = HashMap::new();
                    for p in pts {
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
                        children.entry(ck).or_default().push(p);
                    }
                    let mut child_keys = Vec::new();
                    for (ck, cpts) in children {
                        self.write_node_to_temp(&ck, &cpts)?;
                        child_keys.push(ck);
                    }
                    Ok(child_keys)
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect();

            to_split = new_children.into_iter().filter(|k| oversized(k)).collect();
        }
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
        self.normalize_leaves()?;

        // Phase 1
        let leaf_keys = self.all_node_keys()?;
        let actual_max_depth = leaf_keys.iter().map(|k| k.level as u32).max().unwrap_or(0);
        info!(
            "Leaf nodes after normalization: {}, max depth: {actual_max_depth}",
            leaf_keys.len()
        );

        // Estimate memory required to hold all leaf data.
        let total_bytes: u64 = leaf_keys
            .iter()
            .map(|k| self.node_path(k).metadata().map_or(0, |m| m.len()))
            .sum();

        // Phase 2
        let result = if total_bytes <= config.memory_budget {
            info!("Building octree in-memory ({} MB)", total_bytes / 1_048_576);
            self.bottom_up_in_memory(&leaf_keys, actual_max_depth)?
        } else {
            info!(
                "Building octree out-of-core ({} MB > budget {} MB)",
                total_bytes / 1_048_576,
                config.memory_budget / 1_048_576,
            );
            self.bottom_up_on_disk(leaf_keys, actual_max_depth)?
        };

        let total_pts: usize = result.iter().map(|(_, c)| *c).sum();
        info!(
            "Total octree nodes: {}, total points: {} (original: {})",
            result.len(),
            total_pts,
            self.total_points
        );
        if total_pts as u64 != self.total_points {
            info!(
                "Note: COPC contains {} points vs {} from input headers (diff {}). \
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
    ) -> Result<Vec<(VoxelKey, usize)>> {
        // Load all leaf nodes in parallel.
        let mut nodes: HashMap<VoxelKey, Vec<RawPoint>> = leaf_keys
            .par_iter()
            .map(|k| -> Result<(VoxelKey, Vec<RawPoint>)> { Ok((*k, self.read_node(k)?)) })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .filter(|(_, pts)| !pts.is_empty())
            .collect();

        for d in (0..actual_max_depth).rev() {
            info!("Building ancestor level {d}");

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
                    let (parent_pts, remaining) =
                        self.grid_sample(&parent, all_pts, n, MAX_NODE_POINTS);
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
    fn bottom_up_on_disk(
        &self,
        leaf_keys: Vec<VoxelKey>,
        actual_max_depth: u32,
    ) -> Result<Vec<(VoxelKey, usize)>> {
        // Organise known keys by level — updated as parent nodes are created.
        let mut keys_by_level: HashMap<i32, Vec<VoxelKey>> = HashMap::new();
        for k in leaf_keys {
            keys_by_level.entry(k.level).or_default().push(k);
        }

        for d in (0..actual_max_depth).rev() {
            info!("Building ancestor level {d}");
            let child_keys = match keys_by_level.get(&(d as i32 + 1)) {
                Some(v) => v.clone(),
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

            let parents: Vec<(VoxelKey, Vec<VoxelKey>)> = parent_children.into_iter().collect();
            let new_parent_keys: Vec<VoxelKey> = parents.iter().map(|(p, _)| *p).collect();

            parents
                .par_iter()
                .map(|(parent, children)| -> Result<()> {
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
                        self.grid_sample(parent, all_pts, children.len(), MAX_NODE_POINTS);
                    for (ci, ck) in children.iter().enumerate() {
                        self.write_node_to_temp(ck, &per_child[ci])?;
                    }
                    if !parent_pts.is_empty() {
                        self.write_node_to_temp(parent, &parent_pts)?;
                    }
                    Ok(())
                })
                .collect::<Result<Vec<_>>>()?;

            keys_by_level
                .entry(d as i32)
                .or_default()
                .extend(new_parent_keys);
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
        Ok(result)
    }

    /// Grid-based spatial sampling for one parent node.
    ///
    /// Divides the parent voxel into a uniform grid of ≈ R³ cells (R = ∛max_count).
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
        max_count: usize,
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

        // Grid resolution: R³ ≈ max_count cells; use floor so we never exceed max_count.
        let r = (max_count as f64).cbrt() as i64;
        let cell = (int_size / r).max(1);

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
        let mut parent_pts: Vec<(usize, RawPoint)> = Vec::with_capacity(max_count);
        let mut remaining: Vec<Vec<RawPoint>> = vec![Vec::new(); n_children];

        for (ci, p) in pts {
            if parent_pts.len() < max_count && occupied.insert(grid_key(&p)) {
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

    pub fn cleanup(&self) {
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
        RawPoint::write_bulk(&[p.clone()], &mut bulk_buf).unwrap();

        assert_eq!(
            single_buf, bulk_buf,
            "bulk write must produce identical bytes to single write"
        );
    }
}
