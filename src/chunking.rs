//! Hierarchical counting-sort chunk planner (Schütz et al. 2020).
//!
//! Determines what chunks the build path will produce for a given dataset.
//! Used by the main pipeline's distribute stage and exposed via the
//! `analyze` CLI subcommand for inspection without writing output.
//!
//! # Algorithm
//!
//! 1. **Count**: stream every input file in parallel, atomically incrementing
//!    a flat 3D counting grid at the chosen resolution (128³, 256³, or 512³).
//!    Each cell of the grid corresponds to an octree voxel at the same depth
//!    (`d_grid = log2(grid_size)`).
//!
//! 2. **Merge sparse cells**: walk the implicit pyramid from the finest level
//!    to the root. For each 2×2×2 group of children, if their combined point
//!    count fits within the chunk target size and none of the children are
//!    already "blocked", collapse the group into the parent cell. Otherwise
//!    each non-empty child becomes a chunk root and the parent is blocked.
//!
//! 3. **Result**: a [`ChunkPlan`] listing every chunk root as
//!    `(level, gx, gy, gz, point_count)`. Most chunks are at fine levels in
//!    dense regions and at coarse levels in sparse regions.
//!
//! # Memory bound
//!
//! Peak memory during the merge phase is roughly the size of two adjacent
//! pyramid levels: ~604 MB for a 512³ grid, ~75 MB for 256³, ~10 MB for 128³.
//! This is independent of the input dataset size.

use crate::PipelineConfig;
use crate::octree::{OctreeBuilder, ScanResult, point_to_key};
use crate::validate::ValidatedInputs;
use anyhow::{Context, Result};
use rayon::prelude::*;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use tracing::{debug, info};

/// Sentinel population value meaning "this cell is blocked — it must not be
/// rolled up further because at least one of its children is already a chunk
/// root or itself blocked."
const SENTINEL: i64 = -1;

/// Default chunk target derivation: budget × this fraction / (workers × per-point overhead).
const SAFETY_FACTOR: f64 = 0.6;

/// Estimated bytes per point held in memory while a chunk is being built
/// (`Vec<RawPoint>` + `Vec<(usize, RawPoint)>` sort buffer + grid_sample
/// output, plus a small constant for `HashSet` working state).
const PER_POINT_OVERHEAD_BYTES: u64 = 170;

/// Lower clamp on chunk target size (points). Below this, fixed per-chunk
/// overhead (file open, octree node bookkeeping) dominates point processing.
const MIN_CHUNK_POINTS: u64 = 1_000_000;

/// Per-chunk peak working set during the in-memory build phase, matching
/// `PER_CHUNK_BYTES_PER_POINT` in octree.rs. Used here to derive the upper
/// clamp on chunk target size so a single chunk can never exceed the
/// configured memory budget.
const PER_CHUNK_PEAK_BYTES_PER_POINT: u64 = 600;

/// Fraction of the memory budget a single chunk is allowed to occupy.
/// Set below 0.5 so rayon can run at least two chunks in parallel while
/// leaving slack for allocator overhead and concurrent stages.
const SINGLE_CHUNK_BUDGET_FRACTION: f64 = 0.4;

/// Aim for at least this many chunks per worker so small datasets still
/// have parallelism headroom.
const PARALLELISM_TARGET_PER_WORKER: u64 = 4;

/// Dynamic upper clamp on chunk target size, derived from the memory
/// budget so a single chunk's in-memory build can never exceed the
/// configured budget. Clamped below by `MIN_CHUNK_POINTS` to avoid
/// degenerate chunking on tiny budgets.
pub(crate) fn max_chunk_points(memory_budget: u64) -> u64 {
    let raw = ((memory_budget as f64 * SINGLE_CHUNK_BUDGET_FRACTION)
        / PER_CHUNK_PEAK_BYTES_PER_POINT as f64) as u64;
    raw.max(MIN_CHUNK_POINTS)
}

/// Hard cap on how many extra octree levels a single over-target finest cell
/// may be sub-split into. `8^SPLIT_LEVEL_CAP` sub-chunks is the most one cell
/// can expand to; at the default cap a cell can fan out to 8^6 ≈ 262k
/// sub-chunks, enough to bring a multi-billion-point cell under any sane
/// `chunk_target` while bounding the worst-case sub-chunk index count.
const SPLIT_LEVEL_CAP: u32 = 6;

/// Extra octree levels needed so a finest-grid cell holding `pop` points
/// sub-divides into sub-chunks of ~`chunk_target` points each.
///
/// Returns `ceil(log8(pop / chunk_target))`, clamped to `SPLIT_LEVEL_CAP`.
/// Returns 0 when the cell already fits `chunk_target` (no split). The
/// distribution within the cell is rarely uniform, so the per-sub-chunk
/// estimate (`pop / 8^k`) is only a target; the build phase's adaptive leaf
/// subdivision absorbs any residual imbalance.
fn split_extra_levels(pop: u64, chunk_target: u64) -> u32 {
    if pop <= chunk_target {
        return 0;
    }
    let mut k = 0u32;
    while (pop as f64) / (8u64.pow(k) as f64) > chunk_target as f64 {
        k += 1;
        if k >= SPLIT_LEVEL_CAP {
            break;
        }
    }
    k
}

// ---------------------------------------------------------------------------
// Chunk plan output
// ---------------------------------------------------------------------------

/// A single chunk in the plan: an octree voxel at some level, plus its
/// estimated point count derived from the counting grid.
///
/// Most chunks sit at a grid level (`level <= grid_depth`) and own exactly
/// one fine grid cell (or a merged cube of them). When a single finest-level
/// cell holds more than `chunk_target` points, the planner sub-splits it into
/// deeper octree voxels (`level > grid_depth`) so no chunk ever exceeds the
/// memory budget — see [`merge_sparse_cells`]. Those deeper sub-chunks share
/// one fine grid cell, so distribute routes points into them with a deeper
/// `point_to_key` descent rather than the plain cell→chunk LUT.
#[derive(Debug, Clone, Copy)]
pub struct PlannedChunk {
    /// Octree level of this chunk's root cell.
    pub level: u32,
    /// Voxel x coordinate at the given level.
    pub gx: i32,
    /// Voxel y coordinate at the given level.
    pub gy: i32,
    /// Voxel z coordinate at the given level.
    pub gz: i32,
    /// Estimated point count from the counting pass.
    pub point_count: u64,
}

/// Output of [`analyze_chunking`]: the full chunk plan plus useful metadata
/// about how it was computed.
#[derive(Debug)]
pub struct ChunkPlan {
    /// Grid resolution used for the count pass (128, 256, or 512).
    pub grid_size: u32,
    /// Octree depth corresponding to the grid resolution.
    pub grid_depth: u32,
    /// Chunk target size in points that drove the merge step.
    pub chunk_target: u64,
    /// Total points across all chunks (should equal input total).
    pub total_points: u64,
    /// Every chunk root produced by the merge step.
    pub chunks: Vec<PlannedChunk>,
    /// Set when the LAS header bounds disagree with the actual point data
    /// observed during the counting pass (beyond one scale unit per axis).
    /// The CLI surfaces this as a user-visible warning.
    pub header_mismatch: Option<HeaderBoundsMismatch>,
}

/// Per-axis differences between the LAS header bounds and the actual
/// round-tripped point coordinates observed during counting.
///
/// Each tuple is `(header_value, actual_value)`. Only axes whose absolute
/// difference exceeds 1.5 scale units are populated; the rest are `None`.
#[derive(Debug, Clone, Copy)]
pub struct HeaderBoundsMismatch {
    pub min_x: Option<(f64, f64)>,
    pub max_x: Option<(f64, f64)>,
    pub min_y: Option<(f64, f64)>,
    pub max_y: Option<(f64, f64)>,
    pub min_z: Option<(f64, f64)>,
    pub max_z: Option<(f64, f64)>,
}

impl HeaderBoundsMismatch {
    fn any(&self) -> bool {
        self.min_x.is_some()
            || self.max_x.is_some()
            || self.min_y.is_some()
            || self.max_y.is_some()
            || self.min_z.is_some()
            || self.max_z.is_some()
    }
}

impl ChunkPlan {
    /// Number of chunks in the plan.
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    /// True iff the plan is empty (no chunks). Should only happen on empty
    /// input.
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute the dynamic chunk target size from the memory budget and worker
/// count, applying clamping and a parallelism floor.
///
/// Returns the target size in points. Larger budgets and fewer workers
/// produce larger chunks (less coordination, more cache pressure per chunk);
/// smaller budgets and more workers produce smaller chunks.
pub fn compute_chunk_target(memory_budget: u64, num_workers: usize, total_points: u64) -> u64 {
    let workers = num_workers.max(1) as u64;
    let raw = ((memory_budget as f64 * SAFETY_FACTOR) / (workers * PER_POINT_OVERHEAD_BYTES) as f64)
        as u64;

    // Parallelism floor: ensure we have at least N chunks per worker so small
    // datasets don't end up with one-chunk-per-pod.
    let parallelism_target = workers * PARALLELISM_TARGET_PER_WORKER;
    let max_for_parallelism = total_points
        .checked_div(parallelism_target)
        .map(|v| v.max(MIN_CHUNK_POINTS))
        .unwrap_or(u64::MAX);

    let max_chunk = max_chunk_points(memory_budget);
    raw.min(max_for_parallelism)
        .clamp(MIN_CHUNK_POINTS, max_chunk)
}

/// Pick the counting grid resolution based on total point count and the
/// memory budget.
///
/// Point-count tiers follow Schütz et al. 2020 (paragraph 4.1.2): coarser
/// grids for small datasets save memory, finer grids give better adaptivity
/// in dense regions. The grid and LUT each occupy `grid_size³ × 4` bytes,
/// so we also clamp against a fraction of the memory budget to avoid
/// starving the per-chunk build.
pub fn select_grid_size(total_points: u64, memory_budget: u64) -> u32 {
    let preferred = if total_points < 100_000_000 {
        128
    } else if total_points < 500_000_000 {
        256
    } else {
        512
    };

    // Cap the grid + LUT combined at ~10% of the budget (5% each).
    const GRID_BUDGET_FRACTION_BP: u64 = 500; // 5% in basis points
    let budget_bytes = memory_budget.saturating_mul(GRID_BUDGET_FRACTION_BP) / 10_000;
    let max_cells = budget_bytes / 4;
    // Walk the tier ladder downward from `preferred` until one fits.
    let mut allowed = 128u32;
    for candidate in [512u32, 256, 128] {
        let cells = (candidate as u64).pow(3);
        if cells <= max_cells && candidate <= preferred {
            allowed = candidate;
            break;
        }
    }
    allowed
}

/// Top-level entry point for the analyze tool: builds an `OctreeBuilder`
/// from scan results, then delegates to [`compute_chunk_plan`].
///
/// Reuses scan + validate via the existing pipeline machinery so the analyze
/// tool sees exactly the same bounds, scale, and CRS as a real conversion
/// would. Does **not** write any temp files for the chunks themselves; only
/// the empty `OctreeBuilder` temp dir is created and is cleaned up on drop.
pub fn analyze_chunking(
    input_files: &[PathBuf],
    scan_results: &[ScanResult],
    validated: &ValidatedInputs,
    config: &PipelineConfig,
    chunk_target_override: Option<u64>,
) -> Result<ChunkPlan> {
    let builder = OctreeBuilder::from_scan(scan_results, validated, config)
        .context("constructing OctreeBuilder for chunk analysis")?;
    // The standalone analyzer treats counting as its own user-visible
    // stage. (When called from inside `distribute`, the caller owns the
    // stage events instead.)
    config.report(crate::ProgressEvent::StageStart {
        name: "Counting",
        total: builder.total_points,
    });
    let plan = compute_chunk_plan(&builder, input_files, config, chunk_target_override)?;
    config.report(crate::ProgressEvent::StageDone);
    Ok(plan)
}

/// Run the counting pass and merge step against an existing `OctreeBuilder`.
///
/// Used by both the analyze CLI tool and the main distribute stage, so
/// they share the exact same chunking logic and produce the same plan
/// for the same inputs.
///
/// **Does not emit `StageStart`/`StageDone` events** — the caller owns the
/// progress reporting boundary so this can be embedded inside another stage
/// without clobbering its progress bar.
pub(crate) fn compute_chunk_plan(
    builder: &OctreeBuilder,
    input_files: &[PathBuf],
    config: &PipelineConfig,
    chunk_target_override: Option<u64>,
) -> Result<ChunkPlan> {
    let total_points = builder.total_points;
    let grid_size = select_grid_size(total_points, config.memory_budget);
    let grid_depth = grid_size.trailing_zeros();
    let num_workers = distribute_worker_count(input_files.len());
    let chunk_target = chunk_target_override
        .unwrap_or_else(|| compute_chunk_target(config.memory_budget, num_workers, total_points));

    info!(
        "Chunk plan: total_points={}, grid={}³ (depth {}), chunk_target={}, workers={}",
        total_points, grid_size, grid_depth, chunk_target, num_workers
    );

    // ----- Count pass -----
    let (grid, actual) = count_points(builder, input_files, grid_size, grid_depth, config)?;

    let cells_touched = grid
        .iter()
        .filter(|c| c.load(Ordering::Relaxed) > 0)
        .count();
    debug!(
        "Count pass: {} of {} cells touched ({:.2}%)",
        cells_touched,
        grid.len(),
        cells_touched as f64 / grid.len() as f64 * 100.0
    );

    let header_mismatch = detect_header_mismatch(builder, &actual);

    // ----- Merge sparse cells -----
    let chunks = merge_sparse_cells(&grid, grid_size, grid_depth, chunk_target);

    Ok(ChunkPlan {
        grid_size,
        grid_depth,
        chunk_target,
        total_points,
        chunks,
        header_mismatch,
    })
}

// ---------------------------------------------------------------------------
// Counting pass
// ---------------------------------------------------------------------------

/// Compare the LAS header bounds to the actual point data observed during
/// counting, returning a [`HeaderBoundsMismatch`] if any axis exceeds 1.5
/// scale units of tolerance. The CLI surfaces any mismatch as a warning so
/// users know their input headers are inaccurate.
fn detect_header_mismatch(
    builder: &OctreeBuilder,
    actual: &ActualBounds,
) -> Option<HeaderBoundsMismatch> {
    // Tolerance: 1.5 scale units per axis. Stored point coordinates are
    // `int32 × scale + offset`, so one scale unit is the finest difference
    // that can be real — but float round-tripping of header bounds
    // routinely overshoots one scale unit by a few ULPs. The extra half
    // unit absorbs that noise and the common case of header bounds rounded
    // to one more decimal than the point precision, while still flagging
    // any genuine ≥2-unit disagreement.
    let flag = |hdr: f64, act: f64, scale: f64| -> Option<(f64, f64)> {
        ((hdr - act).abs() > 1.5 * scale).then_some((hdr, act))
    };
    let hdr = &builder.bounds;
    let result = HeaderBoundsMismatch {
        min_x: flag(hdr.min_x, actual.min_x, builder.scale_x),
        max_x: flag(hdr.max_x, actual.max_x, builder.scale_x),
        min_y: flag(hdr.min_y, actual.min_y, builder.scale_y),
        max_y: flag(hdr.max_y, actual.max_y, builder.scale_y),
        min_z: flag(hdr.min_z, actual.min_z, builder.scale_z),
        max_z: flag(hdr.max_z, actual.max_z, builder.scale_z),
    };
    result.any().then_some(result)
}

/// Per-axis actual bounds observed during the counting pass. Used to warn
/// users when the LAS header bounds disagree with the real point data.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ActualBounds {
    pub min_x: f64,
    pub min_y: f64,
    pub min_z: f64,
    pub max_x: f64,
    pub max_y: f64,
    pub max_z: f64,
}

impl ActualBounds {
    fn empty() -> Self {
        Self {
            min_x: f64::INFINITY,
            min_y: f64::INFINITY,
            min_z: f64::INFINITY,
            max_x: f64::NEG_INFINITY,
            max_y: f64::NEG_INFINITY,
            max_z: f64::NEG_INFINITY,
        }
    }

    fn merge(&mut self, other: &ActualBounds) {
        self.min_x = self.min_x.min(other.min_x);
        self.min_y = self.min_y.min(other.min_y);
        self.min_z = self.min_z.min(other.min_z);
        self.max_x = self.max_x.max(other.max_x);
        self.max_y = self.max_y.max(other.max_y);
        self.max_z = self.max_z.max(other.max_z);
    }
}

/// Stream every input file in parallel, classifying each point into the
/// counting grid via `point_to_key` at the grid depth, atomically
/// incrementing the corresponding cell counter, and tracking the actual
/// per-axis bounds of the round-tripped coordinates.
///
/// Returns the flat `Box<[AtomicU32]>` of length `grid_size³` (cell index
/// `gx + gy*G + gz*G²`) plus the observed [`ActualBounds`].
fn count_points(
    builder: &OctreeBuilder,
    input_files: &[PathBuf],
    grid_size: u32,
    grid_depth: u32,
    config: &PipelineConfig,
) -> Result<(Box<[AtomicU32]>, ActualBounds)> {
    let g = grid_size as usize;
    let n_cells = g * g * g;
    debug!(
        "Allocating counting grid: {}³ = {} cells, {} MB",
        grid_size,
        n_cells,
        n_cells * 4 / 1_048_576
    );

    // Allocate a zeroed Vec<u32> via the standard zero-init path (calloc on
    // most allocators) and transmute to AtomicU32. This is sound because
    // AtomicU32 has the same memory layout as u32 (#[repr(C, align(4))]).
    let grid: Box<[AtomicU32]> = {
        let zeros: Vec<u32> = vec![0u32; n_cells];
        let boxed: Box<[u32]> = zeros.into_boxed_slice();
        let ptr = Box::into_raw(boxed) as *mut [AtomicU32];
        // SAFETY: AtomicU32 has the same layout as u32 and the source slice
        // was just allocated zeroed. Ownership transfers cleanly via raw ptr.
        unsafe { Box::from_raw(ptr) }
    };

    let num_workers = distribute_worker_count(input_files.len());
    let files_per_worker = input_files.len().div_ceil(num_workers);
    let progress = AtomicU64::new(0);

    // Transient `las::Point` buffer cost (~120 B/pt). Match distribute:
    // take 1/8 of the per-worker budget to leave headroom for the LAS
    // decoder's own buffers.
    const BYTES_PER_LAS_POINT: u64 = 120;
    let per_worker_budget = config.memory_budget / num_workers as u64;
    let read_chunk_size =
        ((per_worker_budget / 8) / BYTES_PER_LAS_POINT).clamp(50_000, 2_000_000) as usize;

    debug!(
        "Counting with {} workers, {} files per worker, read_chunk_size={}",
        num_workers, files_per_worker, read_chunk_size
    );

    let per_worker_bounds: Vec<ActualBounds> = (0..num_workers)
        .into_par_iter()
        .map(|worker_id| -> Result<ActualBounds> {
            let mut local_bounds = ActualBounds::empty();
            let start = worker_id * files_per_worker;
            let end = (start + files_per_worker).min(input_files.len());
            if start >= end {
                return Ok(local_bounds);
            }

            // Reuse a single point buffer across files in this worker to
            // amortize allocation cost.
            let mut points: Vec<las::Point> = Vec::with_capacity(read_chunk_size);

            for path in &input_files[start..end] {
                let mut reader = las::Reader::from_path(path)
                    .with_context(|| format!("Cannot open {:?}", path))?;
                loop {
                    points.clear();
                    let n = reader.read_points_into(read_chunk_size as u64, &mut points)?;
                    if n == 0 {
                        break;
                    }
                    for p in &points {
                        // Classify using the same round-tripped coordinates
                        // the distribute phase will use, so the grid cell
                        // counted here is exactly the cell the LUT will look
                        // up for the same point at distribute time. Raw `p.x`
                        // and `raw.x*scale+offset` can differ by up to half a
                        // scale step due to LAS scale/offset rounding, which
                        // would put boundary points in different cells and
                        // cause point loss during build.
                        let ix = ((p.x - builder.offset_x) / builder.scale_x).round() as i32;
                        let iy = ((p.y - builder.offset_y) / builder.scale_y).round() as i32;
                        let iz = ((p.z - builder.offset_z) / builder.scale_z).round() as i32;
                        let wx = ix as f64 * builder.scale_x + builder.offset_x;
                        let wy = iy as f64 * builder.scale_y + builder.offset_y;
                        let wz = iz as f64 * builder.scale_z + builder.offset_z;
                        // Track actual bounds of round-tripped coordinates so
                        // we can warn when the header bounds disagree with
                        // the real data (e.g. after a tool rewrote headers
                        // with coarser precision than the point data).
                        if wx < local_bounds.min_x {
                            local_bounds.min_x = wx;
                        }
                        if wy < local_bounds.min_y {
                            local_bounds.min_y = wy;
                        }
                        if wz < local_bounds.min_z {
                            local_bounds.min_z = wz;
                        }
                        if wx > local_bounds.max_x {
                            local_bounds.max_x = wx;
                        }
                        if wy > local_bounds.max_y {
                            local_bounds.max_y = wy;
                        }
                        if wz > local_bounds.max_z {
                            local_bounds.max_z = wz;
                        }
                        let key = point_to_key(
                            wx,
                            wy,
                            wz,
                            builder.cx,
                            builder.cy,
                            builder.cz,
                            builder.halfsize,
                            grid_depth,
                        );
                        // VoxelKey coordinates are non-negative at the given
                        // depth (the octree root encloses all points), but be
                        // defensive against the rare case where a point lands
                        // exactly on the upper boundary and rounds out.
                        let gx = (key.x as usize).min(g - 1);
                        let gy = (key.y as usize).min(g - 1);
                        let gz = (key.z as usize).min(g - 1);
                        let idx = gx + gy * g + gz * g * g;
                        grid[idx].fetch_add(1, Ordering::Relaxed);
                    }
                    let done = progress.fetch_add(n, Ordering::Relaxed) + n;
                    // Reuse the existing pipeline progress channel so the
                    // analyze tool's CLI can show a familiar progress bar.
                    // The stage was started by the caller.
                    config.report(crate::ProgressEvent::StageProgress { done });
                }
            }
            Ok(local_bounds)
        })
        .collect::<Result<Vec<_>>>()?;

    let mut actual = ActualBounds::empty();
    for b in &per_worker_bounds {
        actual.merge(b);
    }

    Ok((grid, actual))
}

// ---------------------------------------------------------------------------
// Merge sparse cells
// ---------------------------------------------------------------------------

/// Emit chunk root(s) for one finest- or coarser-level grid cell.
///
/// For a cell at or below `chunk_target`, pushes a single [`PlannedChunk`] at
/// `level`. For an over-target finest-level cell (which is the only place a
/// cell can exceed `chunk_target` — coarser cells only ever carry merged sums
/// that already fit), sub-splits the cell into `8^k` deeper octree voxels at
/// `grid_depth + k` so each sub-chunk targets ~`chunk_target` points.
///
/// The deeper sub-chunk at local octant `(sx, sy, sz)` (each in `0..2^k`) has
/// voxel coordinates `(gx << k | sx, ...)` — a strict refinement of the cell,
/// so its ancestor at `grid_depth` is exactly `(gx, gy, gz)`. Distribute
/// reproduces this mapping with a `point_to_key` descent to `grid_depth + k`.
///
/// All `8^k` sub-octants are emitted with the uniform estimate `pop / 8^k`,
/// even ones that may turn out empty: the coarse grid carries no sub-cell
/// occupancy, and the build phase skips empty chunks cheaply. `k` is bounded
/// by [`SPLIT_LEVEL_CAP`], so the fan-out per cell is at most `8^cap`.
#[allow(clippy::too_many_arguments)]
fn emit_chunk_root(
    chunks: &mut Vec<PlannedChunk>,
    level: u32,
    grid_depth: u32,
    gx: i32,
    gy: i32,
    gz: i32,
    pop: u64,
    chunk_target: u64,
) {
    debug_assert!(level <= grid_depth, "cell level above grid depth");
    let k = if level == grid_depth {
        split_extra_levels(pop, chunk_target)
    } else {
        // Coarser-level cells only ever hold a merged sum ≤ chunk_target by
        // construction, so they never need splitting.
        0
    };

    if k == 0 {
        chunks.push(PlannedChunk {
            level,
            gx,
            gy,
            gz,
            point_count: pop,
        });
        return;
    }

    let sub_per_axis = 1i32 << k;
    let n_sub = 1u64 << (3 * k); // 8^k
    let est_each = pop / n_sub;
    // Spread the integer-division remainder one point per sub-chunk over the
    // first `remainder` sub-chunks, so the plan's per-chunk estimates sum back
    // to `pop` (preview/analyze checks total conservation) while no estimate
    // exceeds `est_each + 1`. Since `k` was chosen so `est_each <= chunk_target`,
    // every sub-chunk estimate stays at or just below target. The real
    // per-sub-chunk counts come from the data at distribute/build time; this
    // estimate only needs to conserve and stay roughly balanced.
    let mut remainder = pop - est_each * n_sub;
    let sub_level = grid_depth + k;
    let base_x = gx << k;
    let base_y = gy << k;
    let base_z = gz << k;
    for sz in 0..sub_per_axis {
        for sy in 0..sub_per_axis {
            for sx in 0..sub_per_axis {
                let point_count = if remainder > 0 {
                    remainder -= 1;
                    est_each + 1
                } else {
                    est_each
                };
                chunks.push(PlannedChunk {
                    level: sub_level,
                    gx: base_x | sx,
                    gy: base_y | sy,
                    gz: base_z | sz,
                    point_count,
                });
            }
        }
    }
}

/// Walk the implicit pyramid from the finest level to the root, collapsing
/// 2×2×2 groups whose combined point count fits within `chunk_target`.
///
/// Returns the list of chunk roots: cells that ended up either too dense to
/// merge into their parent, or that sit at level 0 because the entire dataset
/// fits in one chunk. Over-target finest-level cells are sub-split into deeper
/// octree sub-chunks via [`emit_chunk_root`].
fn merge_sparse_cells(
    grid: &[AtomicU32],
    grid_size: u32,
    grid_depth: u32,
    chunk_target: u64,
) -> Vec<PlannedChunk> {
    let g = grid_size as usize;

    // Convert atomic counts at the finest level to plain i64s. We use i64
    // so we can store SENTINEL (-1) for blocked cells.
    let mut current: Vec<i64> = grid
        .iter()
        .map(|c| c.load(Ordering::Relaxed) as i64)
        .collect();
    let mut current_size = g;

    let mut chunks: Vec<PlannedChunk> = Vec::new();

    // Iterate from the finest level down to level 1. At each step we
    // collapse groups of 8 children at level L into one parent at level L-1.
    for level in (1..=grid_depth).rev() {
        let parent_size = current_size / 2;
        let mut parent: Vec<i64> = vec![0; parent_size * parent_size * parent_size];

        for pz in 0..parent_size {
            for py in 0..parent_size {
                for px in 0..parent_size {
                    // Gather the 8 children of this parent at level `level`.
                    let mut sum: i64 = 0;
                    let mut any_blocked = false;
                    let mut child_pops = [0i64; 8];
                    let mut child_idx = 0;
                    for dz in 0..2 {
                        for dy in 0..2 {
                            for dx in 0..2 {
                                let cx = px * 2 + dx;
                                let cy = py * 2 + dy;
                                let cz = pz * 2 + dz;
                                let idx = cx + cy * current_size + cz * current_size * current_size;
                                let pop = current[idx];
                                child_pops[child_idx] = pop;
                                child_idx += 1;
                                if pop == SENTINEL {
                                    any_blocked = true;
                                } else {
                                    sum += pop;
                                }
                            }
                        }
                    }

                    let parent_idx = px + py * parent_size + pz * parent_size * parent_size;

                    if any_blocked || sum as u64 > chunk_target {
                        // Cannot merge: parent is blocked. Each non-empty,
                        // non-blocked child becomes a chunk root at `level`.
                        let mut child_idx = 0;
                        for dz in 0..2 {
                            for dy in 0..2 {
                                for dx in 0..2 {
                                    let pop = child_pops[child_idx];
                                    child_idx += 1;
                                    if pop > 0 {
                                        emit_chunk_root(
                                            &mut chunks,
                                            level,
                                            grid_depth,
                                            (px * 2 + dx) as i32,
                                            (py * 2 + dy) as i32,
                                            (pz * 2 + dz) as i32,
                                            pop as u64,
                                            chunk_target,
                                        );
                                    }
                                }
                            }
                        }
                        parent[parent_idx] = SENTINEL;
                    } else if sum > 0 {
                        // Merge: children's points are absorbed into the parent.
                        parent[parent_idx] = sum;
                    }
                    // else sum == 0 → leave parent as 0
                }
            }
        }

        current = parent;
        current_size = parent_size;
    }

    // After collapsing all the way to level 0, `current` has exactly one
    // entry. If it's > 0 (and not blocked) the entire dataset fits in one
    // chunk at level 0.
    debug_assert_eq!(current.len(), 1);
    let root_pop = current[0];
    if root_pop > 0 {
        // The whole dataset merged into one cell at level 0, which only
        // happens when the total fits `chunk_target` — so no split. (If any
        // finest cell had been over target it would have blocked the pyramid
        // all the way up and `root_pop` would be SENTINEL here.) Route through
        // `emit_chunk_root` anyway so the no-split path stays in one place.
        emit_chunk_root(
            &mut chunks,
            0,
            grid_depth,
            0,
            0,
            0,
            root_pop as u64,
            chunk_target,
        );
    }

    chunks
}

// ---------------------------------------------------------------------------
// Worker count helper
// ---------------------------------------------------------------------------

/// ~2/3 of cores, leaving headroom for the LAZ parallel decoder. Capped
/// by input file count.
fn distribute_worker_count(input_file_count: usize) -> usize {
    let cores = rayon::current_num_threads();
    let target = ((cores * 2) / 3).max(2);
    target.min(input_file_count).max(1)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_grid_thresholds() {
        // With a generous budget (16 GB) the point-count tiers dominate.
        let big_budget = 16 * 1024 * 1024 * 1024;
        assert_eq!(select_grid_size(1_000_000, big_budget), 128);
        assert_eq!(select_grid_size(99_999_999, big_budget), 128);
        assert_eq!(select_grid_size(100_000_000, big_budget), 256);
        assert_eq!(select_grid_size(499_999_999, big_budget), 256);
        assert_eq!(select_grid_size(500_000_000, big_budget), 512);
        assert_eq!(select_grid_size(100_000_000_000, big_budget), 512);
    }

    #[test]
    fn select_grid_size_falls_back_on_tiny_budget() {
        // Tiny budget can't afford 256³ or 512³ grids → fall back to 128.
        let tiny = 512 * 1024 * 1024;
        assert_eq!(select_grid_size(10_000_000_000, tiny), 128);
    }

    #[test]
    fn select_grid_size_256_fits_mid_budget() {
        // Mid budget fits 256³ but not 512³ → downgrade preferred 512 to 256.
        let mid = 4 * 1024 * 1024 * 1024;
        assert_eq!(select_grid_size(10_000_000_000, mid), 256);
    }

    #[test]
    fn chunk_target_scales_with_budget() {
        // 64 GB budget, 5 workers: raw target hits the dynamic single-chunk cap.
        let target = compute_chunk_target(64 * 1024 * 1024 * 1024, 5, 42_800_000_000);
        assert!((45_000_000..=47_000_000).contains(&target), "got {target}");

        // 32 GB budget, 10 workers: raw target sits below the cap.
        let target = compute_chunk_target(32 * 1024 * 1024 * 1024, 10, 42_800_000_000);
        assert!((11_000_000..=13_000_000).contains(&target), "got {target}");

        // 16 GB budget, 4 workers: cap is binding.
        let target = compute_chunk_target(16 * 1024 * 1024 * 1024, 4, 42_800_000_000);
        assert!((10_500_000..=11_800_000).contains(&target), "got {target}");
    }

    #[test]
    fn chunk_target_dynamic_max_scales_with_budget() {
        // Tiny budget must shrink the cap so no single chunk can blow it.
        let target = compute_chunk_target(4 * 1024 * 1024 * 1024, 1, 10_000_000_000);
        assert!(target <= 3_000_000, "got {target}, must be ≤ 3M");
        assert!(target >= MIN_CHUNK_POINTS);
    }

    #[test]
    fn chunk_target_parallelism_floor() {
        // Small dataset (50M points), big budget, modest workers — without the
        // parallelism floor, we'd get one chunk total.
        let target = compute_chunk_target(64 * 1024 * 1024 * 1024, 5, 50_000_000);
        // Parallelism floor: 50M / (5 * 4) = 2.5M per chunk
        assert!((2_000_000..=3_000_000).contains(&target));
    }

    #[test]
    fn chunk_target_clamps_to_min() {
        // Tiny budget that would otherwise produce sub-1M chunks.
        let target = compute_chunk_target(512 * 1024 * 1024, 32, 42_800_000_000);
        assert_eq!(target, MIN_CHUNK_POINTS);
    }

    #[test]
    fn split_extra_levels_thresholds() {
        // At or below target → no split.
        assert_eq!(split_extra_levels(0, 100), 0);
        assert_eq!(split_extra_levels(100, 100), 0);
        // Just over target → 1 level (8 sub-chunks, ~target/8 each... but the
        // gate is pop/8^k <= target, so 101/8 = 12 <= 100 → k = 1).
        assert_eq!(split_extra_levels(101, 100), 1);
        assert_eq!(split_extra_levels(800, 100), 1);
        // 8x over → still 1 (800/8 = 100 <= 100). 8x + 1 → 2.
        assert_eq!(split_extra_levels(801, 100), 2);
        // 648M into ~4.3M target (the motivating disjoint-NL case): need
        // ceil(log8(648M/4.3M)) = ceil(log8(150.7)) = 3.
        assert_eq!(split_extra_levels(648_000_000, 4_300_000), 3);
        // Clamped at the cap for absurd ratios.
        assert_eq!(split_extra_levels(u64::MAX, 1), SPLIT_LEVEL_CAP);
    }

    #[test]
    fn emit_chunk_root_conserves_and_bounds() {
        // An over-target finest cell sub-splits into 8^k sub-chunks whose
        // estimates sum back to the original population and none of which
        // exceeds the target by more than the one-point spread.
        let mut chunks = Vec::new();
        let pop = 10_000u64;
        let target = 100u64;
        let grid_depth = 4;
        emit_chunk_root(&mut chunks, grid_depth, grid_depth, 5, 6, 7, pop, target);
        let k = split_extra_levels(pop, target);
        assert_eq!(chunks.len(), 1usize << (3 * k));
        // Conservation.
        let total: u64 = chunks.iter().map(|c| c.point_count).sum();
        assert_eq!(total, pop);
        // All sub-chunks at the deeper level, within the ancestor cell.
        assert!(chunks.iter().all(|c| c.level == grid_depth + k));
        assert!(
            chunks
                .iter()
                .all(|c| (c.gx >> k) == 5 && (c.gy >> k) == 6 && (c.gz >> k) == 7)
        );
        // Distinct positions (no duplicate sub-chunks).
        let mut seen = std::collections::HashSet::new();
        for c in &chunks {
            assert!(
                seen.insert((c.gx, c.gy, c.gz)),
                "duplicate sub-chunk position"
            );
        }
        // Balanced: max estimate is within 1 of the floor estimate.
        let est_floor = pop / (1u64 << (3 * k));
        assert!(chunks.iter().all(|c| c.point_count <= est_floor + 1));
    }

    #[test]
    fn emit_chunk_root_no_split_below_target() {
        // A cell at or below target emits exactly one chunk at its own level.
        let mut chunks = Vec::new();
        emit_chunk_root(&mut chunks, 3, 5, 1, 2, 3, 50, 100);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].level, 3);
        assert_eq!(chunks[0].point_count, 50);
        assert_eq!((chunks[0].gx, chunks[0].gy, chunks[0].gz), (1, 2, 3));
    }

    #[test]
    fn merge_single_chunk_when_fits() {
        // 8x8x8 grid (depth 3), all 512 cells have 1 point each → 512 total.
        // Chunk target 1000 → everything merges into one root at level 0.
        let n = 8 * 8 * 8;
        let grid: Box<[AtomicU32]> = (0..n).map(|_| AtomicU32::new(1)).collect();
        let chunks = merge_sparse_cells(&grid, 8, 3, 1000);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].level, 0);
        assert_eq!(chunks[0].point_count, 512);
    }

    #[test]
    fn merge_splits_when_target_exceeded() {
        // 8x8x8 grid (depth 3), all cells have 100 points each → 51,200 total.
        // Chunk target 50 → no merge possible at any level, and every finest
        // cell is over target (100 > 50), so each sub-splits. split_extra_levels
        // (100, 50) = 1, so each of the 512 cells fans out into 8 sub-chunks at
        // level 4. Point counts are conserved across the split.
        let n = 8 * 8 * 8;
        let grid: Box<[AtomicU32]> = (0..n).map(|_| AtomicU32::new(100)).collect();
        let chunks = merge_sparse_cells(&grid, 8, 3, 50);
        assert_eq!(chunks.len(), 512 * 8);
        assert!(chunks.iter().all(|c| c.level == 4));
        // Each sub-chunk is at or below target; conservation holds overall.
        assert!(chunks.iter().all(|c| c.point_count <= 50));
        let total: u64 = chunks.iter().map(|c| c.point_count).sum();
        assert_eq!(total, 51_200);
    }

    #[test]
    fn merge_no_split_when_finest_cell_fits() {
        // 8x8x8 grid, all cells have 100 points each, target 100 → finest cells
        // are exactly at target (not over), so no sub-split. Merging is still
        // blocked one level up (8×100 = 800 > 100), so every finest cell is its
        // own chunk root at level 3, as before sub-splitting existed.
        let n = 8 * 8 * 8;
        let grid: Box<[AtomicU32]> = (0..n).map(|_| AtomicU32::new(100)).collect();
        let chunks = merge_sparse_cells(&grid, 8, 3, 100);
        assert_eq!(chunks.len(), 512);
        assert!(chunks.iter().all(|c| c.level == 3 && c.point_count == 100));
    }

    #[test]
    fn merge_partial_collapse() {
        // 4x4x4 grid (depth 2). One corner cell has 10000 points; all other
        // cells have 1 point each. Chunk target 100.
        //
        // The dense corner cell (10000 > 100) is over target, so it sub-splits:
        // split_extra_levels(10000, 100) = 3, fanning out into 8³ = 512
        // sub-chunks at level 2 + 3 = 5. Its 7 sibling fine cells (1 pt each)
        // stay at level 2; the other octants merge as before. Point counts are
        // conserved across the split (the integer-division remainder is handed
        // to the first sub-chunk).
        let n = 4 * 4 * 4;
        let grid: Box<[AtomicU32]> = (0..n)
            .map(|i| {
                if i == 0 {
                    AtomicU32::new(10000)
                } else {
                    AtomicU32::new(1)
                }
            })
            .collect();
        let chunks = merge_sparse_cells(&grid, 4, 2, 100);
        // Total points conservation across the sub-split.
        let total_pts: u64 = chunks.iter().map(|c| c.point_count).sum();
        assert_eq!(total_pts, 10000 + 63);
        // The dense cell was sub-split: there are sub-chunks deeper than the
        // grid depth (2), and no single chunk exceeds the target.
        assert!(chunks.iter().any(|c| c.level > 2));
        assert!(
            chunks.iter().all(|c| c.point_count <= 100),
            "no chunk should exceed target after sub-split"
        );
    }
}
