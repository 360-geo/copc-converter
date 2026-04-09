//! Hierarchical counting-sort chunk planner (Schütz et al. 2020).
//!
//! This is a **measurement / planning** tool. It determines what chunks the
//! chunked-build path **would** produce for a given dataset, without actually
//! producing them. Used by the `analyze` CLI subcommand to validate the
//! chunking algorithm against real input data before committing to the full
//! chunked-build rewrite.
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

/// Estimated bytes per point held in memory while a chunk is being processed
/// during chunked build (`Vec<RawPoint>` + `Vec<(usize, RawPoint)>` sort buffer
/// + grid_sample output, plus a small constant for `HashSet` working state).
const PER_POINT_OVERHEAD_BYTES: u64 = 170;

/// Lower clamp on chunk target size (points). Below this, fixed per-chunk
/// overhead (file open, octree node bookkeeping) dominates point processing.
const MIN_CHUNK_POINTS: u64 = 1_000_000;

/// Upper clamp on chunk target size (points). Above this, single-chunk
/// allocations exceed ~2 GB and reduce parallelism on small datasets.
const MAX_CHUNK_POINTS: u64 = 50_000_000;

/// Aim for at least this many chunks per worker so small datasets still
/// have parallelism headroom.
const PARALLELISM_TARGET_PER_WORKER: u64 = 4;

// ---------------------------------------------------------------------------
// Chunk plan output
// ---------------------------------------------------------------------------

/// A single chunk in the plan: an octree voxel at some level, plus its
/// estimated point count derived from the counting grid.
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
    let max_for_parallelism = if parallelism_target > 0 {
        (total_points / parallelism_target).max(MIN_CHUNK_POINTS)
    } else {
        u64::MAX
    };

    raw.min(max_for_parallelism)
        .clamp(MIN_CHUNK_POINTS, MAX_CHUNK_POINTS)
}

/// Pick the counting grid resolution based on total point count.
///
/// Matches Schütz et al. 2020 (paragraph 4.1.2): coarser grids for small
/// datasets save memory, finer grids for large datasets give better
/// adaptivity in dense regions.
pub fn select_grid_size(total_points: u64) -> u32 {
    if total_points < 100_000_000 {
        128
    } else if total_points < 500_000_000 {
        256
    } else {
        512
    }
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
    // The standalone analyzer treats counting as its own user-visible stage.
    // (When called from inside the chunked distribute path, the caller owns
    // the stage events instead.)
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
/// Used by both the analyze CLI tool and the chunked-build distribute path,
/// so they share the exact same chunking logic and produce the same plan
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
    let grid_size = select_grid_size(total_points);
    let grid_depth = grid_size.trailing_zeros();
    let num_workers = distribute_worker_count(input_files.len());
    let chunk_target = chunk_target_override
        .unwrap_or_else(|| compute_chunk_target(config.memory_budget, num_workers, total_points));

    info!(
        "Chunk plan: total_points={}, grid={}³ (depth {}), chunk_target={}, workers={}",
        total_points, grid_size, grid_depth, chunk_target, num_workers
    );

    // ----- Count pass -----
    let grid = count_points(builder, input_files, grid_size, grid_depth, config)?;

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

    // ----- Merge sparse cells -----
    let chunks = merge_sparse_cells(&grid, grid_size, grid_depth, chunk_target);

    Ok(ChunkPlan {
        grid_size,
        grid_depth,
        chunk_target,
        total_points,
        chunks,
    })
}

// ---------------------------------------------------------------------------
// Counting pass
// ---------------------------------------------------------------------------

/// Stream every input file in parallel, classifying each point into the
/// counting grid via `point_to_key` at the grid depth, and atomically
/// incrementing the corresponding cell counter.
///
/// Returns a flat `Box<[AtomicU32]>` of length `grid_size³`. The cell index
/// is `gx + gy*G + gz*G²`.
fn count_points(
    builder: &OctreeBuilder,
    input_files: &[PathBuf],
    grid_size: u32,
    grid_depth: u32,
    config: &PipelineConfig,
) -> Result<Box<[AtomicU32]>> {
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

    debug!(
        "Counting with {} workers, {} files per worker",
        num_workers, files_per_worker
    );

    (0..num_workers)
        .into_par_iter()
        .try_for_each(|worker_id| -> Result<()> {
            let start = worker_id * files_per_worker;
            let end = (start + files_per_worker).min(input_files.len());
            if start >= end {
                return Ok(());
            }

            // Reuse a single scratch buffer of raw scaled `i32` triples
            // across files in this worker to amortize allocation cost.
            // The raw LAZ reader decompresses directly into these; no
            // `las::Point` struct materialization in between.
            let mut xyz: Vec<[i32; 3]> = Vec::with_capacity(1_000_000);

            for path in &input_files[start..end] {
                let meta = crate::laz_read::LazFileMeta::read(path)?;
                let (fsx, fsy, fsz) = meta.scale;
                let (fox, foy, foz) = meta.offset;
                let mut reader = crate::laz_read::RawLazReader::open_selective(
                    path,
                    meta,
                    crate::laz_read::xyz_only_selection(),
                )?;
                loop {
                    xyz.clear();
                    let n = reader.read_xyz_into(&mut xyz, 1_000_000)?;
                    if n == 0 {
                        break;
                    }
                    for triple in &xyz {
                        // File-frame scaled ints → world coordinates (using
                        // the file's own scale/offset) → builder-frame
                        // round-tripped coordinates, matching the exact
                        // quantization the partition pass will apply.
                        let fwx = triple[0] as f64 * fsx + fox;
                        let fwy = triple[1] as f64 * fsy + foy;
                        let fwz = triple[2] as f64 * fsz + foz;
                        let ix = ((fwx - builder.offset_x) / builder.scale_x).round() as i32;
                        let iy = ((fwy - builder.offset_y) / builder.scale_y).round() as i32;
                        let iz = ((fwz - builder.offset_z) / builder.scale_z).round() as i32;
                        let wx = ix as f64 * builder.scale_x + builder.offset_x;
                        let wy = iy as f64 * builder.scale_y + builder.offset_y;
                        let wz = iz as f64 * builder.scale_z + builder.offset_z;
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
                    let done = progress.fetch_add(n as u64, Ordering::Relaxed) + n as u64;
                    // Reuse the existing pipeline progress channel so the
                    // analyze tool's CLI can show a familiar progress bar.
                    // The stage was started by the caller.
                    config.report(crate::ProgressEvent::StageProgress { done });
                }
            }
            Ok(())
        })?;

    Ok(grid)
}

// ---------------------------------------------------------------------------
// Merge sparse cells
// ---------------------------------------------------------------------------

/// Walk the implicit pyramid from the finest level to the root, collapsing
/// 2×2×2 groups whose combined point count fits within `chunk_target`.
///
/// Returns the list of chunk roots: cells that ended up either too dense to
/// merge into their parent, or that sit at level 0 because the entire dataset
/// fits in one chunk.
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
                                        chunks.push(PlannedChunk {
                                            level,
                                            gx: (px * 2 + dx) as i32,
                                            gy: (py * 2 + dy) as i32,
                                            gz: (pz * 2 + dz) as i32,
                                            point_count: pop as u64,
                                        });
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
        chunks.push(PlannedChunk {
            level: 0,
            gx: 0,
            gy: 0,
            gz: 0,
            point_count: root_pop as u64,
        });
    }

    chunks
}

// ---------------------------------------------------------------------------
// Worker count helper
// ---------------------------------------------------------------------------

/// Same heuristic as the per-leaf distribute path: ~2/3 of cores, leaving
/// headroom for the LAZ parallel decoder. Capped by input file count.
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
        assert_eq!(select_grid_size(1_000_000), 128);
        assert_eq!(select_grid_size(99_999_999), 128);
        assert_eq!(select_grid_size(100_000_000), 256);
        assert_eq!(select_grid_size(499_999_999), 256);
        assert_eq!(select_grid_size(500_000_000), 512);
        assert_eq!(select_grid_size(100_000_000_000), 512);
    }

    #[test]
    fn chunk_target_scales_with_budget() {
        // 64 GB budget, 5 workers, large dataset
        let target = compute_chunk_target(64 * 1024 * 1024 * 1024, 5, 42_800_000_000);
        // Raw: (64 GiB * 0.6) / (5 * 170) ≈ 48.5M; below MAX clamp.
        assert!((47_000_000..=50_000_000).contains(&target), "got {target}");

        // 32 GB budget, 10 workers, large dataset
        let target = compute_chunk_target(32 * 1024 * 1024 * 1024, 10, 42_800_000_000);
        // Raw: (32 GiB * 0.6) / (10 * 170) ≈ 12.1M, within bounds
        assert!((11_000_000..=13_000_000).contains(&target), "got {target}");

        // 16 GB budget, 4 workers, large dataset
        let target = compute_chunk_target(16 * 1024 * 1024 * 1024, 4, 42_800_000_000);
        // Raw: (16 GiB * 0.6) / (4 * 170) ≈ 15.1M, within bounds
        assert!((14_000_000..=17_000_000).contains(&target), "got {target}");
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
        // 8x8x8 grid, all cells have 100 points each → 51,200 total.
        // Chunk target 50 → no merge possible at any level. Every non-empty
        // fine cell becomes its own chunk root at level 3.
        let n = 8 * 8 * 8;
        let grid: Box<[AtomicU32]> = (0..n).map(|_| AtomicU32::new(100)).collect();
        let chunks = merge_sparse_cells(&grid, 8, 3, 50);
        assert_eq!(chunks.len(), 512);
        assert!(chunks.iter().all(|c| c.level == 3 && c.point_count == 100));
    }

    #[test]
    fn merge_partial_collapse() {
        // 4x4x4 grid (depth 2). One corner cell has 10000 points; all other
        // cells have 1 point each. Chunk target 100 →
        //  - the dense corner cell becomes a chunk root at level 2
        //  - the 7 sibling fine cells are siblings of a blocked parent and
        //    each become individual chunk roots at level 2
        //  - the other 7 octants at level 1 are sparse enough to merge upward
        //  - at level 0, the merged total of those 7 octants (= 56 points) is
        //    below the target so they all collapse into one root at level 0
        // Note: the level-1 parent that holds the dense cell is blocked, so
        // its parent at level 0 is also blocked → can't merge to level 0.
        // So we get: 8 chunk roots at level 2 (from the dense octant) +
        // 7 separate roots at level 1 (the other octants, each holding 8 pts).
        // Wait — level 0 is blocked because one of its children is blocked,
        // so the 7 other children of root each become chunk roots at level 1.
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
        // Total points conservation
        let total_pts: u64 = chunks.iter().map(|c| c.point_count).sum();
        assert_eq!(total_pts, 10000 + 63);
    }
}
