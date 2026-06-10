## Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- The counting and distribute stages now read LAZ input through the `las`
  byte-slab API (`PointData` / `Reader::fill_points`, from
  [las-rs#143](https://github.com/gadomski/las-rs/pull/143)), eliminating
  per-point `las::Point` materialization — including the per-point heap
  allocation for Extra Bytes payloads. Upstream benchmarks show 1.5–2×
  faster reads with parallel LAZ decoding; both stages are full passes over
  the input, so large conversions should see a substantial wall-clock
  reduction. Until the PR is merged and released, the `las` dependency is
  pinned to a fork via `[patch.crates-io]`.
- Temp-file reads now decode points from bulk contiguous reads instead of
  one small read plus a heap allocation per point. Scratch data is re-read
  several times across the build, merge, and write stages, so this cuts
  allocator traffic substantially on large conversions.

## [0.11.0] - 2026-06-09

### Fixed

- Inputs whose dense data is spread across a very large extent (e.g. a few
  small dense regions scattered across a whole country) no longer OOM during
  the build step. The counting grid has a fixed resolution, so over a huge
  extent a single grid cell can cover kilometres of ground and collect far
  more than the chunk target — the planner then emitted one enormous chunk per
  such cell, and the build loaded it whole into memory. The chunk planner now
  sub-splits an over-target finest-grid cell into deeper octree sub-chunks
  (routed during distribute via a deeper descent), and the build step gained a
  spill path that sub-divides any chunk whose actual point distribution still
  exceeds the per-chunk memory budget. The build and merge steps now also bound
  how many chunks they hold in memory at once — concurrency scales with the
  memory budget (and cores) instead of running one chunk per core regardless of
  budget, so peak build memory tracks `--memory-limit` from ~1 GB up to large
  multi-core hosts. Together these guarantee bounded build memory regardless of
  how points concentrate, letting large, geographically sparse datasets convert
  within a small `--memory-limit`.
- Such inputs no longer produce a wildly over-fragmented octree (millions of
  tiny, nearly-empty deep nodes). When the build had to split a region deep for
  memory reasons, those deep voxels became permanent output nodes even where the
  data was sparse, bloating the node count ~100× and, in turn, the writer's
  memory. The bottom-up build now collapses any subtree whose points fit a
  single leaf into one node, so output node depth tracks point *density* rather
  than the build's memory-driven split depth. The writer's per-node bookkeeping
  was also reworked to keep a single key+count list instead of several parallel
  copies, so it scales to large node counts without holding gigabytes of index.
- The writer no longer OOMs when producing very large outputs. It used to
  encode a whole memory-budget-sized batch of nodes before compressing, holding
  the entire batch's encoded bytes (which could be ~half the budget) resident
  alongside the compressor's working set. It now streams a fixed-size window of
  points at a time — encode in parallel, compress in order, free, advance — so
  the writer's peak memory is a small constant independent of the output size or
  `--memory-limit`.

## [0.10.1] - 2026-05-28

### Fixed

- Multi-file inputs with disjoint extents no longer collapse outlying points
  onto a single wrong coordinate. The output's scale and offset were taken
  blindly from the first input file, so points whose true coordinate sat more
  than `i32::MAX × scale` from that offset would saturate during the
  `f64 → i32` cast and pile up on the i32 boundary (e.g. ~22 000 m of Y
  collapsed onto a single Y value). `OctreeBuilder::from_scan` now re-centers
  the offset on the combined-bounds midpoint when the first file's offset
  doesn't span the merged extent, and doubles the scale if even the
  half-extent still won't fit in i32. (#20)

## [0.10.0] - 2026-05-26

### Changed

- **Breaking:** Merged `--temporal-index` and `--temporal-stride` into a single
  `--temporal-index <STRIDE>` option. Passing the flag now both enables the
  temporal index EVLR and sets the sampling stride (every n-th point); omitting
  it disables the index. The previous `--temporal-index` boolean flag and the
  separate `--temporal-stride` option are gone. (#19)
- **Breaking (API):** `PipelineConfig.temporal_index` changed from `bool` to
  `Option<u32>` and the `temporal_stride` field was removed. Migrate
  `temporal_index: true, temporal_stride: 1000` to `temporal_index: Some(1000)`,
  and `temporal_index: false` to `temporal_index: None`.

### Added

- New `Error::InvalidTemporalStride` variant returned when the temporal index is
  requested with a stride that can't produce a valid index.

## [0.9.15] - 2026-05-20

### Fixed

- Header-bounds mismatch warning no longer fires on float round-tripping noise.
  The tolerance is now 1.5 scale units per axis (was strictly > 1 scale unit),
  which absorbs the few-ULP overshoot from `int32 × scale + offset` reconstruction
  and the common case of LAS headers stored one decimal coarser than point
  precision. A warning now indicates a real ≥2-unit disagreement worth investigating.

## [0.9.14] - 2026-05-18

### Changed

- Bumped dependencies to clear 10 Dependabot alerts.

## [0.9.13] - 2026-05-18

### Added

- GeoTIFF CRS support. When a LAS file has no WKT CRS VLR but does carry GeoTIFF
  keys, the EPSG code is now translated to WKT via the `crs-definitions` registry
  and propagated into the COPC output. Cross-format mismatches between WKT and
  GeoTIFF inputs are caught via a best-effort trailing-EPSG parse. (#13)

## [0.9.12] - 2026-05-11

### Changed

- Extra Bytes validation now compares only the *structural* parts of the schema
  across input files (field count, types, scale/offset). Per-file min/max/no_data
  stats are allowed to differ and are merged honestly (union of mins, union of
  maxes) into the output VLR.

## [0.9.11] - 2026-05-11

### Fixed

- Exclude `tests/data/*` from the published crate tarball. Real-LAS test fixtures
  pushed the tarball over crates.io's 10 MiB limit. The library still builds and
  tests from git, just not from the published tarball.

## [0.9.10] - 2026-05-11

### Added

- LAS Extra Bytes pass-through. The `LASF_Spec/4` Extra Bytes VLR and every
  point's trailing extra bytes are now carried from input to output unchanged.
  Previously both were silently dropped, losing any classification or semantic
  data stored in extras (a common pattern for ML-labelled or research datasets).
  Validation enforces an identical VLR and uniform `num_extra_bytes` across
  inputs.

[Unreleased]: https://github.com/360-geo/copc-converter/compare/v0.11.0...HEAD
[0.11.0]: https://github.com/360-geo/copc-converter/compare/v0.10.1...v0.11.0
[0.10.1]: https://github.com/360-geo/copc-converter/compare/v0.10.0...v0.10.1
[0.10.0]: https://github.com/360-geo/copc-converter/compare/v0.9.15...v0.10.0
[0.9.15]: https://github.com/360-geo/copc-converter/compare/v0.9.14...v0.9.15
[0.9.14]: https://github.com/360-geo/copc-converter/compare/v0.9.13...v0.9.14
[0.9.13]: https://github.com/360-geo/copc-converter/compare/v0.9.12...v0.9.13
[0.9.12]: https://github.com/360-geo/copc-converter/compare/v0.9.11...v0.9.12
[0.9.11]: https://github.com/360-geo/copc-converter/compare/v0.9.10...v0.9.11
[0.9.10]: https://github.com/360-geo/copc-converter/compare/v0.9.9...v0.9.10
