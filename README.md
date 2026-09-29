# copc_converter

[![Crates.io](https://img.shields.io/crates/v/copc_converter)](https://crates.io/crates/copc_converter)
[![docs.rs](https://docs.rs/copc_converter/badge.svg)](https://docs.rs/copc_converter)

A fast, memory-efficient converter that turns LAS/LAZ point cloud files into [COPC](https://copc.io/) (Cloud-Optimized Point Cloud) files.

## Features

- Produces spec-compliant COPC 1.0 files (LAS 1.4, point format 6, 7, or 8 — automatically chosen from input)
- Merges multiple input files into a single COPC output
- Out-of-core processing with a configurable memory budget — handles datasets larger than RAM
- Stays within its memory limit even on adversarial input (millions of coincident points, volumetric data, very large Extra Bytes, LAZ files written as one huge chunk); in a container the limit is detected from its cgroup
- Parallel reading, octree construction, and LAZ compression via rayon
- Preserves WKT and GeoTIFF CRS from input files (GeoTIFF EPSG codes are translated to WKT for the output)
- Preserves LAS Extra Bytes (per-point user-defined attributes such as classification probabilities, intensity ratios, or producer-specific labels) end-to-end, with per-file min/max stats merged honestly into the output VLR
- Optional temporal index for GPS-time-based filtering ([spec](https://github.com/360-geo/copc/blob/master/copc-temporal/docs/temporal-index-spec.md))
- Output checked in CI: every test conversion is decoded and validated against the COPC 1.0 / LAS 1.4 specs, and also read by PDAL and [copc-validator](https://github.com/hobuinc/copc-validator)

## Installation

Requires Rust 1.88+.

### From crates.io

```sh
cargo install copc_converter
```

### From source

```sh
git clone https://github.com/360-geo/copc-converter.git
cd copc-converter
cargo install --path .
```

This installs the `copc_converter` binary to `~/.cargo/bin/`, which should be on your `PATH`.

### Pre-built binaries

Download pre-built binaries from the [GitHub releases](https://github.com/360-geo/copc-converter/releases) page. These are built for broad compatibility and run on any machine.

For best performance, compile for your own CPU's instruction set (AVX2, NEON, etc.). Installing from a clone with `cargo install --path .` does this automatically: the repository's `.cargo/config.toml` sets `target-cpu=native`. `cargo install copc_converter` from crates.io ignores that file, so set the flag yourself:

```sh
RUSTFLAGS="-C target-cpu=native" cargo install copc_converter
```

## Usage

```sh
# Single file
copc_converter input.laz output.copc.laz

# Directory of LAZ/LAS files
copc_converter ./tiles/ merged.copc.laz
```

### Options

| Flag | Description | Default |
|---|---|---|
| `--memory-limit` | Memory limit to stay within (`16G`, `16Gi`, `4096M`, `512Mi`, etc.; binary units). The converter budgets 75% of it, leaving headroom | tightest cgroup limit (`memory.max`/`memory.high`, v1 `memory.limit_in_bytes`), capped at system RAM |
| `--threads` | Max parallel threads | all cores |
| `--temp-dir` | Directory for intermediate files. Use disk-backed storage: on a RAM-backed filesystem (tmpfs, a Kubernetes `emptyDir` with `medium: Memory`) scratch files count against the memory limit, and the converter warns | system temp |
| `--temporal-index` | Set the sampling stride for writing a temporal index EVLR for time-based queries (every n-th point). Good value (depending on density): 1000 | off |
| `--progress` | Progress output format: `bar`, `plain`, or `json` | `bar` |
| `--temp-compression` | Compress scratch temp files: `none` or `lz4` | `none` |
| `--node-storage` | Per-node temp layout: `files` or `packed` | `files` |

#### Temp file compression and node storage layout

Large conversions create a lot of scratch data. Two independent knobs
shape the temp directory's footprint:

- **`--temp-compression`** controls the on-disk encoding of each batch of
  `RawPoint` records. `none` (default) writes raw bytes; `lz4` wraps each
  batch in a self-contained LZ4 frame. On fast local disks it costs wall
  time (see below); on network filesystems (EFS/NFS) it can pay for itself
  because the bottleneck shifts from I/O to compute.
- **`--node-storage`** controls the filesystem layout of per-node point
  data during build. `files` (default) writes a separate file per octree
  node; on very large inputs node counts reach the hundred-thousands,
  which can exhaust inode budgets on shared scratch filesystems.
  `packed` writes all node data into a handful of append-only pack files
  (one per worker thread) with an in-memory key→offset index,
  independent of node count. The distribute stage's per-chunk scratch
  files (one per chunk; ~1,400 in the measurement below) remain either way.

Both flags can be combined freely.

**Measured on a 750M-point / 5.0 GB LAZ input** (three overlapping mobile-mapping
recordings, point format 7), 32 GB limit (24 GB budget), 10-core Apple M1 Pro with local NVMe:

| `--node-storage` | `--temp-compression` | wall   | peak inodes | peak temp bytes | output   |
|------------------|----------------------|--------|-------------|-----------------|----------|
| files            | none                 | 196.8s | 31 715      | 28 358 MB       | 5174 MB  |
| files            | lz4                  | 246.7s | 31 715      | 15 299 MB       | 5174 MB  |
| packed           | none                 | 197.8s | 1 430       | 29 701 MB       | 5174 MB  |
| packed           | lz4                  | 244.6s | 1 428       | 15 700 MB       | 5174 MB  |

LZ4 cuts peak temp bytes by ~46% regardless of storage mode, at ~25% more
wall time on this local disk; packed cuts peak inodes by ~95% regardless
of compression (what remains are the per-chunk scratch files). Output is
byte-identical across all four combinations. Pack-file overwrites leave
~5% dead space on this workload.

Use `packed` when the scratch filesystem has an inode limit, `lz4` when
it is space-constrained, and both together for the most disk-friendly
run on a modest wall-time budget.

### Examples

```sh
copc_converter ./my_survey/ survey.copc.laz --memory-limit 8G

# With temporal index (useful for multi-pass mobile mapping data),
# sampling every 1000th point
copc_converter ./my_survey/ survey.copc.laz --temporal-index 1000
```

## Library usage

The crate exposes a typestate pipeline API that enforces correct step ordering at compile time:

```rust
use copc_converter::{
    NodeStorage, Pipeline, PipelineConfig, TempCompression, collect_input_files,
};

let files = collect_input_files("./tiles/".into())?;
let config = PipelineConfig {
    memory_budget: 12_884_901_888,
    temp_dir: None,
    temporal_index: None,
    progress: None, // or Some(Arc::new(your_observer))
    chunk_target_override: None,
    temp_compression: TempCompression::None,
    node_storage: NodeStorage::Files,
};

Pipeline::scan(&files, config)?
    .validate()?
    .distribute()?
    .build()?
    .write("output.copc.laz")?;
```

## Tools

Optional analysis tools are available behind the `tools` feature:

```sh
cargo build --release --features tools
```

### inspect_copc

Inspect a COPC file's structure, or compare two files side-by-side. Works with local files and HTTP URLs.

```sh
# Inspect a single file
inspect_copc pointcloud.copc.laz

# Compare two files
inspect_copc pointcloud.copc.laz --compare other.copc.laz
```

Prints node counts, point distribution, compressed sizes, and compression ratios per octree level. When the file has a temporal index EVLR, also prints GPS time range, per-level temporal coverage, a time histogram, and sample density stats.

### preview_chunking

Preview how an input LAS/LAZ dataset would be partitioned during conversion, without actually writing anything:

```sh
preview_chunking input.laz [--memory-limit 16G] [--chunk-target 5M]
```

Prints chunk count, target size, grid resolution, and per-chunk size distribution. Useful for tuning `--memory-limit` before running a long conversion.

## How it works

1. **Scan** — reads headers from all input files in parallel to determine point count, point format, CRS (WKT or GeoTIFF), GPS time type, any LAS Extra Bytes schema, and provisional bounds.
2. **Validate** — checks that all input files share the same CRS, point format, GPS time type, and Extra Bytes schema, and selects the appropriate COPC output format (6, 7, or 8). Per-file Extra Bytes min/max stats are merged into a single canonical VLR at this stage.
3. **Count** — first full pass over the input: populates an occupancy grid used by the chunk planner to carve the dataset into chunks sized to be built in memory, and measures the points' actual extents. If the input headers turn out to be inaccurate, the octree is refitted to the actual extents and the pass is repeated.
4. **Distribute** — second full pass over the input: streams every point (including any trailing Extra Bytes) into its chunk's scratch file on disk, bounded by the configured memory budget.
5. **Build** — each chunk's sub-octree is built in parallel, then merged at coarse levels up to a single global root, thinning points at each level to produce multi-resolution LODs. Data too dense to split within the budget (e.g. coincident points) is streamed rather than loaded, and the merge samples one child node at a time.
6. **Write** — sorts each node's points by GPS time, compresses nodes in parallel in batches sized to the memory budget, and writes a single COPC file with a paged hierarchy EVLR for spatial indexing. The header records the points' actual extents.

## Acknowledgments

The chunked octree build is based on the counting-sort approach described in:

> Markus Schütz, Stefan Ohrhallinger, and Michael Wimmer. "Fast Out-of-Core Octree Generation for Massive Point Clouds." *Computer Graphics Forum*, 2020. [doi:10.1111/cgf.14134](https://doi.org/10.1111/cgf.14134)

## License

MIT
