# copc_converter

[![Crates.io](https://img.shields.io/crates/v/copc_converter)](https://crates.io/crates/copc_converter)

A fast, memory-efficient converter that turns LAS/LAZ point cloud files into [COPC](https://copc.io/) (Cloud-Optimized Point Cloud) files.

## Features

- Produces spec-compliant COPC 1.0 files (LAS 1.4, point format 7)
- Merges multiple input files into a single COPC output
- Out-of-core processing with a configurable memory budget — handles datasets larger than RAM
- Parallel reading, octree construction, and LAZ compression via rayon
- Preserves WKT CRS from input files

## Installation

Requires Rust 1.85+.

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

## Usage

```sh
# Single file
copc_converter input.laz -o output.copc.laz

# Multiple files
copc_converter tile1.laz tile2.laz tile3.laz -o merged.copc.laz

# Directory of LAZ/LAS files
copc_converter ./tiles/ -o merged.copc.laz
```

### Options

| Flag | Description | Default |
|---|---|---|
| `-o, --output` | Output COPC file path | *(required)* |
| `--memory-limit` | Max memory budget (`16G`, `4096M`, etc.) | `16G` |
| `--temp-dir` | Directory for intermediate files | system temp |

### Example

```sh
copc_converter ./my_survey/ -o survey.copc.laz --memory-limit 8G
```

## How it works

1. **Scan** — reads headers from all input files in parallel to determine bounds, CRS, and point count.
2. **Distribute** — reads every point, assigns it to an octree leaf voxel, and writes it to a temporary file on disk.
3. **Build** — constructs the octree bottom-up, thinning points at each level to produce multi-resolution LODs.
4. **Write** — encodes and compresses nodes in parallel into a single COPC file with a hierarchy EVLR for spatial indexing.

## License

MIT
