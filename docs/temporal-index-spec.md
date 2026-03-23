# COPC Temporal Index Extension (v1)

## Motivation

COPC files organize points spatially in an octree, enabling efficient spatial queries. However, many workflows need efficient temporal filtering:

- Trajectory replay (show points collected in a time window)
- Time-windowed analysis (compare morning vs afternoon scans)
- Progressive time-based loading in viewers

This converter already sorts points by GPS time within each octree node. The temporal index captures sampled GPS times per node, allowing clients to quickly determine which nodes contain points in a given time range and approximately where within a node those points lie.

## EVLR Identification

| Field | Value |
|---|---|
| user_id | `copc_temporal` (null-padded to 16 bytes) |
| record_id | `1000` |

This EVLR is optional. Readers that do not recognize it can safely skip it. The file remains a valid COPC 1.0 file.

## Binary Layout

All values are **little-endian**.

### Header (16 bytes)

| Offset | Type | Field | Description |
|--------|------|-------|-------------|
| 0 | u32 | version | Format version (currently `1`) |
| 4 | u32 | stride | Sampling stride (every n-th point is sampled) |
| 8 | u32 | node_count | Number of node entries that follow |
| 12 | u32 | reserved | Must be `0` |

### Node Entry (repeated `node_count` times, variable size)

| Offset | Type | Field | Description |
|--------|------|-------|-------------|
| 0 | i32 | level | Octree level (0 = root) |
| 4 | i32 | x | Voxel X coordinate |
| 8 | i32 | y | Voxel Y coordinate |
| 12 | i32 | z | Voxel Z coordinate |
| 16 | u32 | sample_count | Number of GPS time samples |
| 20 | f64[] | samples | `sample_count` GPS time values |

Each node entry is `20 + sample_count * 8` bytes.

### Sampling Rules

For a node with `N` points (already sorted by GPS time) and a stride of `S`:

- Sample the GPS time at point indices `0, S, 2S, 3S, ...`
- Always include the last point (index `N-1`), even if it doesn't fall on a stride boundary
- Samples are therefore monotonically non-decreasing
- `samples[0]` is the minimum GPS time in the node
- `samples[sample_count - 1]` is the maximum GPS time in the node

Only nodes with at least one point are included in the index.

## Client Usage

### Finding nodes in a time range [t_start, t_end]

1. Parse the temporal index EVLR
2. For each node entry:
   - If `samples[last] < t_start` or `samples[0] > t_end`, skip the node entirely
   - Otherwise, the node contains points in the query range

### Estimating byte offsets within a node

For a matching node, binary-search the `samples` array to find the first sample >= `t_start` and last sample <= `t_end`. The sample index multiplied by `stride` gives an approximate point index. Multiply by `point_record_length` to get an approximate byte offset in the uncompressed chunk data.

Note: since chunks are LAZ-compressed, byte-level seeking within a compressed chunk is not directly possible. The approximate point index is useful after decompression to skip to the relevant portion.

## Compatibility

- This extension is **additive** — the file remains fully COPC 1.0 compliant
- The LAS 1.4 header's `number_of_evlrs` field is incremented to account for this EVLR
- Readers unaware of this extension will ignore EVLRs with an unrecognized `user_id`
- The `copc_temporal` user_id is distinct from the standard `copc` user_id to avoid any ambiguity

## Generator Notes

The `copc_converter` tool generates this EVLR when invoked with `--temporal-index`. The sampling stride can be configured with `--temporal-stride` (default: 1000).
