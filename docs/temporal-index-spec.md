# COPC Temporal Index Extension

**Version:** 1.0 draft

## 1. Introduction

COPC (Cloud-Optimized Point Cloud) files organize point data spatially in an octree, enabling efficient spatial queries. The COPC Temporal Index Extension adds an optional Extended Variable Length Record (EVLR) that enables efficient temporal queries over GPS time.

The temporal index provides a per-node lookup table of sampled GPS timestamps, allowing clients to:

- Determine which octree nodes contain points within a given time range without decompressing point data.
- Estimate the approximate point index within a node for a given time boundary.

A motivating use case is mobile mapping. A fleet of survey vehicles captures a city over days or weeks, producing a single merged COPC file. At busy intersections, the same spatial area may contain points from 10+ passes by different vehicles at different times. A client that needs the points for a specific location and a specific pass must currently decompress all spatially overlapping nodes and discard the majority of points belonging to other passes. The temporal index allows the client to combine spatial octree queries with time-range filtering, skipping nodes from unrelated passes before any decompression occurs.

## 2. Scope

This specification defines a single optional EVLR that may be appended to any COPC 1.0 file. A file with this EVLR remains fully COPC 1.0 compliant. Readers that do not recognize the EVLR SHALL ignore it per standard LAS 1.4 behavior.

## 3. Definitions

| Term | Definition |
|---|---|
| Node | An octree node identified by a VoxelKey (level, x, y, z) |
| Chunk | The compressed point data for a single node |
| Stride | The sampling interval: every S-th point is recorded in the index |
| Sample | A GPS time value recorded in the index |

## 4. Requirements

### 4.1 Point Format

Input files MUST use a LAS point format that includes GPS time (formats 1, 3, 4, 5, 6, 7, 8, 9, or 10). Formats 0 and 2 do not contain GPS time and are not compatible with this extension.

### 4.2 Point Ordering

Points within each octree node MUST be sorted in non-decreasing order of GPS time before the index is constructed. This ordering is a prerequisite for the index to be meaningful: it ensures that samples are monotonically non-decreasing and that sample positions correspond to contiguous point ranges.

### 4.3 COPC Compliance

The file MUST be a valid COPC 1.0 file. The LAS 1.4 header field `number_of_evlrs` MUST be incremented to account for the temporal index EVLR.

## 5. EVLR Identification

| Field | Value |
|---|---|
| user_id | `copc_temporal` (null-padded to 16 bytes) |
| record_id | `1000` |

The `copc_temporal` user_id is distinct from the standard `copc` user_id to avoid ambiguity with current or future COPC records.

## 6. Binary Layout

All multi-byte values are **little-endian**.

### 6.1 Header (16 bytes)

| Offset | Type | Field | Description |
|--------|------|-------|-------------|
| 0 | uint32 | version | Format version. MUST be `1`. |
| 4 | uint32 | stride | Sampling stride S. MUST be >= 1. |
| 8 | uint32 | node_count | Number of node entries that follow. |
| 12 | uint32 | reserved | MUST be `0`. Readers SHOULD ignore this field. |

### 6.2 Node Entry (repeated `node_count` times)

| Offset | Type | Field | Description |
|--------|------|-------|-------------|
| 0 | int32 | level | Octree level (0 = root) |
| 4 | int32 | x | Voxel X coordinate |
| 8 | int32 | y | Voxel Y coordinate |
| 12 | int32 | z | Voxel Z coordinate |
| 16 | uint32 | sample_count | Number of GPS time samples that follow |
| 20 | float64[] | samples | `sample_count` GPS time values |

Each node entry is `20 + sample_count * 8` bytes. Entries are variable-length; readers MUST use `sample_count` to determine where the next entry begins.

### 6.3 C Struct Definitions

```c
// VoxelKey identifies an octree node (same as COPC hierarchy).
struct VoxelKey
{
    int32_t level;
    int32_t x;
    int32_t y;
    int32_t z;
};

// Fixed-size header at the start of the temporal index EVLR payload.
struct TemporalIndexHeader
{
    uint32_t version;       // Must be 1
    uint32_t stride;        // Sampling stride (>= 1)
    uint32_t node_count;    // Number of TemporalIndexEntry records
    uint32_t reserved;      // Must be 0
};

// Variable-size entry for one octree node.
// Immediately followed by `sample_count` double-precision GPS times.
struct TemporalIndexEntry
{
    VoxelKey key;
    uint32_t sample_count;
    // double samples[sample_count];  // variable-length, not part of fixed struct
};
```

### 6.4 Node Entry Ordering

Node entries SHOULD appear in the same order as nodes in the COPC hierarchy EVLR. Only nodes with `point_count > 0` in the hierarchy (i.e., nodes that contain actual point data) SHALL appear in the temporal index.

## 7. Sampling Rules

Given a node containing `N` points sorted by GPS time and a stride of `S`:

1. The GPS time at point index `0` is always sampled.
2. The GPS time at each point index that is a multiple of `S` is sampled (indices `0, S, 2S, 3S, ...`).
3. The GPS time at point index `N-1` (the last point) is always sampled, even if `N-1` is not a multiple of `S`.
4. `sample_count` equals the number of distinct indices sampled by rules 1-3.

Consequently:

- `samples[0]` is the minimum GPS time in the node.
- `samples[sample_count - 1]` is the maximum GPS time in the node.
- All samples are in monotonically non-decreasing order.
- A node with a single point has `sample_count = 1`.

## 8. Client Usage

### 8.1 Node-Level Filtering

To find nodes containing points in a time range `[t_start, t_end]`:

1. Read and parse the temporal index EVLR.
2. For each node entry:
   - If `samples[sample_count - 1] < t_start`, the node contains no points in the range. Skip it.
   - If `samples[0] > t_end`, the node contains no points in the range. Skip it.
   - Otherwise, the node may contain points in the range.

### 8.2 Intra-Node Point Estimation

For a node that passes the filter in 8.1, a client can estimate the approximate point range:

1. Binary-search `samples` for the first index `i` where `samples[i] >= t_start`.
2. Binary-search `samples` for the last index `j` where `samples[j] <= t_end`.
3. The approximate starting point index is `i * stride`.
4. The approximate ending point index is `min(j * stride + stride - 1, point_count - 1)`.

Since chunks are LAZ-compressed, byte-level seeking within a compressed chunk is not possible. The estimated point range is applied after decompression.

## 9. Implementation Guide

The following examples illustrate common workflows. Pseudocode assumes the temporal index EVLR has been parsed into a list of `(voxel_key, samples[])` entries and that the COPC hierarchy is available.

### 9.1 Loading a COPC File with a Time Filter

Goal: load only the points that fall within a user-specified time window `[t_start, t_end]`.

```
temporal_index = read_temporal_index_evlr(file)
hierarchy      = read_hierarchy_evlr(file)
stride         = temporal_index.header.stride

for entry in temporal_index:
    # Skip nodes entirely outside the time window
    if entry.samples[last] < t_start:  continue
    if entry.samples[0]    > t_end:    continue

    # Use samples to estimate the point range within this node
    i = binary_search_first(entry.samples, >= t_start)
    j = binary_search_last(entry.samples, <= t_end)
    approx_start = i * stride
    approx_end   = min((j + 1) * stride, hierarchy[entry.key].point_count)

    # Decompress the chunk and scan only the estimated range
    chunk = decompress_chunk(file, hierarchy[entry.key])

    # Start at the estimated position; scan forward/backward to exact boundary
    for point in chunk[approx_start .. approx_end]:
        if point.gps_time > t_end:   break
        if point.gps_time >= t_start: emit(point)
```

The samples narrow the search to approximately `(j - i + 1) * stride` points instead of the entire node. With a stride of 1000 and a node of 100,000 points, a query matching one sample interval only scans ~1,000 points.

### 9.2 Finding Which Octree Nodes Contain a Given Time

Goal: given a single timestamp `t`, return all octree nodes whose points span that time.

```
temporal_index = read_temporal_index_evlr(file)

matching_nodes = []
for entry in temporal_index:
    if entry.samples[0] <= t <= entry.samples[last]:
        matching_nodes.append(entry.key)
```

This can be used to highlight or pre-fetch nodes in a viewer when scrubbing a timeline.

### 9.3 Building a Time Histogram Without Decompressing Points

Goal: produce a coarse histogram of point density over time without reading any point data.

```
temporal_index = read_temporal_index_evlr(file)
hierarchy      = read_hierarchy_evlr(file)
stride         = temporal_index.header.stride

for entry in temporal_index:
    point_count = hierarchy[entry.key].point_count

    # Each consecutive pair of samples spans approximately `stride` points.
    # The last interval may be shorter.
    for i in 0 .. entry.sample_count - 1:
        bin_start  = entry.samples[i]
        bin_end    = entry.samples[i + 1]
        if i == entry.sample_count - 2:
            bin_points = point_count - i * stride   # remainder
        else:
            bin_points = stride
        histogram.add(bin_start, bin_end, bin_points)
```

This builds an approximate time histogram using only the temporal index — no point data is decompressed. The resolution is determined by the stride.

### 9.4 Progressive Time-Based Loading in a Viewer

Goal: stream points into a 3D viewer ordered by acquisition time (e.g., to animate a scan trajectory).

```
temporal_index = read_temporal_index_evlr(file)
hierarchy      = read_hierarchy_evlr(file)
stride         = temporal_index.header.stride

# Build a global timeline from all samples across all nodes
timeline = []  # list of (gps_time, node_key, sample_index)
for entry in temporal_index:
    for i, t in enumerate(entry.samples):
        timeline.append((t, entry.key, i))

timeline.sort_by(gps_time)

# Stream point slices in temporal order
loaded_chunks = {}  # cache decompressed chunks
for (t, key, sample_idx) in timeline:
    if key not in loaded_chunks:
        loaded_chunks[key] = decompress_chunk(file, hierarchy[key])

    chunk      = loaded_chunks[key]
    slice_start = sample_idx * stride
    slice_end   = min(slice_start + stride, hierarchy[key].point_count)
    render(chunk[slice_start .. slice_end])
```

Each sample represents a stride-sized slice of a node. By sorting all samples globally, the viewer renders point slices in approximate acquisition order across the entire dataset, enabling trajectory animation without loading everything upfront.

## 10. Compatibility

- This EVLR is additive. A file containing it remains a valid COPC 1.0 file.
- The `copc_temporal` user_id does not collide with any user_id defined by the COPC 1.0 or LAS 1.4 specifications.
- Readers that do not recognize this EVLR SHALL skip it per LAS 1.4 EVLR handling rules.
- Writers that do not support this extension need not produce it.
