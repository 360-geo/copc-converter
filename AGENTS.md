# COPC Converter

A Rust CLI that converts LAS/LAZ point cloud files into [COPC](https://copc.io/) (Cloud-Optimized Point Cloud) files.

## Agent rules

- **Never push, tag, or release without explicit user confirmation.** Always show what you intend to do and wait for approval before any action that affects the remote repository or creates a release.
- Run `cargo fmt` to fix formatting — don't manually edit whitespace.
- Do not add `Co-Authored-By` lines to commits.

## Spec & References

- COPC 1.0 specification: https://github.com/copcio/copcio.github.io/blob/main/copc-specification-1.0.pdf
- Reference implementations: [untwine](https://github.com/hobuinc/untwine), [LAStools lascopcindex64](https://github.com/LAStools/LAStools) (note: LAStools sometimes produces invalid files)

## Architecture

The public API is a typestate pipeline (`Pipeline<S>`) that enforces step ordering at compile time. Internal modules are `pub(crate)` — only `Pipeline`, `PipelineConfig`, and utility functions are public.

### Pipeline stages

```
Pipeline::scan(&files, config)?  -> Pipeline<Scanned>
  .validate()?                   -> Pipeline<Validated>
  .distribute()?                 -> Pipeline<Distributed>
  .build()?                      -> Pipeline<Built>
  .write(&output)?               -> ()
```

### Source files

| File | Purpose |
|---|---|
| `lib.rs` | Public API: `Pipeline<S>`, `PipelineConfig`, stage markers, utility functions |
| `main.rs` | CLI args (clap), calls the pipeline |
| `octree.rs` | `OctreeBuilder`, voxel key math, point distribution, octree construction; CRS detection (WKT + GeoTIFF) at scan time |
| `validate.rs` | Input validation (CRS, point format, GPS time, Extra Bytes schema) and stat merging across input files |
| `writer.rs` | COPC file writer with parallel LAZ encoding |
| `copc_types.rs` | COPC-specific structs (header, VLRs, hierarchy entries, temporal index) |
| `extra_bytes.rs` | LAS Extra Bytes VLR parsing, schema-vs-stats split, structural diff, stat merging |
| `node_store.rs` | Per-node point-data storage backends (`FileNodeStore`, `PackedNodeStore`) used during build |
| `chunking.rs` | Hierarchical counting-sort chunk planner (Schütz et al. 2020) used during distribute |
| `tools/` | Optional HTTP source adapter for `inspect_copc`, gated behind the `tools` feature |

### Key design decisions

- **Typestate pipeline**: compile-time enforcement of step ordering — can't distribute before validating, can't write before building
- **Minimal public API**: only `Pipeline`, `PipelineConfig`, stage markers, and two utility functions are public; all internal types are `pub(crate)`
- **Out-of-core**: points are written to per-voxel temp files during distribution to stay within a configurable memory budget (default 16 GB, applied with a 0.75 safety factor)
- **Temp cleanup**: `OctreeBuilder` implements `Drop` to remove the temp directory, ensuring cleanup even on error
- **Point formats**: automatically selects LAS point format 6, 7, or 8 based on input — uses the `las` crate for reading and `laz` for compression
- **Parallelism**: uses rayon throughout for reading, octree building, and LAZ compression
- **LOD thinning**: 128 grid cells per axis (matching untwine's CellCount) for good progressive rendering, declared in the spec-mandated `CopcInfo.spacing` as `2 × halfsize / 128`
- **CRS detection**: WKT VLRs are preferred; if absent, GeoTIFF EPSG codes are translated to WKT via the `crs-definitions` registry. Cross-format mismatches are caught via a best-effort trailing-EPSG parse on the WKT side
- **LAS Extra Bytes pass-through**: per-point trailing bytes and the Extra Bytes VLR are preserved end-to-end. Validation compares only the *structural* parts of the schema across inputs (field count, types, scale/offset) — per-file min/max/no_data stats are merged honestly (union of mins and maxes) into the output VLR
- **Node storage backends**: build-stage point data goes to either one file per octree node (`FileNodeStore`, default) or a small number of append-only pack files with an in-memory index (`PackedNodeStore`) when filesystem inode budgets matter
- **Temp compression**: scratch batches can be wrapped in self-contained LZ4 frames (`TempCompression::Lz4`) to cut disk footprint on network filesystems
- **Version in Cargo.toml**: kept as `0.0.0-dev`; CI patches it from the git tag at release time
- **Published crate excludes `tests/data/*`**: real-LAS test fixtures push the tarball over crates.io's 10 MiB limit. The lib still builds + tests-from-git, just not from the published tarball

## Development

```sh
cargo fmt            # format code
cargo clippy         # lint
cargo test           # run tests
```

CI runs all three on every push to `master` and on PRs. All must pass.

Every integration-test conversion goes through `tests/common` (`assert_valid_copc`), which decodes each chunk and checks the output against COPC 1.0 / LAS 1.4 R15. The `validator_catches_*` tests keep it honest. CI also runs two jobs that need external tools:

```sh
scripts/ci/external-validators.sh          # PDAL (full decode) + copc-validator --deep on converted fixtures; needs pdal, jq, npx
scripts/ci/memory-limits.sh gating         # converts synthetic inputs in 1 GB / 4-CPU Docker containers; any OOM kill fails
scripts/ci/memory-limits.sh known-issues   # inputs known to exceed the limit today; reported, never fails
```

`memory-limits.sh` needs a Linux binary (`BIN=`), the host-built `examples/rechunk_laz` (`RECHUNK=`, writes LAZ files with one huge chunk), Docker and PDAL. On macOS, build one with `docker run --rm -v "$PWD":/src -w /src -e CARGO_TARGET_DIR=/src/target/linux-docker -e RUSTFLAGS="" rust:1 cargo build --release`. Once a memory fix lands, move its case from `known-issues` to `gating`.

## Releasing

1. Move `## [Unreleased]` entries in `CHANGELOG.md` into a new `## [X.Y.Z] - YYYY-MM-DD` section and update the link references at the bottom. Leave an empty `## [Unreleased]` heading at the top for the next cycle.
2. Commit and push to `master`
3. Create and push a git tag: `git tag vX.Y.Z && git push origin vX.Y.Z`
4. Create a GitHub Release using the new CHANGELOG section as the body, with the auto-generated compare link appended:
   ```sh
   # Extract the [X.Y.Z] section (everything between its heading and the next ## heading) into a temp file:
   awk '/^## \[X\.Y\.Z\]/{flag=1; next} /^## /{flag=0} flag' CHANGELOG.md > /tmp/release-notes.md
   gh release create vX.Y.Z --title "vX.Y.Z" --generate-notes --notes-file /tmp/release-notes.md
   ```
   `--notes-file` provides the body; `--generate-notes` appends the `**Full Changelog**: …compare/…` link (and, on tags with multiple commits, a per-commit list).
5. CI triggers on the release event and automatically:
   - Builds binaries for linux (x86_64, aarch64), macOS (x86_64, aarch64), and Windows (x86_64)
   - Publishes to crates.io

**Important:** A git tag alone is not enough — the CI workflows trigger on `release: [published]`, so the GitHub Release (step 4) is required.

## Changelog

User-visible changes go in `CHANGELOG.md` under `## [Unreleased]` as part of the same commit that makes the change. Use Keep a Changelog groups (`Added`, `Changed`, `Fixed`, `Removed`, `Deprecated`, `Security`). Skip internal-only changes (refactors, test infra, dep bumps that don't affect users).
