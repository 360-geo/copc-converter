# COPC converter
COPC stands for cloud-optimized point cloud. More about it can be found at https://github.com/copcio/copcio.github.io. The official specification can be found in https://github.com/copcio/copcio.github.io/blob/main/copc-specification-1.0.pdf.

Example implementations are 
- lascopcindex64 of the lastools (https://github.com/LAStools/LAStools), but this sometimes generates invalid files
- untwine (https://github.com/hobuinc/untwine)

I would like to have a COPC converter written in Rust with the following features:
- creation of standard-compliant COPC files
- operates on LAZ files
- uses the las crate (specifically read_all_points_into function) for fast reading and writing of LAZ files
- can create a single COPC file from many input files
- does so using at most 16 GB of RAM, so an out-of-core approach might be necessary

## CLI Usage

```
copc_converter [OPTIONS] <INPUT> <OUTPUT>
```

- `<INPUT>` — a single LAZ/LAS file or a directory containing them (positional, required)
- `<OUTPUT>` — output COPC file path (positional, required)
- `--memory-limit <SIZE>` — memory budget, e.g. "16G", "4096M" (default: "16G")
- `--temp-dir <DIR>` — temp directory for intermediate files (default: system temp)