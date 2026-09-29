//! Compress an uncompressed LAS file to LAZ with a chosen fixed chunk size.
//!
//! Test-data tool for the memory-limit CI job: it builds LAZ inputs written
//! as one huge chunk. The logic lives in `tests/common/rechunk.rs`, shared
//! with the integration tests.
//!
//! Usage: rechunk_laz <input.las> <output.laz> <chunk_points>

use anyhow::{Context, Result, bail};
use std::path::Path;

#[path = "../tests/common/rechunk.rs"]
mod rechunk;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let [_, input, output, chunk] = args.as_slice() else {
        bail!("usage: rechunk_laz <input.las> <output.laz> <chunk_points>");
    };
    let chunk_points: u32 = chunk.parse().context("chunk_points must be a u32")?;
    rechunk::rechunk_las_to_laz(Path::new(input), Path::new(output), chunk_points)
}
