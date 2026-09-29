//! Compress an uncompressed LAS 1.4 file to LAZ with a chosen fixed chunk
//! size. Shared by the integration tests and `examples/rechunk_laz.rs`.
//!
//! PDAL and las-rs always write 50k-point LAZ chunks, but the converter must
//! also handle files written as a few huge chunks, which a parallel decoder
//! would hold whole in memory.

use anyhow::{Context, Result, bail, ensure};
use laz::{LasZipCompressor, LazVlrBuilder};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

const HEADER_SIZE: usize = 375;
const VLR_HEADER_SIZE: usize = 54;

pub fn rechunk_las_to_laz(input: &Path, output: &Path, chunk_points: u32) -> Result<()> {
    let mut src = BufReader::new(File::open(input).with_context(|| format!("opening {input:?}"))?);
    let mut header = vec![0u8; HEADER_SIZE];
    src.read_exact(&mut header)?;
    ensure!(&header[0..4] == b"LASF", "{input:?} is not a LAS file");
    ensure!(header[25] == 4, "only LAS 1.4 input is supported");
    let format = header[104];
    ensure!(format & 0x80 == 0, "{input:?} is already compressed");
    let u16_at = |at: usize| u16::from_le_bytes([header[at], header[at + 1]]);
    let u32_at = |at: usize| u32::from_le_bytes(header[at..at + 4].try_into().unwrap());
    let header_size = u16_at(94) as usize;
    let offset_to_points = u32_at(96) as usize;
    let num_vlrs = u32_at(100);
    let record_len = u16_at(105);
    let num_evlrs = u32_at(243);
    let num_points = u64::from_le_bytes(header[247..255].try_into().unwrap());
    ensure!(
        header_size == HEADER_SIZE,
        "unexpected header size {header_size}"
    );
    ensure!(num_evlrs == 0, "inputs with EVLRs are not supported");

    let base_len: u16 = match format {
        0 => 20,
        1 => 28,
        2 => 26,
        3 => 34,
        6 => 30,
        7 => 36,
        8 => 38,
        f => bail!("unsupported point format {f}"),
    };
    let vlr = LazVlrBuilder::default()
        .with_point_format(format, record_len - base_len)?
        .with_fixed_chunk_size(chunk_points)
        .build();
    let mut vlr_payload = Vec::new();
    vlr.write_to(&mut vlr_payload)?;

    // Existing VLRs are copied as-is; the laszip VLR is appended after them.
    let mut vlrs = vec![0u8; offset_to_points - HEADER_SIZE];
    src.read_exact(&mut vlrs)?;
    let mut laz_vlr = vec![0u8; VLR_HEADER_SIZE];
    laz_vlr[2..16].copy_from_slice(b"laszip encoded");
    laz_vlr[18..20].copy_from_slice(&22204u16.to_le_bytes());
    laz_vlr[20..22].copy_from_slice(&(vlr_payload.len() as u16).to_le_bytes());
    let new_offset = offset_to_points + VLR_HEADER_SIZE + vlr_payload.len();
    header[96..100].copy_from_slice(&(new_offset as u32).to_le_bytes());
    header[100..104].copy_from_slice(&(num_vlrs + 1).to_le_bytes());
    header[104] = format | 0x80;

    let mut out =
        BufWriter::new(File::create(output).with_context(|| format!("creating {output:?}"))?);
    out.write_all(&header)?;
    out.write_all(&vlrs)?;
    out.write_all(&laz_vlr)?;
    out.write_all(&vlr_payload)?;
    let mut compressor = LasZipCompressor::new(out, vlr)?;
    let batch = 100_000usize;
    let mut buf = vec![0u8; batch * record_len as usize];
    let mut left = num_points;
    while left > 0 {
        let n = (left as usize).min(batch);
        let bytes = &mut buf[..n * record_len as usize];
        src.read_exact(bytes)?;
        compressor.compress_many(bytes)?;
        left -= n as u64;
    }
    compressor.done()?;
    compressor.into_inner().flush()?;
    Ok(())
}
