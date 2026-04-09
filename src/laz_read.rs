//! Direct LAZ decompression via `laz-rs`, bypassing `las::Reader`'s
//! per-point `las::Point` struct materialization.
//!
//! `las::Reader::read_points_into` does two things per point after the
//! (parallel, SIMD-friendly) bulk LAZ decompression:
//!
//! 1. Parses the raw LAS point record bytes into a `raw::Point` via 14
//!    `byteorder` reads through a `Cursor`.
//! 2. Calls `Point::new(raw_point, transforms)` to materialize a fully
//!    typed `las::Point` — applies f64 scale/offset to x/y/z, unpacks
//!    flag bits, clones the classification enum, allocates
//!    `Option<Color>`, `Option<Waveform>`, `extra_bytes: Vec<u8>`, etc.
//!
//! For the chunked build's two full passes over the input, every single
//! one of those `las::Point` values is immediately torn apart by
//! `OctreeBuilder::convert_point`, which round-trips x/y/z *back* to
//! scaled i32 and discards almost every other field. See
//! gadomski/las-rs#121 for context.
//!
//! This module skips that machinery: it calls `laz::ParLasZipDecompressor`
//! directly, parses point record bytes into a flat `i32`/`u16`/`f64`
//! layout that matches `RawPoint`, and (for the counting pass) uses
//! `DecompressionSelection` to skip decompression of fields we don't
//! need.
//!
//! # Supported input point formats
//!
//! LAS 1.2–1.4 define two families:
//!
//! - **Legacy formats (0–5)** use a one-byte flags word and store scan
//!   angle as a signed `i8` rank in [-90, 90]. Format lengths vary with
//!   optional GPS time, RGB, and waveform fields.
//! - **Extended formats (6–10)** use a two-byte flags word and store
//!   scan angle as an `i16` (in 0.006° units). All extended formats
//!   carry GPS time; RGB appears on 7/8/10; NIR appears on 8/10;
//!   waveform on 9/10.
//!
//! This reader supports all ten formats but discards fields that
//! `RawPoint` doesn't store (`waveform`, `extra_bytes`, and the
//! additional bit-flags in the two-byte extended flags word). Formats
//! 4, 5, 9, 10 are accepted by ignoring the waveform block at the end
//! of each record.
//!
//! ## Legacy record layout (formats 0–5)
//!
//! ```text
//! 0   i32  x
//! 4   i32  y
//! 8   i32  z
//! 12  u16  intensity
//! 14  u8   flags  (return_number 0-2 | number_of_returns 3-5 | scan_dir 6 | edge 7)
//! 15  u8   classification
//! 16  i8   scan_angle_rank      (stored as [-90, 90])
//! 17  u8   user_data
//! 18  u16  point_source_id
//! 20  f64  gps_time             (formats 1, 3, 4, 5)
//! 20/28  u16 red, u16 green, u16 blue  (formats 2, 3, 5 — offset depends on gps)
//! ```
//!
//! ## Extended record layout (formats 6–10)
//!
//! ```text
//! 0   i32  x
//! 4   i32  y
//! 8   i32  z
//! 12  u16  intensity
//! 14  u8   flags_a  (return_number lo nibble | number_of_returns hi nibble)
//! 15  u8   flags_b  (classification flags | scanner channel | scan dir | edge)
//! 16  u8   classification
//! 17  u8   user_data
//! 18  i16  scan_angle          (×0.006°, already in `RawPoint`'s native units)
//! 20  u16  point_source_id
//! 22  f64  gps_time
//! 30  u16 red, u16 green, u16 blue  (formats 7, 8, 10)
//! 36  u16 nir                        (formats 8, 10)
//! ```

use crate::octree::RawPoint;
use anyhow::{Context, Result};
use laz::las::selective::DecompressionSelection;
use laz::{LazVlr, ParLasZipDecompressor};
use std::fs::File;
use std::io::{BufReader, Seek, SeekFrom};
use std::path::Path;

/// Per-file LAS metadata needed to open a [`RawLazReader`].
#[derive(Debug, Clone)]
pub(crate) struct LazFileMeta {
    pub n_points: u64,
    pub point_format_id: u8,
    pub point_record_len: u16,
    pub offset_to_point_data: u64,
    pub scale: (f64, f64, f64),
    pub offset: (f64, f64, f64),
    pub laz_vlr: LazVlr,
}

impl LazFileMeta {
    /// Open `path` with `las::Reader` just long enough to extract header
    /// metadata and the LAZ VLR, then drop the reader.
    pub fn read(path: &Path) -> Result<Self> {
        let reader = las::Reader::from_path(path)
            .with_context(|| format!("opening {:?} for metadata", path))?;
        let header = reader.header();
        let format = header.point_format();
        let format_id = format
            .to_u8()
            .with_context(|| format!("unsupported point format in {:?}", path))?;
        // Validation layer already restricts inputs to point formats 0..=10
        // via `las::Reader::from_path` (which will itself reject anything
        // out of range). No re-check here.
        let t = header.transforms();
        let n_points = header.number_of_points();
        let point_record_len = format.len();
        let laz_vlr = header
            .laz_vlr()
            .with_context(|| format!("{:?}: missing LASzip VLR (not a LAZ file?)", path))?;

        // `las::Header` does not expose `offset_to_point_data`, so re-read
        // the raw header from the start of the file. This is cheap — 230
        // bytes of fixed-layout fields.
        let mut f =
            File::open(path).with_context(|| format!("reopening {:?} to read raw header", path))?;
        let raw = las::raw::Header::read_from(&mut f)
            .with_context(|| format!("{:?}: parsing raw LAS header", path))?;
        let offset_to_point_data = u64::from(raw.offset_to_point_data);

        Ok(Self {
            n_points,
            point_format_id: format_id,
            point_record_len,
            offset_to_point_data,
            scale: (t.x.scale, t.y.scale, t.z.scale),
            offset: (t.x.offset, t.y.offset, t.z.offset),
            laz_vlr,
        })
    }
}

/// Direct LAZ reader that decompresses into a tight byte buffer and parses
/// records straight into [`RawPoint`], bypassing `las::Point`.
pub(crate) struct RawLazReader {
    decompressor: ParLasZipDecompressor<BufReader<File>>,
    meta: LazFileMeta,
    index: u64,
    /// Reusable decompression scratch buffer (one batch of raw point records).
    scratch: Vec<u8>,
}

impl RawLazReader {
    /// Open the file for full decompression — every field is decoded.
    pub fn open_full(path: &Path, meta: LazFileMeta) -> Result<Self> {
        Self::open(path, meta, DecompressionSelection::all())
    }

    /// Open the file with a reduced decompression selection. Fields the
    /// selection excludes will come back as zeros in the decompressed
    /// bytes — the caller must only read fields that were selected.
    pub fn open_selective(
        path: &Path,
        meta: LazFileMeta,
        selection: DecompressionSelection,
    ) -> Result<Self> {
        Self::open(path, meta, selection)
    }

    fn open(path: &Path, meta: LazFileMeta, selection: DecompressionSelection) -> Result<Self> {
        let mut f =
            File::open(path).with_context(|| format!("opening {:?} for raw LAZ decode", path))?;
        f.seek(SeekFrom::Start(meta.offset_to_point_data))
            .with_context(|| format!("seeking to point data in {:?}", path))?;
        let buffered = BufReader::with_capacity(1 << 20, f);
        let decompressor =
            ParLasZipDecompressor::selective(buffered, meta.laz_vlr.clone(), selection)
                .with_context(|| format!("constructing ParLasZipDecompressor for {:?}", path))?;
        Ok(Self {
            decompressor,
            meta,
            index: 0,
            scratch: Vec::new(),
        })
    }

    /// How many points are left to read.
    fn points_left(&self) -> u64 {
        self.meta.n_points.saturating_sub(self.index)
    }

    /// Decompress up to `max` points' worth of raw record bytes into the
    /// internal scratch buffer. Returns the number of points actually
    /// decompressed (may be less than `max` at end-of-file).
    fn decompress_into_scratch(&mut self, max: usize) -> Result<usize> {
        let left = self.points_left() as usize;
        let n = left.min(max);
        if n == 0 {
            return Ok(0);
        }
        let rec_len = self.meta.point_record_len as usize;
        let bytes_needed = n * rec_len;
        self.scratch.resize(bytes_needed, 0);
        self.decompressor
            .decompress_many(&mut self.scratch)
            .context("laz decompress_many")?;
        self.index += n as u64;
        Ok(n)
    }

    /// Read up to `max` points into `out`, parsing every field needed to
    /// reconstruct a [`RawPoint`]. Returns the number of points read.
    ///
    /// Only valid when the reader was opened with `open_full` (or at
    /// least a selection that includes all the fields `RawPoint` stores);
    /// fields that were skipped will read back as zero.
    pub fn read_full_into(&mut self, out: &mut Vec<RawPoint>, max: usize) -> Result<usize> {
        let n = self.decompress_into_scratch(max)?;
        if n == 0 {
            return Ok(0);
        }
        let rec_len = self.meta.point_record_len as usize;
        let layout = RecordLayout::for_format(self.meta.point_format_id);
        out.reserve(n);
        let bytes = &self.scratch[..n * rec_len];
        for i in 0..n {
            let rec = &bytes[i * rec_len..(i + 1) * rec_len];
            out.push(parse_point_record(rec, layout));
        }
        Ok(n)
    }

    /// Read up to `max` points into `out` as raw `[i32; 3]` (scaled, file
    /// frame). Used by the counting pass with a selective decompressor
    /// that only emits x/y/z.
    pub fn read_xyz_into(&mut self, out: &mut Vec<[i32; 3]>, max: usize) -> Result<usize> {
        let n = self.decompress_into_scratch(max)?;
        if n == 0 {
            return Ok(0);
        }
        let rec_len = self.meta.point_record_len as usize;
        out.reserve(n);
        let bytes = &self.scratch[..n * rec_len];
        for i in 0..n {
            let base = i * rec_len;
            let x = i32::from_le_bytes(bytes[base..base + 4].try_into().unwrap());
            let y = i32::from_le_bytes(bytes[base + 4..base + 8].try_into().unwrap());
            let z = i32::from_le_bytes(bytes[base + 8..base + 12].try_into().unwrap());
            out.push([x, y, z]);
        }
        Ok(n)
    }
}

/// Offsets within a LAS point record for the fields `RawPoint` stores.
/// `None` means the format does not carry the field.
#[derive(Debug, Clone, Copy)]
struct RecordLayout {
    is_extended: bool,
    gps_time: Option<usize>,
    rgb: Option<usize>,
    nir: Option<usize>,
}

impl RecordLayout {
    const fn for_format(id: u8) -> Self {
        match id {
            // Legacy: common header ends at byte 20.
            0 => RecordLayout {
                is_extended: false,
                gps_time: None,
                rgb: None,
                nir: None,
            },
            1 => RecordLayout {
                is_extended: false,
                gps_time: Some(20),
                rgb: None,
                nir: None,
            },
            2 => RecordLayout {
                is_extended: false,
                gps_time: None,
                rgb: Some(20),
                nir: None,
            },
            3 => RecordLayout {
                is_extended: false,
                gps_time: Some(20),
                rgb: Some(28),
                nir: None,
            },
            4 => RecordLayout {
                is_extended: false,
                gps_time: Some(20),
                rgb: None,
                nir: None,
                // waveform packet follows at 28 and is ignored.
            },
            5 => RecordLayout {
                is_extended: false,
                gps_time: Some(20),
                rgb: Some(28),
                nir: None,
                // waveform packet follows at 34 and is ignored.
            },

            // Extended: common header ends at byte 30 (includes gps_time).
            6 => RecordLayout {
                is_extended: true,
                gps_time: Some(22),
                rgb: None,
                nir: None,
            },
            7 => RecordLayout {
                is_extended: true,
                gps_time: Some(22),
                rgb: Some(30),
                nir: None,
            },
            8 => RecordLayout {
                is_extended: true,
                gps_time: Some(22),
                rgb: Some(30),
                nir: Some(36),
            },
            9 => RecordLayout {
                is_extended: true,
                gps_time: Some(22),
                rgb: None,
                nir: None,
                // waveform packet follows at 30 and is ignored.
            },
            10 => RecordLayout {
                is_extended: true,
                gps_time: Some(22),
                rgb: Some(30),
                nir: Some(36),
                // waveform packet follows at 38 and is ignored.
            },
            _ => RecordLayout {
                is_extended: true,
                gps_time: None,
                rgb: None,
                nir: None,
            },
        }
    }
}

/// Parse one LAS point record into a [`RawPoint`]. Fields absent from
/// the format or stored in waveform / extra-bytes blocks are returned
/// as zero.
#[inline]
fn parse_point_record(rec: &[u8], layout: RecordLayout) -> RawPoint {
    let x = i32::from_le_bytes(rec[0..4].try_into().unwrap());
    let y = i32::from_le_bytes(rec[4..8].try_into().unwrap());
    let z = i32::from_le_bytes(rec[8..12].try_into().unwrap());
    let intensity = u16::from_le_bytes(rec[12..14].try_into().unwrap());

    let (classification, return_number, number_of_returns, user_data, scan_angle, point_source_id) =
        if layout.is_extended {
            let flags_a = rec[14];
            // flags_b (rec[15]) carries classification_flags / scanner_channel /
            // scan_direction / edge — none of which `RawPoint` stores.
            let classification = rec[16];
            let user_data = rec[17];
            let scan_angle = i16::from_le_bytes(rec[18..20].try_into().unwrap());
            let point_source_id = u16::from_le_bytes(rec[20..22].try_into().unwrap());
            (
                classification,
                flags_a & 0x0F,
                (flags_a >> 4) & 0x0F,
                user_data,
                scan_angle,
                point_source_id,
            )
        } else {
            let flags = rec[14];
            let classification = rec[15] & 0x1F; // low 5 bits; upper 3 are sync/key/withheld
            let scan_angle_rank = rec[16] as i8;
            let user_data = rec[17];
            let point_source_id = u16::from_le_bytes(rec[18..20].try_into().unwrap());
            // Scale rank (integer degrees) to the extended format's
            // 0.006° units so downstream code sees a uniform value.
            // rank * (1.0 / 0.006) rounded, matching `convert_point`.
            let scan_angle = (scan_angle_rank as f32 / 0.006).round() as i16;
            (
                classification,
                flags & 0x07,
                (flags >> 3) & 0x07,
                user_data,
                scan_angle,
                point_source_id,
            )
        };

    let gps_time = match layout.gps_time {
        Some(off) => f64::from_le_bytes(rec[off..off + 8].try_into().unwrap()),
        None => 0.0,
    };
    let (red, green, blue) = match layout.rgb {
        Some(off) => (
            u16::from_le_bytes(rec[off..off + 2].try_into().unwrap()),
            u16::from_le_bytes(rec[off + 2..off + 4].try_into().unwrap()),
            u16::from_le_bytes(rec[off + 4..off + 6].try_into().unwrap()),
        ),
        None => (0, 0, 0),
    };
    let nir = match layout.nir {
        Some(off) => u16::from_le_bytes(rec[off..off + 2].try_into().unwrap()),
        None => 0,
    };

    RawPoint {
        x,
        y,
        z,
        intensity,
        return_number,
        number_of_returns,
        classification,
        scan_angle,
        user_data,
        point_source_id,
        gps_time,
        red,
        green,
        blue,
        nir,
    }
}

/// Selection for the counting pass — only x/y/z are needed to classify
/// points into chunks. `xy_returns_channel` is the base layer (always
/// decoded); `.decompress_z()` opts into the Z layer.
pub(crate) fn xyz_only_selection() -> DecompressionSelection {
    DecompressionSelection::xy_returns_channel().decompress_z()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn test_laz() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/input.laz")
    }

    /// Every point decoded by the raw reader must match what `las::Reader`
    /// produces, field-for-field, once both are funnelled through
    /// `OctreeBuilder::convert_point`'s equivalent transform.
    #[test]
    fn raw_reader_matches_las_reader_full() {
        let path = test_laz();
        let meta = LazFileMeta::read(&path).unwrap();
        let n = meta.n_points;

        // Reference: las::Reader → convert_point-equivalent → RawPoint.
        let mut las_reader = las::Reader::from_path(&path).unwrap();
        let t = las_reader.header().transforms();
        let (sx, sy, sz) = (t.x.scale, t.y.scale, t.z.scale);
        let (ox, oy, oz) = (t.x.offset, t.y.offset, t.z.offset);
        let mut las_points: Vec<las::Point> = Vec::with_capacity(n as usize);
        las_reader.read_points_into(n, &mut las_points).unwrap();
        let reference: Vec<RawPoint> = las_points
            .iter()
            .map(|p| RawPoint {
                x: ((p.x - ox) / sx).round() as i32,
                y: ((p.y - oy) / sy).round() as i32,
                z: ((p.z - oz) / sz).round() as i32,
                intensity: p.intensity,
                return_number: p.return_number,
                number_of_returns: p.number_of_returns,
                classification: p.classification.into(),
                scan_angle: (p.scan_angle / 0.006).round() as i16,
                user_data: p.user_data,
                point_source_id: p.point_source_id,
                gps_time: p.gps_time.unwrap_or(0.0),
                red: p.color.as_ref().map(|c| c.red).unwrap_or(0),
                green: p.color.as_ref().map(|c| c.green).unwrap_or(0),
                blue: p.color.as_ref().map(|c| c.blue).unwrap_or(0),
                nir: p.nir.unwrap_or(0),
            })
            .collect();

        // Direct: RawLazReader.
        let mut raw_reader = RawLazReader::open_full(&path, meta).unwrap();
        let mut got: Vec<RawPoint> = Vec::with_capacity(n as usize);
        while raw_reader.read_full_into(&mut got, 250_000).unwrap() > 0 {}

        assert_eq!(got.len(), reference.len(), "point count");
        for (i, (a, b)) in got.iter().zip(reference.iter()).enumerate() {
            assert_eq!(a.x, b.x, "point {i} x");
            assert_eq!(a.y, b.y, "point {i} y");
            assert_eq!(a.z, b.z, "point {i} z");
            assert_eq!(a.intensity, b.intensity, "point {i} intensity");
            assert_eq!(a.return_number, b.return_number, "point {i} return_number");
            assert_eq!(
                a.number_of_returns, b.number_of_returns,
                "point {i} number_of_returns"
            );
            assert_eq!(
                a.classification, b.classification,
                "point {i} classification"
            );
            assert_eq!(a.scan_angle, b.scan_angle, "point {i} scan_angle");
            assert_eq!(a.user_data, b.user_data, "point {i} user_data");
            assert_eq!(
                a.point_source_id, b.point_source_id,
                "point {i} point_source_id"
            );
            assert_eq!(a.gps_time, b.gps_time, "point {i} gps_time");
            assert_eq!(a.red, b.red, "point {i} red");
            assert_eq!(a.green, b.green, "point {i} green");
            assert_eq!(a.blue, b.blue, "point {i} blue");
            assert_eq!(a.nir, b.nir, "point {i} nir");
        }
    }

    /// XYZ-selective decode must match the full-decode x/y/z values.
    #[test]
    fn raw_reader_xyz_matches_full_xyz() {
        let path = test_laz();
        let meta = LazFileMeta::read(&path).unwrap();
        let n = meta.n_points as usize;

        let mut full = RawLazReader::open_full(&path, meta.clone()).unwrap();
        let mut full_pts: Vec<RawPoint> = Vec::with_capacity(n);
        while full.read_full_into(&mut full_pts, 250_000).unwrap() > 0 {}

        let mut xyz = RawLazReader::open_selective(&path, meta, xyz_only_selection()).unwrap();
        let mut xyz_pts: Vec<[i32; 3]> = Vec::with_capacity(n);
        while xyz.read_xyz_into(&mut xyz_pts, 250_000).unwrap() > 0 {}

        assert_eq!(xyz_pts.len(), full_pts.len());
        for (i, (a, b)) in xyz_pts.iter().zip(full_pts.iter()).enumerate() {
            assert_eq!(a[0], b.x, "point {i} x");
            assert_eq!(a[1], b.y, "point {i} y");
            assert_eq!(a[2], b.z, "point {i} z");
        }
    }
}
