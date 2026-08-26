//! LAS 1.4 Extra Bytes VLR parsing and schema comparison.
//!
//! The Extra Bytes VLR (`user_id = "LASF_Spec"`, `record_id = 4`) declares
//! the per-point trailing-byte schema as a sequence of 192-byte
//! descriptors. Each descriptor describes one logical field (name, data
//! type, optional scale/offset for value decoding) and may also carry
//! producer-computed *stats* (min, max, no_data values).
//!
//! When merging many LAS tiles into a single COPC we have to handle a
//! reality of upstream pipelines: every tile declares the *same logical
//! schema* (same fields, same data types, same scale/offset) but each
//! tile's stats cover only its own points. Byte-level VLR comparison
//! over-rejects those inputs. This module separates the structural
//! parts (which must agree across all inputs for the per-point bytes to
//! be interpretable consistently) from the stats (which we union across
//! inputs to produce honest output VLR stats).

use byteorder::{LittleEndian, ReadBytesExt};
use std::io::Cursor;

/// Size of one Extra Bytes descriptor block.
pub(crate) const DESCRIPTOR_SIZE: usize = 192;

/// LAS Extra Bytes `options` bit positions. The spec lays these out as
/// bit-flags marking whether the corresponding stat fields are populated.
mod options_bits {
    pub const NO_DATA: u8 = 1 << 0;
    pub const MIN: u8 = 1 << 1;
    pub const MAX: u8 = 1 << 2;
    pub const SCALE: u8 = 1 << 3;
    pub const OFFSET: u8 = 1 << 4;
}

/// Human-readable name for an LAS Extra Bytes data type code.
pub(crate) fn data_type_name(data_type: u8) -> &'static str {
    match data_type {
        0 => "undocumented",
        1 => "u8",
        2 => "i8",
        3 => "u16",
        4 => "i16",
        5 => "u32",
        6 => "i32",
        7 => "u64",
        8 => "i64",
        9 => "f32",
        10 => "f64",
        _ => "?",
    }
}

/// Decode one 8-byte "anytype" stat slot (no_data/min/max) per the
/// descriptor's data type. LAS 1.4 stores integer-typed stats upcast to
/// u64/i64 and float-typed stats upcast to double — reading every slot
/// as a double (the old behaviour) turns e.g. TerraScan's i64 `-5000`
/// Reflectance min into NaN, which then poisons the min/max fold and
/// leaves the `f64::INFINITY` fold initializer to be serialized into
/// the output VLR (bigint 9218868437227405312 to a spec-following
/// reader). Returns `None` for undocumented/raw types (0, 11+), whose
/// stat slots have no defined interpretation.
pub(crate) fn anytype_to_f64(data_type: u8, raw: u64) -> Option<f64> {
    match data_type {
        1 | 3 | 5 | 7 => Some(raw as f64),
        2 | 4 | 6 | 8 => Some(raw as i64 as f64),
        9 | 10 => Some(f64::from_bits(raw)),
        _ => None,
    }
}

/// Encode a stat value back into an 8-byte anytype slot, inverse of
/// [`anytype_to_f64`]. Integer domains round and saturate; u64 values
/// beyond 2^53 lose low-bit precision on the f64 round-trip, which is
/// acceptable for advisory stats.
pub(crate) fn f64_to_anytype(data_type: u8, value: f64) -> Option<u64> {
    match data_type {
        1 | 3 | 5 | 7 => Some(value.round() as u64),
        2 | 4 | 6 | 8 => Some(value.round() as i64 as u64),
        9 | 10 => Some(value.to_bits()),
        _ => None,
    }
}

/// Structural fingerprint of one Extra Bytes field — the parts that
/// define the per-point byte layout and value decoding. Two fields with
/// identical `FieldSchema` produce identical per-point bytes for the same
/// underlying value, so files sharing all field schemas (in order) can
/// be merged. Stats (min/max/no_data) are excluded — those vary per-file.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FieldSchema {
    pub data_type: u8,
    /// Field name, with trailing NUL bytes trimmed.
    pub name: String,
    /// Field description, with trailing NUL bytes trimmed. Treated as
    /// structural because it captures producer intent and would let a
    /// reader misinterpret an otherwise-matching field if it changed
    /// silently.
    pub description: String,
    /// Scale/offset bits from the descriptor `options` byte. We mask out
    /// the no_data / min / max bits because those only flag stat
    /// presence; scale/offset bits change how readers decode values.
    pub options_structural: u8,
    /// Per-axis scale (when `options.scale_bit` is set). Three slots per
    /// spec; we keep all three so the structural hash detects partial
    /// scale changes.
    pub scale: [f64; 3],
    /// Per-axis offset.
    pub offset: [f64; 3],
    /// Per-axis no_data values. Treated as structural because changing a
    /// no_data sentinel changes the semantics of equal bit-patterns in
    /// the data — readers comparing against the descriptor's no_data
    /// would silently mis-classify points.
    pub no_data: [u64; 3],
}

/// Per-field stat ranges captured at scan time. These vary per file and
/// are merged (union of mins / maxes) when building the output VLR so
/// the COPC honestly advertises the range across all merged data.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct FieldStats {
    /// Bits 0..3 of `options` (no_data, min, max bits). If `min_bit`
    /// isn't set in a file, that file has no min to merge; same for max.
    pub options_stats: u8,
    pub min: [f64; 3],
    pub max: [f64; 3],
}

/// A parsed Extra Bytes VLR: schema (structural) + per-field stats.
#[derive(Debug, Clone)]
pub(crate) struct ParsedExtraBytes {
    pub fields: Vec<FieldSchema>,
    pub stats: Vec<FieldStats>,
}

impl ParsedExtraBytes {
    /// Parse the VLR payload. Length must be a multiple of 192 bytes
    /// (the descriptor size); anything else is malformed and we bail.
    pub fn parse(payload: &[u8]) -> anyhow::Result<Self> {
        if !payload.len().is_multiple_of(DESCRIPTOR_SIZE) {
            anyhow::bail!(
                "Extra Bytes VLR payload length {} is not a multiple of {}",
                payload.len(),
                DESCRIPTOR_SIZE
            );
        }
        let n = payload.len() / DESCRIPTOR_SIZE;
        let mut fields = Vec::with_capacity(n);
        let mut stats = Vec::with_capacity(n);
        for i in 0..n {
            let off = i * DESCRIPTOR_SIZE;
            let bytes = &payload[off..off + DESCRIPTOR_SIZE];
            let mut r = Cursor::new(bytes);

            let _reserved = r.read_u16::<LittleEndian>()?;
            let data_type = r.read_u8()?;
            let options = r.read_u8()?;

            let mut name_buf = [0u8; 32];
            std::io::Read::read_exact(&mut r, &mut name_buf)?;
            let name = trim_nul(&name_buf);

            let _unused = r.read_u32::<LittleEndian>()?;

            let mut no_data = [0u64; 3];
            for slot in &mut no_data {
                *slot = r.read_u64::<LittleEndian>()?;
            }

            // Min/max are "anytype" slots: their encoding depends on
            // data_type. Undecodable slots (undocumented types) read as
            // NaN and are masked out of the stat bits below so the merge
            // never folds or re-serializes them.
            let mut min = [0f64; 3];
            for slot in &mut min {
                let raw = r.read_u64::<LittleEndian>()?;
                *slot = anytype_to_f64(data_type, raw).unwrap_or(f64::NAN);
            }

            let mut max = [0f64; 3];
            for slot in &mut max {
                let raw = r.read_u64::<LittleEndian>()?;
                *slot = anytype_to_f64(data_type, raw).unwrap_or(f64::NAN);
            }

            let mut scale = [0f64; 3];
            for slot in &mut scale {
                *slot = r.read_f64::<LittleEndian>()?;
            }

            let mut offset = [0f64; 3];
            for slot in &mut offset {
                *slot = r.read_f64::<LittleEndian>()?;
            }

            let mut desc_buf = [0u8; 32];
            std::io::Read::read_exact(&mut r, &mut desc_buf)?;
            let description = trim_nul(&desc_buf);

            // Structural mask: keep only scale_bit + offset_bit. The
            // no_data/min/max bits flag stat presence and may legitimately
            // differ between tiles of the same logical schema.
            let options_structural = options & (options_bits::SCALE | options_bits::OFFSET);

            fields.push(FieldSchema {
                data_type,
                name,
                description,
                options_structural,
                scale,
                offset,
                no_data,
            });
            // For undocumented data types the min/max slots are
            // uninterpretable — drop their stat bits so downstream
            // merging leaves those descriptors untouched.
            let stat_mask = if anytype_to_f64(data_type, 0).is_some() {
                options_bits::NO_DATA | options_bits::MIN | options_bits::MAX
            } else {
                options_bits::NO_DATA
            };
            stats.push(FieldStats {
                options_stats: options & stat_mask,
                min,
                max,
            });
        }
        Ok(Self { fields, stats })
    }
}

fn trim_nul(buf: &[u8]) -> String {
    let s = std::str::from_utf8(buf).unwrap_or("");
    s.trim_end_matches('\0').to_owned()
}

/// Describe every structural difference between two parsed schemas.
/// Returns an empty Vec when the two are equivalent for COPC merging
/// purposes. The strings are framed as "reference (file_a) expects X,
/// file_b has Y" so the user sees both sides.
pub(crate) fn diff_schemas(reference: &ParsedExtraBytes, other: &ParsedExtraBytes) -> Vec<String> {
    let mut diffs = Vec::new();

    if reference.fields.len() != other.fields.len() {
        diffs.push(format!(
            "field count differs: reference has {} ({}), other has {} ({})",
            reference.fields.len(),
            field_name_list(&reference.fields),
            other.fields.len(),
            field_name_list(&other.fields),
        ));
        // Don't try to do field-by-field comparison when counts disagree —
        // alignment becomes ambiguous and we'd produce confusing diffs.
        return diffs;
    }

    for (i, (rf, of)) in reference.fields.iter().zip(other.fields.iter()).enumerate() {
        if rf.name != of.name {
            diffs.push(format!(
                "field {i} name mismatch: reference {:?}, other {:?}",
                rf.name, of.name
            ));
        }
        if rf.data_type != of.data_type {
            diffs.push(format!(
                "field {i} {:?} data_type mismatch: reference {} ({}), other {} ({})",
                rf.name,
                rf.data_type,
                data_type_name(rf.data_type),
                of.data_type,
                data_type_name(of.data_type),
            ));
        }
        if rf.description != of.description {
            diffs.push(format!(
                "field {i} {:?} description mismatch: reference {:?}, other {:?}",
                rf.name, rf.description, of.description
            ));
        }
        if rf.options_structural != of.options_structural {
            diffs.push(format!(
                "field {i} {:?} options (scale/offset bits) mismatch: \
                 reference 0b{:08b}, other 0b{:08b}",
                rf.name, rf.options_structural, of.options_structural
            ));
        }
        if rf.scale != of.scale {
            diffs.push(format!(
                "field {i} {:?} scale mismatch: reference {:?}, other {:?}",
                rf.name, rf.scale, of.scale
            ));
        }
        if rf.offset != of.offset {
            diffs.push(format!(
                "field {i} {:?} offset mismatch: reference {:?}, other {:?}",
                rf.name, rf.offset, of.offset
            ));
        }
        if rf.no_data != of.no_data {
            diffs.push(format!(
                "field {i} {:?} no_data mismatch: reference {:?}, other {:?}",
                rf.name, rf.no_data, of.no_data
            ));
        }
    }

    diffs
}

fn field_name_list(fields: &[FieldSchema]) -> String {
    if fields.is_empty() {
        return "no fields".to_owned();
    }
    fields
        .iter()
        .map(|f| {
            if f.name.is_empty() {
                "<unnamed>".to_owned()
            } else {
                f.name.clone()
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Stable 64-bit hash of a schema's structural fields. Built from
/// `FieldSchema` (no stats) so files with identical logical schemas but
/// different per-tile min/max hash to the same value.
pub(crate) fn schema_hash(parsed: &ParsedExtraBytes) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{BuildHasher, BuildHasherDefault, Hash, Hasher};
    let mut h = BuildHasherDefault::<DefaultHasher>::default().build_hasher();
    for f in &parsed.fields {
        f.data_type.hash(&mut h);
        f.name.hash(&mut h);
        f.description.hash(&mut h);
        f.options_structural.hash(&mut h);
        for v in &f.scale {
            h.write_u64(v.to_bits());
        }
        for v in &f.offset {
            h.write_u64(v.to_bits());
        }
        for v in &f.no_data {
            h.write_u64(*v);
        }
    }
    h.finish()
}

/// Take the canonical VLR bytes (the first file's payload, preserved
/// verbatim) and patch the min/max fields with the union of stats
/// across all input files. The structural bytes are untouched; only the
/// per-axis min[0..3] and max[0..3] f64 slots inside each descriptor are
/// rewritten. The options byte for each descriptor is OR-ed so the
/// `min_bit` / `max_bit` reflect whether *any* input had that stat.
pub(crate) fn merge_stats_into_canonical(
    canonical: &mut [u8],
    per_file_parsed: &[ParsedExtraBytes],
) -> anyhow::Result<()> {
    if per_file_parsed.is_empty() {
        return Ok(());
    }
    let n_fields = canonical.len() / DESCRIPTOR_SIZE;
    if !canonical.len().is_multiple_of(DESCRIPTOR_SIZE) {
        anyhow::bail!(
            "canonical VLR length {} not a multiple of descriptor size",
            canonical.len()
        );
    }

    for field_idx in 0..n_fields {
        // Collect all per-file stats for this field. Each file's parsed
        // payload has `fields.len() == n_fields` (validated upstream).
        let mut any_no_data = false;
        let mut merged_min = [f64::INFINITY; 3];
        let mut merged_max = [f64::NEG_INFINITY; 3];
        // Whether a finite value was folded into each axis. Non-finite
        // stats (NaN slots from garbage float stats) must neither poison
        // the fold via a false comparison nor let the fold initializers
        // leak into the output — an axis nothing folded into keeps its
        // stat bit clear.
        let mut min_folded = [false; 3];
        let mut max_folded = [false; 3];

        for parsed in per_file_parsed {
            if parsed.fields.len() != n_fields {
                anyhow::bail!(
                    "stat merge expected {} fields per file but one has {}",
                    n_fields,
                    parsed.fields.len()
                );
            }
            let stats = parsed.stats[field_idx];
            if stats.options_stats & options_bits::NO_DATA != 0 {
                any_no_data = true;
            }
            if stats.options_stats & options_bits::MIN != 0 {
                for ((merged, folded), &incoming) in merged_min
                    .iter_mut()
                    .zip(&mut min_folded)
                    .zip(stats.min.iter())
                {
                    if incoming.is_finite() {
                        *folded = true;
                        if incoming < *merged {
                            *merged = incoming;
                        }
                    }
                }
            }
            if stats.options_stats & options_bits::MAX != 0 {
                for ((merged, folded), &incoming) in merged_max
                    .iter_mut()
                    .zip(&mut max_folded)
                    .zip(stats.max.iter())
                {
                    if incoming.is_finite() {
                        *folded = true;
                        if incoming > *merged {
                            *merged = incoming;
                        }
                    }
                }
            }
        }
        // Gate the stat bits on axis 0, the meaningful axis (multi-axis
        // extra bytes are spec-deprecated; axes 1-2 are zero padding in
        // practice, and zeros always fold as finite). A field whose
        // axis-0 stat never folded has no trustworthy range to advertise.
        let any_min = min_folded[0];
        let any_max = max_folded[0];

        let off = field_idx * DESCRIPTOR_SIZE;
        let descriptor = &mut canonical[off..off + DESCRIPTOR_SIZE];
        let data_type = descriptor[2];

        // Patch the options byte: keep the structural bits (scale_bit,
        // offset_bit) from the canonical VLR, OR in the union of stat
        // bits across all inputs.
        let existing_options = descriptor[3];
        let new_options = (existing_options & (options_bits::SCALE | options_bits::OFFSET))
            | (if any_no_data {
                options_bits::NO_DATA
            } else {
                0
            })
            | (if any_min { options_bits::MIN } else { 0 })
            | (if any_max { options_bits::MAX } else { 0 });
        descriptor[3] = new_options;

        // Patch min / max if any input had them, re-encoding per the
        // field's data type (integers as 64-bit ints, floats as doubles
        // — the LAS 1.4 anytype rules). Axes nothing finite folded into
        // are zeroed. no_data values are left alone — they're
        // producer-defined sentinels, not data-derived.
        if any_min {
            let mut min_off = 4 + 32 + 4 + 24; // skip header + name + unused + no_data
            for (v, folded) in merged_min.iter().zip(min_folded) {
                let raw = if folded {
                    f64_to_anytype(data_type, *v).unwrap_or(0)
                } else {
                    0
                };
                descriptor[min_off..min_off + 8].copy_from_slice(&raw.to_le_bytes());
                min_off += 8;
            }
        }
        if any_max {
            let mut max_off = 4 + 32 + 4 + 24 + 24; // ... + min[3]
            for (v, folded) in merged_max.iter().zip(max_folded) {
                let raw = if folded {
                    f64_to_anytype(data_type, *v).unwrap_or(0)
                } else {
                    0
                };
                descriptor[max_off..max_off + 8].copy_from_slice(&raw.to_le_bytes());
                max_off += 8;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a single-field VLR payload with the given parameters.
    /// Useful for synthesising small fixtures without needing real LAS files.
    #[allow(clippy::too_many_arguments)]
    fn build_single_field(
        data_type: u8,
        name: &str,
        description: &str,
        options: u8,
        min: [f64; 3],
        max: [f64; 3],
        scale: [f64; 3],
        offset: [f64; 3],
        no_data: [u64; 3],
    ) -> Vec<u8> {
        let mut buf = vec![0u8; DESCRIPTOR_SIZE];
        // u16 reserved (2) at bytes 0..2 — leave zero.
        buf[2] = data_type;
        buf[3] = options;
        let name_bytes = name.as_bytes();
        let n = name_bytes.len().min(32);
        buf[4..4 + n].copy_from_slice(&name_bytes[..n]);
        // bytes 36..40 unused.
        let mut off = 40;
        for v in no_data {
            buf[off..off + 8].copy_from_slice(&v.to_le_bytes());
            off += 8;
        }
        for v in min {
            buf[off..off + 8].copy_from_slice(&v.to_le_bytes());
            off += 8;
        }
        for v in max {
            buf[off..off + 8].copy_from_slice(&v.to_le_bytes());
            off += 8;
        }
        for v in scale {
            buf[off..off + 8].copy_from_slice(&v.to_le_bytes());
            off += 8;
        }
        for v in offset {
            buf[off..off + 8].copy_from_slice(&v.to_le_bytes());
            off += 8;
        }
        let desc_bytes = description.as_bytes();
        let dn = desc_bytes.len().min(32);
        buf[160..160 + dn].copy_from_slice(&desc_bytes[..dn]);
        buf
    }

    #[test]
    fn parse_single_field_round_trip() {
        let payload = build_single_field(
            10, // f64
            "zNorm",
            "height above ground",
            options_bits::MIN | options_bits::MAX,
            [-4.5, 0.0, 0.0],
            [24.7, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0, 0, 0],
        );
        let parsed = ParsedExtraBytes::parse(&payload).unwrap();
        assert_eq!(parsed.fields.len(), 1);
        assert_eq!(parsed.fields[0].data_type, 10);
        assert_eq!(parsed.fields[0].name, "zNorm");
        assert_eq!(parsed.fields[0].description, "height above ground");
        assert_eq!(parsed.fields[0].options_structural, 0); // no scale/offset bits set
        assert_eq!(parsed.stats[0].min[0], -4.5);
        assert_eq!(parsed.stats[0].max[0], 24.7);
    }

    #[test]
    fn schema_hash_ignores_stats() {
        // Same schema, different min/max stats — hashes must match.
        let a = build_single_field(
            10,
            "zNorm",
            "",
            options_bits::MIN | options_bits::MAX,
            [-4.5, 0.0, 0.0],
            [24.7, 0.0, 0.0],
            [0.0; 3],
            [0.0; 3],
            [0; 3],
        );
        let b = build_single_field(
            10,
            "zNorm",
            "",
            options_bits::MIN | options_bits::MAX,
            [-99.0, 0.0, 0.0], // wildly different min
            [200.0, 0.0, 0.0], // wildly different max
            [0.0; 3],
            [0.0; 3],
            [0; 3],
        );
        let pa = ParsedExtraBytes::parse(&a).unwrap();
        let pb = ParsedExtraBytes::parse(&b).unwrap();
        assert_eq!(schema_hash(&pa), schema_hash(&pb));
        assert!(diff_schemas(&pa, &pb).is_empty());
    }

    #[test]
    fn diff_reports_data_type_mismatch() {
        let a = build_single_field(
            3, "treeId", "", 0, [0.0; 3], [0.0; 3], [0.0; 3], [0.0; 3], [0; 3],
        );
        let b = build_single_field(
            5, "treeId", "", 0, [0.0; 3], [0.0; 3], [0.0; 3], [0.0; 3], [0; 3],
        );
        let pa = ParsedExtraBytes::parse(&a).unwrap();
        let pb = ParsedExtraBytes::parse(&b).unwrap();
        let diffs = diff_schemas(&pa, &pb);
        assert_eq!(diffs.len(), 1);
        assert!(diffs[0].contains("data_type mismatch"));
        assert!(diffs[0].contains("3 (u16)"));
        assert!(diffs[0].contains("5 (u32)"));
    }

    #[test]
    fn diff_reports_field_count_mismatch() {
        let single = build_single_field(
            10, "zNorm", "", 0, [0.0; 3], [0.0; 3], [0.0; 3], [0.0; 3], [0; 3],
        );
        let mut double = single.clone();
        double.extend_from_slice(&build_single_field(
            1, "semantic", "", 0, [0.0; 3], [0.0; 3], [0.0; 3], [0.0; 3], [0; 3],
        ));
        let pa = ParsedExtraBytes::parse(&single).unwrap();
        let pb = ParsedExtraBytes::parse(&double).unwrap();
        let diffs = diff_schemas(&pa, &pb);
        assert_eq!(diffs.len(), 1);
        assert!(diffs[0].contains("field count differs"));
        assert!(diffs[0].contains("zNorm"));
        assert!(diffs[0].contains("semantic"));
    }

    #[test]
    fn diff_reports_all_field_differences() {
        // Two-field schemas with mismatches in both fields.
        let mut a = build_single_field(
            3, "treeId", "id", 0, [0.0; 3], [0.0; 3], [0.0; 3], [0.0; 3], [0; 3],
        );
        a.extend(build_single_field(
            10, "zNorm", "z", 0, [0.0; 3], [0.0; 3], [0.0; 3], [0.0; 3], [0; 3],
        ));
        let mut b = build_single_field(
            5, "treeId", "id", 0, [0.0; 3], [0.0; 3], [0.0; 3], [0.0; 3], [0; 3],
        );
        b.extend(build_single_field(
            10, "zNorm", "z-norm", 0, [0.0; 3], [0.0; 3], [0.0; 3], [0.0; 3], [0; 3],
        ));
        let pa = ParsedExtraBytes::parse(&a).unwrap();
        let pb = ParsedExtraBytes::parse(&b).unwrap();
        let diffs = diff_schemas(&pa, &pb);
        // Should report both: field 0 data_type, field 1 description.
        assert!(
            diffs
                .iter()
                .any(|d| d.contains("field 0") && d.contains("data_type"))
        );
        assert!(
            diffs
                .iter()
                .any(|d| d.contains("field 1") && d.contains("description"))
        );
    }

    #[test]
    fn merge_stats_takes_union() {
        // Two files with same schema, different per-file mins/maxes.
        let a = build_single_field(
            10,
            "zNorm",
            "",
            options_bits::MIN | options_bits::MAX,
            [-4.5, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0; 3],
            [0.0; 3],
            [0; 3],
        );
        let b = build_single_field(
            10,
            "zNorm",
            "",
            options_bits::MIN | options_bits::MAX,
            [-2.0, 0.0, 0.0],
            [24.7, 0.0, 0.0],
            [0.0; 3],
            [0.0; 3],
            [0; 3],
        );
        let pa = ParsedExtraBytes::parse(&a).unwrap();
        let pb = ParsedExtraBytes::parse(&b).unwrap();

        let mut canonical = a.clone();
        merge_stats_into_canonical(&mut canonical, &[pa, pb]).unwrap();

        let merged = ParsedExtraBytes::parse(&canonical).unwrap();
        // Min should be -4.5 (from a), max 24.7 (from b).
        assert_eq!(merged.stats[0].min[0], -4.5);
        assert_eq!(merged.stats[0].max[0], 24.7);
        // Options byte must retain MIN|MAX bits.
        assert_eq!(
            merged.stats[0].options_stats & (options_bits::MIN | options_bits::MAX),
            options_bits::MIN | options_bits::MAX
        );
    }

    #[test]
    fn merge_handles_mixed_stat_presence() {
        // One file declares min/max, the other doesn't. Output should
        // still carry the stats from the file that had them.
        let with_stats = build_single_field(
            10,
            "zNorm",
            "",
            options_bits::MIN | options_bits::MAX,
            [-4.5, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0; 3],
            [0.0; 3],
            [0; 3],
        );
        let without_stats = build_single_field(
            10, "zNorm", "", 0, // no min/max bits set
            [0.0; 3], [0.0; 3], [0.0; 3], [0.0; 3], [0; 3],
        );
        let p1 = ParsedExtraBytes::parse(&with_stats).unwrap();
        let p2 = ParsedExtraBytes::parse(&without_stats).unwrap();

        let mut canonical = without_stats.clone();
        merge_stats_into_canonical(&mut canonical, &[p1, p2]).unwrap();

        let merged = ParsedExtraBytes::parse(&canonical).unwrap();
        assert_eq!(merged.stats[0].min[0], -4.5);
        assert_eq!(merged.stats[0].max[0], 10.0);
        assert_eq!(
            merged.stats[0].options_stats & (options_bits::MIN | options_bits::MAX),
            options_bits::MIN | options_bits::MAX
        );
    }

    /// Overwrite the first min/max anytype slots with raw little-endian
    /// bytes, for building integer-encoded fixtures (`build_single_field`
    /// writes f64 bytes, which is only valid for float-typed fields).
    fn set_raw_stat_slots(payload: &mut [u8], min_raw: u64, max_raw: u64) {
        payload[64..72].copy_from_slice(&min_raw.to_le_bytes());
        payload[88..96].copy_from_slice(&max_raw.to_le_bytes());
    }

    #[test]
    fn parses_integer_typed_stats_per_spec() {
        // TerraScan-style i16 field: min stored as i64 -5000, max as 15000.
        // The old f64-everywhere parse decoded that min as NaN.
        let mut payload = build_single_field(
            4, // i16
            "Reflectance",
            "",
            options_bits::MIN | options_bits::MAX,
            [0.0; 3],
            [0.0; 3],
            [0.0; 3],
            [0.0; 3],
            [0; 3],
        );
        set_raw_stat_slots(&mut payload, (-5000i64) as u64, 15000);
        let parsed = ParsedExtraBytes::parse(&payload).unwrap();
        assert_eq!(parsed.stats[0].min[0], -5000.0);
        assert_eq!(parsed.stats[0].max[0], 15000.0);
    }

    #[test]
    fn merge_preserves_integer_stats() {
        // Regression: NaN-decoded integer mins used to poison the fold,
        // serializing the f64::INFINITY initializer (bit pattern
        // 0x7FF0000000000000) into the output — which spec-following
        // readers surface as bigint 9218868437227405312.
        let build = |min: i64, max: i64| {
            let mut p = build_single_field(
                4, // i16
                "Reflectance",
                "",
                options_bits::MIN | options_bits::MAX,
                [0.0; 3],
                [0.0; 3],
                [0.0; 3],
                [0.0; 3],
                [0; 3],
            );
            set_raw_stat_slots(&mut p, min as u64, max as u64);
            p
        };
        let a = build(-5000, 15000);
        let b = build(-4000, 15500);
        let pa = ParsedExtraBytes::parse(&a).unwrap();
        let pb = ParsedExtraBytes::parse(&b).unwrap();

        let mut canonical = a.clone();
        merge_stats_into_canonical(&mut canonical, &[pa, pb]).unwrap();

        let min_raw = u64::from_le_bytes(canonical[64..72].try_into().unwrap());
        let max_raw = u64::from_le_bytes(canonical[88..96].try_into().unwrap());
        assert_ne!(
            min_raw,
            f64::INFINITY.to_bits(),
            "must not leak the fold initializer"
        );
        assert_eq!(min_raw as i64, -5000);
        assert_eq!(max_raw as i64, 15500);
    }

    #[test]
    fn merge_skips_non_finite_float_stats() {
        let garbage = build_single_field(
            10,
            "zNorm",
            "",
            options_bits::MIN | options_bits::MAX,
            [f64::NAN, 0.0, 0.0],
            [f64::INFINITY, 0.0, 0.0],
            [0.0; 3],
            [0.0; 3],
            [0; 3],
        );
        let good = build_single_field(
            10,
            "zNorm",
            "",
            options_bits::MIN | options_bits::MAX,
            [-2.0, 0.0, 0.0],
            [7.5, 0.0, 0.0],
            [0.0; 3],
            [0.0; 3],
            [0; 3],
        );
        let pg = ParsedExtraBytes::parse(&garbage).unwrap();
        let pk = ParsedExtraBytes::parse(&good).unwrap();

        // Garbage + good: the finite stats win.
        let mut canonical = garbage.clone();
        merge_stats_into_canonical(&mut canonical, &[pg.clone(), pk]).unwrap();
        let merged = ParsedExtraBytes::parse(&canonical).unwrap();
        assert_eq!(merged.stats[0].min[0], -2.0);
        assert_eq!(merged.stats[0].max[0], 7.5);

        // Garbage only: no finite value ever folds, so the stat bits
        // must come out cleared rather than advertising the initializers.
        let mut canonical = garbage.clone();
        merge_stats_into_canonical(&mut canonical, &[pg]).unwrap();
        let merged = ParsedExtraBytes::parse(&canonical).unwrap();
        assert_eq!(
            merged.stats[0].options_stats & (options_bits::MIN | options_bits::MAX),
            0
        );
    }

    #[test]
    fn structural_options_changes_hash() {
        // Two files with same fields but different scale_bit setting
        // (one has scale, the other doesn't) — value decoding differs.
        let a = build_single_field(
            10,
            "zNorm",
            "",
            options_bits::SCALE,
            [0.0; 3],
            [0.0; 3],
            [0.001, 0.0, 0.0],
            [0.0; 3],
            [0; 3],
        );
        let b = build_single_field(
            10, "zNorm", "", 0, [0.0; 3], [0.0; 3], [0.0; 3], [0.0; 3], [0; 3],
        );
        let pa = ParsedExtraBytes::parse(&a).unwrap();
        let pb = ParsedExtraBytes::parse(&b).unwrap();
        assert_ne!(schema_hash(&pa), schema_hash(&pb));
        let diffs = diff_schemas(&pa, &pb);
        assert!(
            diffs
                .iter()
                .any(|d| d.contains("options") || d.contains("scale")),
            "expected diff about options/scale, got {diffs:?}"
        );
    }
}
