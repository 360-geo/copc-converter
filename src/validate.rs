/// Validate consistency of scanned input files before building the octree.
use crate::Error;
use crate::extra_bytes::{ParsedExtraBytes, diff_schemas, merge_stats_into_canonical};
use crate::octree::{CrsKind, ScanResult, get_epsg_from_wkt_crs_bytes, input_to_copc_format};
use std::path::PathBuf;
use tracing::{debug, warn};

/// Validated output: consistent properties across all input files.
#[derive(Debug)]
pub struct ValidatedInputs {
    /// WKT CRS string for the output COPC (if available). Either the
    /// canonical WKT pulled from the first input that had one, or — if
    /// the inputs only carry GeoTIFF CRS info — a WKT synthesized via
    /// the `crs-definitions` registry from the EPSG code.
    pub wkt_crs: Option<Vec<u8>>,
    /// LAS Extra Bytes VLR payload (if present). Single canonical copy
    /// from the first scanned file, with per-field min/max stats merged
    /// across every input so the output COPC honestly advertises the
    /// full range of merged data. `None` when no input declares extras.
    pub extra_bytes_vlr: Option<Vec<u8>>,
    /// Trailing extra-byte width per point record. Zero when no extras
    /// are declared. Validated to be uniform across all inputs.
    pub num_extra_bytes: u16,
    /// COPC output point format (6, 7, or 8).
    pub point_format: u8,
}

/// Returns true if the LAS point format includes GPS time.
fn format_has_gps_time(fmt: u8) -> bool {
    // LAS formats 0 and 2 lack GPS time; all others (1, 3–10) include it.
    !matches!(fmt, 0 | 2)
}

/// Compare two `CrsKind` values, using `canonical_wkt` only as a
/// fallback when the kinds are mixed (one WKT-hash, one GeoTIFF-EPSG).
///
/// Returns `(equal, vertical_dropped)`. `vertical_dropped` is true when
/// horizontal codes match but vertical components differ — the caller
/// decides whether to accept this or treat it as a hard mismatch.
fn crs_kinds_compatible(a: CrsKind, b: CrsKind, canonical_wkt: Option<&[u8]>) -> (bool, bool) {
    match (a, b) {
        (CrsKind::WktHash(ha), CrsKind::WktHash(hb)) => (ha == hb, false),
        (CrsKind::GeoTiffEpsg(ah, av), CrsKind::GeoTiffEpsg(bh, bv)) => {
            if ah == bh && av == bv {
                (true, false)
            } else if ah == bh {
                (true, true)
            } else {
                (false, false)
            }
        }
        (CrsKind::WktHash(_), CrsKind::GeoTiffEpsg(gh, gv))
        | (CrsKind::GeoTiffEpsg(gh, gv), CrsKind::WktHash(_)) => {
            // Cross-format: parse the canonical WKT once to get its
            // trailing EPSG code(s) and compare to the GeoTIFF tuple.
            let Some(wkt) = canonical_wkt else {
                return (false, false);
            };
            let Some((wh, wv)) = get_epsg_from_wkt_crs_bytes(wkt) else {
                return (false, false);
            };
            if wh == gh && wv == gv {
                (true, false)
            } else if wh == gh {
                (true, true)
            } else {
                (false, false)
            }
        }
    }
}

/// Check that all scanned files agree on CRS and point format,
/// and derive the COPC output point format.
pub fn validate(
    input_files: &[PathBuf],
    results: &[ScanResult],
    canonical_wkt: Option<Vec<u8>>,
    canonical_extra_bytes_vlr: Option<Vec<u8>>,
    temporal_index: Option<u32>,
) -> crate::Result<ValidatedInputs> {
    // If no input file carried a WKT CRS but at least one carried a
    // GeoTIFF EPSG, synthesize a canonical WKT from the registry. The
    // vertical component (if any) is discarded since `crs-definitions`
    // only covers horizontal codes.
    let canonical_wkt = match canonical_wkt {
        Some(w) => Some(w),
        None => results
            .iter()
            .find_map(|r| match r.crs {
                Some(CrsKind::GeoTiffEpsg(h, v)) => Some((h, v)),
                _ => None,
            })
            .and_then(|(horizontal, vertical)| {
                if vertical.is_some() {
                    warn!(
                        "Translating GeoTIFF CRS (EPSG:{horizontal}) to WKT for COPC output; \
                         vertical CRS will be dropped"
                    );
                }
                match crs_definitions::from_code(horizontal) {
                    Some(def) => Some(def.wkt.as_bytes().to_vec()),
                    None => {
                        warn!(
                            "GeoTIFF EPSG:{horizontal} not found in crs-definitions registry; \
                             output COPC will have no CRS"
                        );
                        None
                    }
                }
            }),
    };

    // Reference index for CRS errors: first file that has any CRS.
    // Naming this file (rather than file 0) in `CrsMismatch` errors
    // gives clearer attribution when the offending file is later in
    // the input order.
    let crs_file_index = results.iter().position(|r| r.crs.is_some()).unwrap_or(0);
    let reference_crs = results[crs_file_index].crs;

    let first_format = results[0].point_format_id;

    // First pass: CRS + point format consistency.
    for (i, r) in results.iter().enumerate() {
        if i != crs_file_index {
            let (equal, dropped_vertical) = match (reference_crs, r.crs) {
                (None, None) => (true, false),
                (Some(a), Some(b)) => crs_kinds_compatible(a, b, canonical_wkt.as_deref()),
                _ => (false, false),
            };
            if !equal {
                return Err(Error::CrsMismatch {
                    file_a: input_files[crs_file_index].clone(),
                    file_b: input_files[i].clone(),
                });
            }
            if dropped_vertical {
                warn!(
                    "Vertical CRS component differs between {:?} and {:?}; ignoring it",
                    input_files[crs_file_index], input_files[i]
                );
            }
        }
        if r.point_format_id != first_format {
            return Err(Error::PointFormatMismatch {
                file_a: input_files[0].clone(),
                format_a: first_format,
                file_b: input_files[i].clone(),
                format_b: r.point_format_id,
            });
        }
    }

    // Reference for Extra Bytes diagnostics: first file with extras.
    let reference_extras_idx = results
        .iter()
        .position(|r| r.extra_bytes_parsed.is_some())
        .unwrap_or(0);
    let reference_extras_hash = results[reference_extras_idx].extra_bytes_schema_hash;
    let reference_num_extras = results[reference_extras_idx].num_extra_bytes;
    let reference_parsed = results[reference_extras_idx].extra_bytes_parsed.as_ref();

    // Second pass: Extra Bytes schema consistency.
    for (i, r) in results.iter().enumerate() {
        if i == reference_extras_idx {
            continue;
        }
        if r.extra_bytes_schema_hash == reference_extras_hash
            && r.num_extra_bytes == reference_num_extras
        {
            continue;
        }
        return Err(extras_mismatch_error(
            input_files,
            reference_extras_idx,
            reference_parsed,
            reference_num_extras,
            i,
            &results[i],
        ));
    }

    if let Some(temporal_stride) = temporal_index {
        if temporal_stride == 0 {
            return Err(Error::InvalidTemporalStride {
                stride: temporal_stride,
            });
        }

        if !format_has_gps_time(first_format) {
            return Err(Error::NoGpsTime {
                format: first_format,
            });
        }
    }

    // Stat-merge: take the canonical VLR bytes (first file's payload)
    // and patch the per-field min/max stats with the union across all
    // inputs. Producers compute per-file stats from each tile's own
    // points; without this, the output COPC would advertise stats from
    // an arbitrary single tile.
    let extra_bytes_vlr = match (canonical_extra_bytes_vlr, reference_parsed) {
        (Some(mut bytes), Some(_)) => {
            let per_file_parsed: Vec<ParsedExtraBytes> = results
                .iter()
                .filter_map(|r| r.extra_bytes_parsed.clone())
                .collect();
            if let Err(e) = merge_stats_into_canonical(&mut bytes, &per_file_parsed) {
                warn!("Could not merge Extra Bytes stats, using reference file's stats as-is: {e}");
            }
            Some(bytes)
        }
        (vlr, _) => vlr,
    };

    let point_format = input_to_copc_format(first_format);
    debug!(
        "Input point format: {first_format}, output COPC point format: {point_format}, \
         extra bytes per point: {reference_num_extras}"
    );

    Ok(ValidatedInputs {
        wkt_crs: canonical_wkt,
        extra_bytes_vlr,
        num_extra_bytes: reference_num_extras,
        point_format,
    })
}

/// Build an `Error::ExtraBytesMismatch` with a multi-line `detail`
/// string covering every structural difference between the reference
/// file and the offending file.
fn extras_mismatch_error(
    input_files: &[PathBuf],
    reference_idx: usize,
    reference_parsed: Option<&ParsedExtraBytes>,
    reference_num_extras: u16,
    other_idx: usize,
    other: &ScanResult,
) -> Error {
    let mut detail_lines: Vec<String> = Vec::new();

    if other.num_extra_bytes != reference_num_extras {
        detail_lines.push(format!(
            "  - point trailing-byte width differs: reference declares {} bytes, other declares {}",
            reference_num_extras, other.num_extra_bytes
        ));
    }

    match (reference_parsed, other.extra_bytes_parsed.as_ref()) {
        (Some(reference), Some(other_parsed)) => {
            for diff in diff_schemas(reference, other_parsed) {
                detail_lines.push(format!("  - {diff}"));
            }
        }
        (Some(_), None) => {
            detail_lines
                .push("  - reference declares an Extra Bytes VLR, other declares none".to_owned());
        }
        (None, Some(_)) => {
            detail_lines
                .push("  - other declares an Extra Bytes VLR, reference declares none".to_owned());
        }
        (None, None) => {}
    }

    let detail = if detail_lines.is_empty() {
        "  - (no structural differences detected; schema hashes diverged for an unknown reason)"
            .to_owned()
    } else {
        detail_lines.join("\n")
    };

    Error::ExtraBytesMismatch {
        file_a: input_files[reference_idx].clone(),
        file_b: input_files[other_idx].clone(),
        detail,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extra_bytes::ParsedExtraBytes;
    use crate::octree::{Bounds, ScanResult, bytes_hash};

    const WGS84_WKT: &[u8] = br#"GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]"#;
    const EPSG3006_WKT: &[u8] = br#"PROJCRS["SWEREF99 TM",BASEGEOGCRS["SWEREF99",DATUM["SWEREF99",ELLIPSOID["GRS 1980",6378137,298.257222101,LENGTHUNIT["metre",1]]],PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],ID["EPSG",4619]],CONVERSION["SWEREF99 TM",METHOD["Transverse Mercator",ID["EPSG",9807]],PARAMETER["Latitude of natural origin",0,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8801]],PARAMETER["Longitude of natural origin",15,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8802]],PARAMETER["Scale factor at natural origin",0.9996,SCALEUNIT["unity",1],ID["EPSG",8805]],PARAMETER["False easting",500000,LENGTHUNIT["metre",1],ID["EPSG",8806]],PARAMETER["False northing",0,LENGTHUNIT["metre",1],ID["EPSG",8807]]],CS[Cartesian,2],AXIS["northing (N)",north,ORDER[1],LENGTHUNIT["metre",1]],AXIS["easting (E)",east,ORDER[2],LENGTHUNIT["metre",1]],USAGE[SCOPE["Topographic mapping (medium and small scale)."],AREA["Sweden - onshore and offshore."],BBOX[54.96,10.03,69.07,24.17]],ID["EPSG",3006]]"#;

    /// Build a minimal one-field Extra Bytes VLR payload with the given
    /// data type and per-axis min/max. Mirrors the helper in
    /// `extra_bytes::tests`, kept here so the validate tests can
    /// construct synthetic ScanResults without touching real LAS files.
    fn build_one_field_vlr(
        data_type: u8,
        name: &str,
        min0: f64,
        max0: f64,
    ) -> (Vec<u8>, ParsedExtraBytes) {
        let mut buf = vec![0u8; 192];
        buf[2] = data_type;
        buf[3] = 0b0000_0110; // min_bit + max_bit
        let nm = name.as_bytes();
        buf[4..4 + nm.len()].copy_from_slice(nm);
        buf[64..72].copy_from_slice(&min0.to_le_bytes()); // min
        buf[88..96].copy_from_slice(&max0.to_le_bytes()); // max
        let parsed = ParsedExtraBytes::parse(&buf).unwrap();
        (buf, parsed)
    }

    fn make_result(crs: Option<CrsKind>, fmt: u8) -> ScanResult {
        ScanResult {
            bounds: Bounds::empty(),
            point_count: 100,
            scale_x: 0.001,
            scale_y: 0.001,
            scale_z: 0.001,
            offset_x: 0.0,
            offset_y: 0.0,
            offset_z: 0.0,
            crs,
            extra_bytes_parsed: None,
            extra_bytes_schema_hash: None,
            num_extra_bytes: 0,
            point_format_id: fmt,
        }
    }

    fn make_result_with_extras(
        crs: Option<CrsKind>,
        parsed: ParsedExtraBytes,
        num_extra: u16,
        fmt: u8,
    ) -> ScanResult {
        let schema_hash = crate::extra_bytes::schema_hash(&parsed);
        ScanResult {
            bounds: Bounds::empty(),
            point_count: 100,
            scale_x: 0.001,
            scale_y: 0.001,
            scale_z: 0.001,
            offset_x: 0.0,
            offset_y: 0.0,
            offset_z: 0.0,
            crs,
            extra_bytes_parsed: Some(parsed),
            extra_bytes_schema_hash: Some(schema_hash),
            num_extra_bytes: num_extra,
            point_format_id: fmt,
        }
    }

    #[test]
    fn validate_single_file() {
        let files = vec![PathBuf::from("a.laz")];
        let results = vec![make_result(None, 3)];
        let v = validate(&files, &results, None, None, None).unwrap();
        assert_eq!(v.point_format, 7);
        assert!(v.wkt_crs.is_none());
        assert!(v.extra_bytes_vlr.is_none());
        assert_eq!(v.num_extra_bytes, 0);
    }

    #[test]
    fn validate_matching_files() {
        let files = vec![PathBuf::from("a.laz"), PathBuf::from("b.laz")];
        let h = bytes_hash(WGS84_WKT);
        let results = vec![
            make_result(Some(CrsKind::WktHash(h)), 8),
            make_result(Some(CrsKind::WktHash(h)), 8),
        ];
        let v = validate(&files, &results, Some(WGS84_WKT.to_vec()), None, None).unwrap();
        assert_eq!(v.point_format, 8);
        assert_eq!(v.wkt_crs.as_deref(), Some(WGS84_WKT));
    }

    #[test]
    fn validate_crs_wkt_and_geotiff() {
        let files = vec![PathBuf::from("a.laz"), PathBuf::from("b.laz")];
        let results = vec![
            make_result(Some(CrsKind::WktHash(bytes_hash(EPSG3006_WKT))), 7),
            make_result(Some(CrsKind::GeoTiffEpsg(3006, None)), 7),
        ];
        let v = validate(&files, &results, Some(EPSG3006_WKT.to_vec()), None, None).unwrap();
        assert_eq!(v.wkt_crs.as_deref(), Some(EPSG3006_WKT));
    }

    #[test]
    fn validate_crs_geotiff_and_wkt() {
        let files = vec![PathBuf::from("a.laz"), PathBuf::from("b.laz")];
        let results = vec![
            make_result(Some(CrsKind::GeoTiffEpsg(3006, None)), 7),
            make_result(Some(CrsKind::WktHash(bytes_hash(EPSG3006_WKT))), 7),
        ];
        // canonical_wkt comes from the WKT file (file 1), even though the
        // GeoTIFF file is listed first.
        let v = validate(&files, &results, Some(EPSG3006_WKT.to_vec()), None, None).unwrap();
        assert_eq!(v.wkt_crs.as_deref(), Some(EPSG3006_WKT));
    }

    #[test]
    fn validate_crs_geotiff_only_translates_via_registry() {
        let files = vec![PathBuf::from("a.laz"), PathBuf::from("b.laz")];
        let results = vec![
            make_result(Some(CrsKind::GeoTiffEpsg(3006, None)), 7),
            make_result(Some(CrsKind::GeoTiffEpsg(3006, None)), 7),
        ];
        let v = validate(&files, &results, None, None, None).unwrap();
        // The registry returns *some* WKT for EPSG:3006; we don't pin
        // the exact bytes (depends on crs-definitions version).
        assert!(v.wkt_crs.is_some());
        let wkt = v.wkt_crs.unwrap();
        assert!(wkt.windows(4).any(|w| w == b"3006"));
    }

    #[test]
    fn validate_crs_mismatch() {
        let files = vec![PathBuf::from("a.laz"), PathBuf::from("b.laz")];
        let results = vec![
            make_result(Some(CrsKind::WktHash(bytes_hash(WGS84_WKT))), 7),
            make_result(Some(CrsKind::WktHash(bytes_hash(EPSG3006_WKT))), 7),
        ];
        let err = validate(&files, &results, Some(WGS84_WKT.to_vec()), None, None).unwrap_err();
        assert!(matches!(err, Error::CrsMismatch { .. }));
    }

    #[test]
    fn validate_format_mismatch() {
        let files = vec![PathBuf::from("a.laz"), PathBuf::from("b.laz")];
        let results = vec![make_result(None, 3), make_result(None, 7)];
        let err = validate(&files, &results, None, None, None).unwrap_err();
        assert!(matches!(err, Error::PointFormatMismatch { .. }));
    }

    #[test]
    fn validate_temporal_index_requires_gps_time() {
        let files = vec![PathBuf::from("a.laz")];
        let results = vec![make_result(None, 0)];
        let err = validate(&files, &results, None, None, Some(1000)).unwrap_err();
        assert!(matches!(err, Error::NoGpsTime { .. }));
    }

    #[test]
    fn validate_temporal_index_with_gps_time() {
        let files = vec![PathBuf::from("a.laz")];
        let results = vec![make_result(None, 1)];
        let v = validate(&files, &results, None, None, Some(1000)).unwrap();
        assert_eq!(v.point_format, 6);
    }

    #[test]
    fn validate_extras_matching_with_identical_stats() {
        let files = vec![PathBuf::from("a.laz"), PathBuf::from("b.laz")];
        let (canonical, parsed_a) = build_one_field_vlr(10, "zNorm", -1.0, 10.0);
        let (_, parsed_b) = build_one_field_vlr(10, "zNorm", -1.0, 10.0);
        let results = vec![
            make_result_with_extras(None, parsed_a, 8, 3),
            make_result_with_extras(None, parsed_b, 8, 3),
        ];
        let v = validate(&files, &results, None, Some(canonical), None).unwrap();
        assert_eq!(v.num_extra_bytes, 8);
        let merged = ParsedExtraBytes::parse(v.extra_bytes_vlr.as_ref().unwrap()).unwrap();
        assert_eq!(merged.stats[0].min[0], -1.0);
        assert_eq!(merged.stats[0].max[0], 10.0);
    }

    /// Two files declare the *same logical schema* but have different
    /// per-file stats (different min/max because the data ranges
    /// differ). The schema-hash check must accept the inputs and merge
    /// stats honestly into the output.
    #[test]
    fn validate_extras_same_schema_different_stats_is_accepted() {
        let files = vec![PathBuf::from("a.laz"), PathBuf::from("b.laz")];
        let (canonical, parsed_a) = build_one_field_vlr(10, "zNorm", -4.5, 10.0);
        let (_, parsed_b) = build_one_field_vlr(10, "zNorm", -2.0, 24.7);
        let results = vec![
            make_result_with_extras(None, parsed_a, 8, 3),
            make_result_with_extras(None, parsed_b, 8, 3),
        ];
        let v = validate(&files, &results, None, Some(canonical), None).unwrap();
        let merged = ParsedExtraBytes::parse(v.extra_bytes_vlr.as_ref().unwrap()).unwrap();
        // Output stats must be the union: min from a, max from b.
        assert_eq!(merged.stats[0].min[0], -4.5);
        assert_eq!(merged.stats[0].max[0], 24.7);
    }

    #[test]
    fn validate_extras_data_type_mismatch_lists_field() {
        let files = vec![PathBuf::from("a.laz"), PathBuf::from("b.laz")];
        let (canonical, parsed_a) = build_one_field_vlr(3, "treeId", 0.0, 0.0); // u16
        let (_, parsed_b) = build_one_field_vlr(5, "treeId", 0.0, 0.0); // u32
        let results = vec![
            make_result_with_extras(None, parsed_a, 2, 3),
            make_result_with_extras(None, parsed_b, 4, 3),
        ];
        let err = validate(&files, &results, None, Some(canonical), None).unwrap_err();
        let Error::ExtraBytesMismatch {
            file_a,
            file_b,
            detail,
        } = err
        else {
            panic!("expected ExtraBytesMismatch");
        };
        assert_eq!(file_a, PathBuf::from("a.laz"));
        assert_eq!(file_b, PathBuf::from("b.laz"));
        // Trailing-byte width mismatch (u16 -> 2 bytes, u32 -> 4 bytes)
        // and per-field data_type difference must both be reported.
        assert!(
            detail.contains("trailing-byte width"),
            "expected width line, got:\n{detail}"
        );
        assert!(
            detail.contains("data_type") && detail.contains("treeId"),
            "expected per-field data_type diff for treeId, got:\n{detail}"
        );
        assert!(detail.contains("u16"));
        assert!(detail.contains("u32"));
    }

    #[test]
    fn validate_extras_one_file_lacks_them() {
        let files = vec![PathBuf::from("a.laz"), PathBuf::from("b.laz")];
        let (canonical, parsed_a) = build_one_field_vlr(10, "zNorm", 0.0, 0.0);
        let results = vec![
            make_result_with_extras(None, parsed_a, 8, 3),
            make_result(None, 3), // no extras
        ];
        let err = validate(&files, &results, None, Some(canonical), None).unwrap_err();
        let Error::ExtraBytesMismatch { detail, .. } = err else {
            panic!("expected ExtraBytesMismatch");
        };
        assert!(
            detail.contains("declares an Extra Bytes VLR") && detail.contains("declares none"),
            "expected presence-mismatch line, got:\n{detail}"
        );
    }

    /// All differences must be listed in a single error, not just the
    /// first one — so the user can fix the whole input set in one pass
    /// rather than re-running once per discovered problem.
    #[test]
    fn validate_extras_lists_all_differences() {
        let files = vec![PathBuf::from("a.laz"), PathBuf::from("b.laz")];
        let (canonical, parsed_a) = build_one_field_vlr(3, "treeId", 0.0, 0.0);
        let (mut other_bytes, _) = build_one_field_vlr(5, "treeId", 0.0, 0.0);
        // Force a description mismatch on top of the data_type mismatch.
        let desc = b"different desc";
        other_bytes[160..160 + desc.len()].copy_from_slice(desc);
        let parsed_b = ParsedExtraBytes::parse(&other_bytes).unwrap();
        let results = vec![
            make_result_with_extras(None, parsed_a, 2, 3),
            make_result_with_extras(None, parsed_b, 4, 3),
        ];
        let err = validate(&files, &results, None, Some(canonical), None).unwrap_err();
        let Error::ExtraBytesMismatch { detail, .. } = err else {
            panic!("expected ExtraBytesMismatch");
        };
        assert!(
            detail.contains("data_type"),
            "missing data_type diff in: {detail}"
        );
        assert!(
            detail.contains("description"),
            "missing description diff in: {detail}"
        );
    }
}
