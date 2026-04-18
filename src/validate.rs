/// Validate consistency of scanned input files before building the octree.
use crate::Error;
use crate::octree::{Crs, ScanResult, input_to_copc_format};
use std::path::PathBuf;
use tracing::debug;

/// Validated output: consistent properties across all input files.
#[derive(Debug)]
pub struct ValidatedInputs {
    /// WKT CRS string from input files (if present).
    pub wkt_crs: Option<Vec<u8>>,
    /// COPC output point format (6, 7, or 8).
    pub point_format: u8,
}

/// Returns true if the LAS point format includes GPS time.
fn format_has_gps_time(fmt: u8) -> bool {
    // LAS formats 0 and 2 lack GPS time; all others (1, 3–10) include it.
    !matches!(fmt, 0 | 2)
}

/// Check that all scanned files agree on CRS and point format,
/// and derive the COPC output point format.
pub fn validate(
    input_files: &[PathBuf],
    results: &[ScanResult],
    temporal_index: bool,
) -> crate::Result<ValidatedInputs> {
    // find a wkt crs
    let mut wkt_crs = results.iter().enumerate().find_map(|(i, result)| {
        if let Some(Crs::Wkt(wkt)) = &result.crs {
            Some((i, Crs::Wkt(wkt.to_owned())))
        } else {
            None
        }
    });
    // None of the files have a wkt defined crs
    if wkt_crs.is_none() {
        let epsg_crs = results.iter().enumerate().find_map(|(i, r)| {
            if let Some(Crs::GeoTiffEpsg(horizontal, _vertical)) = r.crs {
                Some((i, horizontal))
            } else {
                None
            }
        });

        if let Some((i, epsg)) = epsg_crs {
            // use the crs-definitions registry to get wkt data
            debug!("Translating GeoTiffCrs to Wkt, ignoring vertical component.");
            wkt_crs = crs_definitions::from_code(epsg)
                .map(|epsg| epsg.wkt.as_bytes().to_vec())
                .map(Crs::Wkt)
                .map(|w| (i, w));
            if wkt_crs.is_none() {
                debug!("Translating GeoTiffCrs to Wkt failed. Code not found in registry.");
            }
        } else {
            debug!("None of the files contain CRS information.")
        }
    }

    let first_format = results[0].point_format_id;

    let crs_file_index = if let Some((i, _)) = &wkt_crs { *i } else { 0 };
    let wkt_crs = wkt_crs.map(|(_, crs)| crs);

    for (i, r) in results.iter().enumerate().skip(1) {
        match (&wkt_crs, &r.crs) {
            (None, None) => (),
            (Some(crs), Some(other)) => {
                if !crs.is_equal_to(other)? {
                    debug!("Ignoring vertical CRS-component in CRS comparison");
                    if !crs.is_equal_to_ignore_vertical_component(other)? {
                        return Err(Error::CrsMismatch {
                            file_a: input_files[crs_file_index].clone(),
                            file_b: input_files[i].clone(),
                        });
                    }
                }
            }
            _ => {
                return Err(Error::CrsMismatch {
                    file_a: input_files[crs_file_index].clone(),
                    file_b: input_files[i].clone(),
                });
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

    let wkt_crs = match wkt_crs {
        Some(Crs::Wkt(v)) => Some(v),
        None => None,
        _ => unreachable!(),
    };

    if temporal_index && !format_has_gps_time(first_format) {
        return Err(Error::NoGpsTime {
            format: first_format,
        });
    }

    let point_format = input_to_copc_format(first_format);
    debug!("Input point format: {first_format}, output COPC point format: {point_format}");

    Ok(ValidatedInputs {
        wkt_crs,
        point_format,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::octree::{Bounds, ScanResult};

    const WGS84_WKT: &[u8; 256] = br#"GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]"#;
    const EPSG3006_WKT: &[u8; 980] = br#"PROJCRS["SWEREF99 TM",BASEGEOGCRS["SWEREF99",DATUM["SWEREF99",ELLIPSOID["GRS 1980",6378137,298.257222101,LENGTHUNIT["metre",1]]],PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],ID["EPSG",4619]],CONVERSION["SWEREF99 TM",METHOD["Transverse Mercator",ID["EPSG",9807]],PARAMETER["Latitude of natural origin",0,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8801]],PARAMETER["Longitude of natural origin",15,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8802]],PARAMETER["Scale factor at natural origin",0.9996,SCALEUNIT["unity",1],ID["EPSG",8805]],PARAMETER["False easting",500000,LENGTHUNIT["metre",1],ID["EPSG",8806]],PARAMETER["False northing",0,LENGTHUNIT["metre",1],ID["EPSG",8807]]],CS[Cartesian,2],AXIS["northing (N)",north,ORDER[1],LENGTHUNIT["metre",1]],AXIS["easting (E)",east,ORDER[2],LENGTHUNIT["metre",1]],USAGE[SCOPE["Topographic mapping (medium and small scale)."],AREA["Sweden - onshore and offshore."],BBOX[54.96,10.03,69.07,24.17]],ID["EPSG",3006]]"#;

    fn make_result(crs: Option<Crs>, fmt: u8) -> ScanResult {
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
            point_format_id: fmt,
        }
    }

    #[test]
    fn validate_single_file() {
        let files = vec![PathBuf::from("a.laz")];
        let results = vec![make_result(None, 3)];
        let v = validate(&files, &results, false).unwrap();
        assert_eq!(v.point_format, 7);
        assert!(v.wkt_crs.is_none());
    }

    #[test]
    fn validate_matching_files() {
        let files = vec![PathBuf::from("a.laz"), PathBuf::from("b.laz")];
        let wkt = Some(Crs::Wkt(WGS84_WKT.to_vec()));
        let results = vec![make_result(wkt.clone(), 8), make_result(wkt, 8)];
        let v = validate(&files, &results, false).unwrap();
        assert_eq!(v.point_format, 8);
    }

    #[test]
    fn validate_crs_wkt_and_geotiff() {
        let files = vec![PathBuf::from("a.laz"), PathBuf::from("b.laz")];
        let results = vec![
            make_result(Some(Crs::Wkt(EPSG3006_WKT.to_vec())), 7),
            make_result(Some(Crs::GeoTiffEpsg(3006, None)), 7),
        ];
        let v = validate(&files, &results, false).unwrap();
        assert_eq!(v.wkt_crs, Some(EPSG3006_WKT.to_vec()));
    }

    #[test]
    fn validate_crs_geotiff_and_wkt() {
        let files = vec![PathBuf::from("a.laz"), PathBuf::from("b.laz")];
        let results = vec![
            make_result(Some(Crs::GeoTiffEpsg(3006, None)), 7),
            make_result(Some(Crs::Wkt(EPSG3006_WKT.to_vec())), 7),
        ];
        let v = validate(&files, &results, false).unwrap();
        assert_eq!(v.wkt_crs, Some(EPSG3006_WKT.to_vec()));
    }

    #[test]
    fn validate_crs_geotiff_and_geotiff() {
        let files = vec![PathBuf::from("a.laz"), PathBuf::from("b.laz")];
        let results = vec![
            make_result(Some(Crs::GeoTiffEpsg(3006, None)), 7),
            make_result(Some(Crs::GeoTiffEpsg(3006, None)), 7),
        ];
        let v = validate(&files, &results, false).unwrap();
        assert_eq!(v.wkt_crs, Some(EPSG3006_WKT.to_vec()));
    }

    #[test]
    fn validate_crs_mismatch() {
        let files = vec![PathBuf::from("a.laz"), PathBuf::from("b.laz")];
        let results = vec![
            make_result(Some(Crs::Wkt(WGS84_WKT.to_vec())), 7),
            make_result(Some(Crs::Wkt(EPSG3006_WKT.to_vec())), 7),
        ];
        let err = validate(&files, &results, false).unwrap_err();
        assert!(matches!(err, Error::CrsMismatch { .. }));
    }

    #[test]
    fn validate_format_mismatch() {
        let files = vec![PathBuf::from("a.laz"), PathBuf::from("b.laz")];
        let results = vec![make_result(None, 3), make_result(None, 7)];
        let err = validate(&files, &results, false).unwrap_err();
        assert!(matches!(err, Error::PointFormatMismatch { .. }));
    }

    #[test]
    fn validate_temporal_index_requires_gps_time() {
        let files = vec![PathBuf::from("a.laz")];
        let results = vec![make_result(None, 0)];
        let err = validate(&files, &results, true).unwrap_err();
        assert!(matches!(err, Error::NoGpsTime { .. }));
    }

    #[test]
    fn validate_temporal_index_with_gps_time() {
        let files = vec![PathBuf::from("a.laz")];
        let results = vec![make_result(None, 1)];
        let v = validate(&files, &results, true).unwrap();
        assert_eq!(v.point_format, 6);
    }
}
