//! Parsing of SMARD `chart_data` weekly files:
//! `{"meta_data": {...}, "series": [[ts_ms, value|null], ...]}`.

use std::{fs, path::Path};

use serde::Deserialize;

use crate::error::CoreError;

/// One observation: epoch milliseconds (UTC) and an optional value.
pub type Row = (i64, Option<f64>);

#[derive(Deserialize)]
struct Payload {
    series: Vec<Row>,
}

pub fn parse_bytes(bytes: &[u8]) -> Result<Vec<Row>, serde_json::Error> {
    Ok(serde_json::from_slice::<Payload>(bytes)?.series)
}

pub fn parse_file(path: &Path) -> Result<Vec<Row>, CoreError> {
    let bytes = fs::read(path).map_err(|source| CoreError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    parse_bytes(&bytes).map_err(|source| CoreError::Json {
        path: path.to_path_buf(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_values_nulls_and_negatives() {
        let raw = br#"{"meta_data":{"version":1},"series":[[1000,41.5],[2000,null],[3000,-3.25],[4000,7]]}"#;
        let rows = parse_bytes(raw).unwrap();
        assert_eq!(
            rows,
            vec![
                (1000, Some(41.5)),
                (2000, None),
                (3000, Some(-3.25)),
                (4000, Some(7.0))
            ]
        );
    }

    #[test]
    fn rejects_payload_without_series() {
        assert!(parse_bytes(br#"{"other":1}"#).is_err());
    }
}
