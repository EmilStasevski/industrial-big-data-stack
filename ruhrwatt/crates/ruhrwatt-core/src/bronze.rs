//! Merge weekly files into one clean series, check its quality, write Parquet.

use std::{
    fs::{self, File},
    path::{Path, PathBuf},
    sync::Arc,
};

use arrow_array::{ArrayRef, BooleanArray, Float64Array, RecordBatch, TimestampMillisecondArray};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use parquet::{arrow::ArrowWriter, basic::Compression, file::properties::WriterProperties};
use rayon::prelude::*;

use crate::{
    error::CoreError,
    parse::{Row, parse_file},
};

const CHUNK_ROWS: usize = 500_000;
const HOUR_MS: i64 = 3_600_000;

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Stats {
    pub files: usize,
    pub rows: usize,
    pub nulls: usize,
    pub duplicates: usize,
    pub gaps: usize,
    pub misaligned: usize,
    pub native_rows: usize,
    pub first_ts_ms: Option<i64>,
    pub last_ts_ms: Option<i64>,
}

/// Merged series: timestamps, values (same length), and the number of duplicates dropped.
pub type Merged = (Vec<i64>, Vec<Option<f64>>, usize);

/// Sort by timestamp and drop duplicates.
///
/// Identical duplicates are collapsed; a null next to a value keeps the value;
/// two different values for one timestamp are an error (never silently pick one).
pub fn merge(mut rows: Vec<Row>) -> Result<Merged, CoreError> {
    rows.sort_unstable_by_key(|r| r.0);
    let mut ts: Vec<i64> = Vec::with_capacity(rows.len());
    let mut vals: Vec<Option<f64>> = Vec::with_capacity(rows.len());
    let mut duplicates = 0;
    for (t, v) in rows {
        if ts.last() == Some(&t) {
            duplicates += 1;
            let last = vals.last_mut().expect("ts and vals grow together");
            match (*last, v) {
                (Some(a), Some(b)) if a != b => {
                    return Err(CoreError::Conflict { ts_ms: t, a, b });
                }
                (None, Some(_)) => *last = v,
                _ => {}
            }
        } else {
            ts.push(t);
            vals.push(v);
        }
    }
    Ok((ts, vals, duplicates))
}

/// Quality counters for a merged series.
///
/// `step_ms <= 0` disables gap and alignment checks (week/month/year series).
/// `native_from_ms = None` means every row is native resolution.
pub fn quality(
    ts: &[i64],
    vals: &[Option<f64>],
    step_ms: i64,
    native_from_ms: Option<i64>,
) -> Stats {
    let mut stats = Stats {
        rows: ts.len(),
        nulls: vals.iter().filter(|v| v.is_none()).count(),
        first_ts_ms: ts.first().copied(),
        last_ts_ms: ts.last().copied(),
        ..Stats::default()
    };
    if step_ms > 0 {
        for w in ts.windows(2) {
            let d = w[1] - w[0];
            if d > step_ms {
                stats.gaps += (d / step_ms - 1) as usize;
            }
        }
        // Day-or-longer buckets start at Berlin midnight, so only sub-day steps align to UTC.
        if step_ms <= HOUR_MS {
            stats.misaligned = ts.iter().filter(|t| t.rem_euclid(step_ms) != 0).count();
        }
    }
    stats.native_rows = match native_from_ms {
        None => ts.len(),
        Some(n) => ts.iter().filter(|&&t| t >= n).count(),
    };
    stats
}

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new(
            "ts_utc",
            DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into())),
            false,
        ),
        Field::new("value", DataType::Float64, true),
        Field::new("is_native_15min", DataType::Boolean, false),
    ]))
}

pub fn write_parquet(
    out: &Path,
    ts: &[i64],
    vals: &[Option<f64>],
    native_from_ms: Option<i64>,
) -> Result<(), CoreError> {
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent).map_err(|source| CoreError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    let mut tmp_name = out.as_os_str().to_owned();
    tmp_name.push(".part");
    let tmp = PathBuf::from(tmp_name);

    let schema = schema();
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let file = File::create(&tmp).map_err(|source| CoreError::Io {
        path: tmp.clone(),
        source,
    })?;
    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;

    for (t_chunk, v_chunk) in ts.chunks(CHUNK_ROWS).zip(vals.chunks(CHUNK_ROWS)) {
        let native: Vec<bool> = t_chunk
            .iter()
            .map(|&t| native_from_ms.is_none_or(|n| t >= n))
            .collect();
        let columns: Vec<ArrayRef> = vec![
            Arc::new(TimestampMillisecondArray::from(t_chunk.to_vec()).with_timezone("UTC")),
            Arc::new(Float64Array::from(v_chunk.to_vec())),
            Arc::new(BooleanArray::from(native)),
        ];
        writer.write(&RecordBatch::try_new(schema.clone(), columns)?)?;
    }
    writer.close()?;
    fs::rename(&tmp, out).map_err(|source| CoreError::Io {
        path: out.to_path_buf(),
        source,
    })?;
    Ok(())
}

/// Parse all files in parallel, merge, check quality and write one Parquet file.
pub fn build(
    files: &[PathBuf],
    out: &Path,
    step_ms: i64,
    native_from_ms: Option<i64>,
) -> Result<Stats, CoreError> {
    if files.is_empty() {
        return Err(CoreError::NoFiles);
    }
    let parts: Vec<Vec<Row>> = files
        .par_iter()
        .map(|p| parse_file(p))
        .collect::<Result<_, _>>()?;
    let all: Vec<Row> = parts.into_iter().flatten().collect();
    let (ts, vals, duplicates) = merge(all)?;
    let mut stats = quality(&ts, &vals, step_ms, native_from_ms);
    stats.files = files.len();
    stats.duplicates = duplicates;
    write_parquet(out, &ts, &vals, native_from_ms)?;
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_sorts_and_collapses_identical_duplicates() {
        let rows = vec![
            (3, Some(3.0)),
            (1, Some(1.0)),
            (2, None),
            (2, None),
            (3, Some(3.0)),
        ];
        let (ts, vals, dups) = merge(rows).unwrap();
        assert_eq!(ts, vec![1, 2, 3]);
        assert_eq!(vals, vec![Some(1.0), None, Some(3.0)]);
        assert_eq!(dups, 2);
    }

    #[test]
    fn merge_prefers_value_over_null() {
        let (_, vals, _) = merge(vec![(1, None), (1, Some(5.0))]).unwrap();
        assert_eq!(vals, vec![Some(5.0)]);
    }

    #[test]
    fn merge_rejects_conflicting_values() {
        assert!(matches!(
            merge(vec![(1, Some(1.0)), (1, Some(2.0))]),
            Err(CoreError::Conflict { ts_ms: 1, .. })
        ));
    }

    #[test]
    fn quality_counts_gaps_nulls_and_native_rows() {
        let step = 900_000;
        let ts = vec![0, step, 4 * step, 5 * step + 7];
        let vals = vec![Some(1.0), None, Some(2.0), Some(3.0)];
        let s = quality(&ts, &vals, step, Some(4 * step));
        assert_eq!(s.gaps, 2); // two missing quarter-hours between 1 and 4
        assert_eq!(s.nulls, 1);
        assert_eq!(s.misaligned, 1);
        assert_eq!(s.native_rows, 2);
    }

    #[test]
    fn day_series_skips_alignment_check() {
        let s = quality(&[1, 90_000_000], &[Some(1.0), Some(2.0)], 86_400_000, None);
        assert_eq!(s.misaligned, 0);
    }
}
