//! ruhrwatt-core: fast SMARD parsing and bronze-layer Parquet writing.

mod bronze;
mod error;
mod parse;

use std::path::{Path, PathBuf};

use pyo3::{
    exceptions::{PyOSError, PyValueError},
    prelude::*,
    types::PyDict,
};

use error::CoreError;

fn to_py(err: CoreError) -> PyErr {
    match err {
        CoreError::Io { .. } => PyOSError::new_err(err.to_string()),
        _ => PyValueError::new_err(err.to_string()),
    }
}

/// Parse one weekly SMARD file into parallel lists of epoch-ms timestamps and values.
#[pyfunction]
fn parse_week(py: Python<'_>, path: &str) -> PyResult<(Vec<i64>, Vec<Option<f64>>)> {
    let rows = py
        .detach(|| parse::parse_file(Path::new(path)))
        .map_err(to_py)?;
    Ok(rows.into_iter().unzip())
}

/// Parse, merge, validate and write many weekly files to a single Parquet file.
///
/// Returns a dict of quality counters. The GIL is released for the whole job.
#[pyfunction]
#[pyo3(signature = (files, out_path, step_ms, native_from_ms=None))]
fn build_bronze<'py>(
    py: Python<'py>,
    files: Vec<String>,
    out_path: String,
    step_ms: i64,
    native_from_ms: Option<i64>,
) -> PyResult<Bound<'py, PyDict>> {
    let paths: Vec<PathBuf> = files.into_iter().map(PathBuf::from).collect();
    let stats = py
        .detach(|| bronze::build(&paths, Path::new(&out_path), step_ms, native_from_ms))
        .map_err(to_py)?;

    let d = PyDict::new(py);
    d.set_item("files", stats.files)?;
    d.set_item("rows", stats.rows)?;
    d.set_item("nulls", stats.nulls)?;
    d.set_item("duplicates", stats.duplicates)?;
    d.set_item("gaps", stats.gaps)?;
    d.set_item("misaligned", stats.misaligned)?;
    d.set_item("native_rows", stats.native_rows)?;
    d.set_item("first_ts_ms", stats.first_ts_ms)?;
    d.set_item("last_ts_ms", stats.last_ts_ms)?;
    Ok(d)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_week, m)?)?;
    m.add_function(wrap_pyfunction!(build_bronze, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
