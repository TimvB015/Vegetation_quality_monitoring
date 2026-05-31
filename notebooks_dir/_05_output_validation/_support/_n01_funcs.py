from __future__ import annotations

###############################################################################
## READ THE ID OF THE RUN
###############################################################################
import re
from typing import Dict

def run_id_read(run_id: str) -> Dict[str, str]:
    """
    Parse a run_id like:
        "WD1_plusOW_p80_tmp2_cart_at1__gelderland__RD"

    Extracts:
      - hab_selection: everything before the first "_"
      - train_split_attempt: token that starts with "at" followed by digits (e.g. "at1"),
        also works when surrounded by underscores.

    Returns: {"hab_selection": ..., "train_split_attempt": ...}
    """
    if run_id is None or str(run_id).strip() == "":
        raise ValueError("run_id is empty/None")

    s = str(run_id).strip()

    hab_selection = s.split("_", 1)[0].strip()
    if not hab_selection:
        raise ValueError(f"Could not parse hab_selection from run_id='{run_id}'")

    # Use underscore/end boundaries instead of \b (because "_" counts as a word char)
    m = re.search(r"(?:(?<=_)|^)at\d+(?:(?=_)|$)", s)
    if not m:
        raise ValueError(f"Could not find train_split_attempt like 'at1' in run_id='{run_id}'")

    train_split_attempt = m.group(0)

    return {
        "hab_selection": hab_selection,
        "train_split_attempt": train_split_attempt,
    }



###############################################################################
## FIND THE CARTO OUTPUT ZIPPED FOLDER BASED ON GELDERLAND DIV
###############################################################################
from pathlib import Path
from typing import Union

def find_gelderland_zip_folder(
    input_dir: Union[str, Path],
    hab_selection: str,
    train_split_attempt: str,
) -> Path:
    """
    Build the expected path to the Gelderland division zip in a robust, OS-safe way.
    Returns a Path object.
    """
    input_dir = Path(input_dir)

    hab_selection = str(hab_selection).strip()
    train_split_attempt = str(train_split_attempt).strip()

    return (
        input_dir
        / "01_RAW_carto_output"
        / f"{hab_selection}_{train_split_attempt}_gelderland_division.zip"
    )



################################################################################
## READ CARTO README FILE
################################################################################
import re
import zipfile
from pathlib import Path
from typing import Dict, Union


def read_carto_README_file(zipped_folder_path: Union[str, Path]) -> Dict[str, str]:
    """
    Read README.txt inside a zip archive and extract the class mapping.

    Expected README structure includes a section like:
        Class mapping:
        - 1: Remaining
        - 2: Wet Nature
        - 3: Open Water

    Returns:
        Dict[str, str]: e.g. {"1": "Remaining", "2": "Wet Nature", "3": "Open Water"}
    """
    zipped_folder_path = Path(zipped_folder_path)

    with zipfile.ZipFile(zipped_folder_path, "r") as zf:
        # Find README.txt anywhere in the zip (case-insensitive)
        readme_name = next(
            (name for name in zf.namelist() if Path(name).name.lower() == "readme.txt"),
            None,
        )
        if readme_name is None:
            raise FileNotFoundError("README.txt not found inside the provided zip archive.")

        with zf.open(readme_name) as f:
            text = f.read().decode("utf-8", errors="replace")

    # Extract lines like: - 1: Remaining
    mapping = {}
    for m in re.finditer(r"^\s*-\s*(\d+)\s*:\s*(.+?)\s*$", text, flags=re.MULTILINE):
        mapping[m.group(1)] = m.group(2)

    if not mapping:
        raise ValueError("No class mapping entries found in README.txt.")

    return mapping



################################################################################
## EXTRACT TIF FILES FROM THE ZIP
################################################################################
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import List, Tuple, Union, Optional


def extract_carto_tifs_from_zip(
    zipped_folder_path: Union[str, Path],
    extract_dir: Optional[Union[str, Path]] = None,
    overwrite: bool = True,
) -> tuple[List[Path], Path, List[str]]:
    """
    Extract GeoTIFFs from a zip whose *filename* ends with _<YEAR>.tif (case-insensitive),
    and place them flatly into `extract_dir` (ignoring any internal zip subfolders).

    Prints exactly:
      "in Carto-files the following years were found and used: YEARS"

    Returns (paths, extract_dir, years), where:
      - paths: list of extracted GeoTIFF paths usable by rasterio
      - extract_dir: the directory containing the extracted files
      - years: list of years (strings) in the same order as `paths`
    """
    zipped_folder_path = Path(zipped_folder_path)

    year_pat = re.compile(r".*_(\d{4})\.tif$", re.IGNORECASE)

    if extract_dir is None:
        extract_dir = Path(tempfile.mkdtemp(prefix="carto_zip_"))
    else:
        extract_dir = Path(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)

    matches: List[Tuple[int, str, str]] = []  # (year, member_name, basename)

    with zipfile.ZipFile(zipped_folder_path, "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            member_name = member.filename
            base = Path(member_name).name
            m = year_pat.match(base)
            if m:
                matches.append((int(m.group(1)), member_name, base))

        matches.sort(key=lambda x: x[0])
        years = [str(y) for y, _, _ in matches]

        out_paths: List[Path] = []
        for _, member_name, base in matches:
            out_fp = extract_dir / base

            if out_fp.exists() and not overwrite:
                out_paths.append(out_fp)
                continue

            with zf.open(member_name) as src, open(out_fp, "wb") as dst:
                shutil.copyfileobj(src, dst)

            out_paths.append(out_fp)

    print(f"in Carto-files the following years were found and used: {', '.join(years)}")
    return out_paths, extract_dir, years



################################################################################
## BUILD THE CLASSIFICATION ACCURACY DF
################################################################################
from pathlib import Path
from typing import Optional, Union, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from functions.ML_classification_check_funcs import ml_classification_check


def _as_clean_str_labels(arr) -> list[str]:
    """Normalize labels to consistent string type for sklearn metrics."""
    s = pd.Series(list(arr), dtype="object")
    s = s.map(lambda v: None if v is None else str(v).strip())
    s = s.dropna()
    s = s.astype("string").str.replace(r"\.0$", "", regex=True)
    s = s.str.casefold()
    return s.tolist()


def _sanitize_col_suffix(v: str) -> str:
    """Make a safe column suffix from a class label."""
    v = str(v).strip()
    v = v.replace(" ", "_").replace("/", "_").replace("\\", "_")
    return v


def _round2_or_none(x):
    if x is None:
        return None
    try:
        if pd.isna(x):
            return np.nan
    except Exception:
        pass
    return float(np.round(float(x), 2))


def _per_class_acc(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, float | None]:
    """
    Per-class accuracy (recall): P(pred==c | true==c) for each class c.
    Returns dict like {"acc_<label>": 0.83, ...}
    If a class has no true samples in this year -> None.
    """
    yt = pd.Series(y_true, dtype="string")
    yp = pd.Series(y_pred, dtype="string")

    out: dict[str, float | None] = {}
    for lab in labels:
        m = (yt == lab)
        n_c = int(m.sum())
        key = f"acc_{_sanitize_col_suffix(lab)}"
        out[key] = (_round2_or_none((yp[m] == lab).mean()) if n_c > 0 else None)
    return out


def _per_class_prf_n(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
) -> dict[str, float | int | None]:
    """
    Per-class precision/recall/f1 and support (renamed to n_<class>).

    If a class has no true samples in this year -> prec/rec/f1 should be None and n_<class>=0.
    """
    out: dict[str, float | int | None] = {}

    if len(y_true) == 0:
        for lab in labels:
            suf = _sanitize_col_suffix(lab)
            out[f"f1_{suf}"] = None
            out[f"prec_{suf}"] = None
            out[f"rec_{suf}"] = None
            out[f"n_{suf}"] = 0
        return out

    p, r, f1s, s = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,  # will be overridden to None for classes with n==0
    )

    for lab, pv, rv, fv, sv in zip(labels, p, r, f1s, s):
        suf = _sanitize_col_suffix(lab)
        sv = int(sv)
        if sv == 0:
            out[f"f1_{suf}"] = None
            out[f"prec_{suf}"] = None
            out[f"rec_{suf}"] = None
            out[f"n_{suf}"] = 0
        else:
            out[f"f1_{suf}"] = _round2_or_none(fv)
            out[f"prec_{suf}"] = _round2_or_none(pv)
            out[f"rec_{suf}"] = _round2_or_none(rv)
            out[f"n_{suf}"] = sv

    return out


def classification_accuracy_per_year(
    validation_gdf,
    stacked_rstr_path: Union[str, Path],
    label_col: str,
    *,
    year_col: str = "years",
    raster_band_years: Optional[Sequence[Union[int, str]]] = None,
    nodata_value=None,
    class_map=None,
    assume_points: bool = False,
    dropna_year: bool = True,
) -> pd.DataFrame:
    """
    Output columns (order):
      clas_year, n, acc, macro_f1, weighted_f1,
      acc_<class>, f1_<class>, prec_<class>, rec_<class>, n_<class>, ...
    All scores rounded to 2 decimals. If a class has no validation data in a year -> None for scores.
    """
    stacked_rstr_path = Path(stacked_rstr_path)

    if validation_gdf is None or len(validation_gdf) == 0:
        return pd.DataFrame(columns=["clas_year", "n", "acc", "macro_f1", "weighted_f1"])

    gdf = validation_gdf.copy()
    if dropna_year:
        gdf = gdf[gdf[year_col].notna()].copy()
    if len(gdf) == 0:
        return pd.DataFrame(columns=["clas_year", "n", "acc", "macro_f1", "weighted_f1"])

    # --- label set mismatch warning (cleaned, case-insensitive) ---
    raster_labels_clean: Optional[set[str]] = None
    if class_map is not None:
        raster_labels_clean = set(_as_clean_str_labels(class_map.values()))

    gdf_labels_clean = set(_as_clean_str_labels(gdf[label_col].to_list()))

    if raster_labels_clean is not None:
        missing_in_raster = sorted(gdf_labels_clean - raster_labels_clean)
        extra_in_raster = sorted(raster_labels_clean - gdf_labels_clean)
        if missing_in_raster or extra_in_raster:
            msg = "Warning: label mismatch between gdf and raster class map after cleaning.\n"
            if missing_in_raster:
                msg += f"  Present in gdf but not in raster CLASS-MAP: {missing_in_raster}\n"
            if extra_in_raster:
                msg += f"  Present in raster CLASS-MAP but not in gdf: {extra_in_raster}\n"
            msg += "  This may reduce per-class scores (unmapped labels will never match)."
            print(msg)

    # Years to process
    if raster_band_years is not None:
        years_sorted = [str(y) for y in raster_band_years]
    else:
        years = pd.unique(gdf[year_col])
        years_sorted = sorted([str(y) for y in years], key=lambda x: int(x) if x.isdigit() else x)

    # Stable per-class columns (from gdf labels, cleaned)
    all_classes = sorted(gdf_labels_clean)

    results = []
    for y in years_sorted:
        gdf_y = gdf[gdf[year_col].astype(str) == str(y)].copy()

        if len(gdf_y) == 0:
            row = {"clas_year": str(y), "n": 0, "acc": np.nan, "macro_f1": np.nan, "weighted_f1": np.nan}
            # stable columns even if no samples at all this year
            for c in all_classes:
                suf = _sanitize_col_suffix(c)
                row[f"acc_{suf}"] = None
                row[f"f1_{suf}"] = None
                row[f"prec_{suf}"] = None
                row[f"rec_{suf}"] = None
                row[f"n_{suf}"] = 0
            results.append(row)
            continue

        # Determine band index for this year
        if raster_band_years is not None:
            try:
                band = list(map(str, raster_band_years)).index(str(y)) + 1
            except ValueError:
                continue
        else:
            import rasterio

            with rasterio.open(stacked_rstr_path) as src:
                desc = [src.descriptions[i] for i in range(src.count)]
            try:
                band = [str(d) for d in desc].index(str(y)) + 1
            except ValueError:
                continue

        y_true_raw, y_pred_raw, _ = ml_classification_check(
            gdf_y,
            stacked_rstr_path,
            label_col,
            raster_band=band,
            nodata_value=nodata_value,
            class_map=class_map,
            assume_points=assume_points,
            warn_multiband=False,
        )

        y_true = _as_clean_str_labels(y_true_raw)
        y_pred = _as_clean_str_labels(y_pred_raw)

        # Safety: ensure same length
        m = min(len(y_true), len(y_pred))
        y_true = y_true[:m]
        y_pred = y_pred[:m]

        n = int(len(y_true))
        if n == 0:
            acc = macro = w = np.nan
        else:
            acc = _round2_or_none(accuracy_score(y_true, y_pred))
            macro = _round2_or_none(f1_score(y_true, y_pred, average="macro", zero_division=0))
            w = _round2_or_none(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        row = {"clas_year": str(y), "n": n, "acc": acc, "macro_f1": macro, "weighted_f1": w}

        # Per-class metrics
        row.update(_per_class_acc(y_true, y_pred, labels=all_classes))         # acc_<class>
        row.update(_per_class_prf_n(y_true, y_pred, labels=all_classes))       # f1_/prec_/rec_/n_

        results.append(row)

    # Column order exactly as requested:
    base_cols = ["clas_year", "n", "acc", "macro_f1", "weighted_f1"]
    per_class_cols: list[str] = []
    for c in all_classes:
        suf = _sanitize_col_suffix(c)
        per_class_cols += [
            f"acc_{suf}",
            f"f1_{suf}",
            f"prec_{suf}",
            f"rec_{suf}",
            f"n_{suf}",
        ]

    return pd.DataFrame(results, columns=base_cols + per_class_cols)