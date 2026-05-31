from __future__ import annotations

import warnings

warnings.filterwarnings(
    "ignore",
    message=r"^['`]?sklearn\.utils\.parallel\.delayed['`]? should be used with ['`]?sklearn\.utils\.parallel\.Parallel['`]?\b.*",
    category=UserWarning,
)

from pathlib import Path
from typing import Sequence, Union
import json
import joblib

import numpy as np
import pandas as pd
import rasterio
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

from functions.gpkg_funcs import import_gpkg_func
from functions.s2_import_funcs import sentinel_path_builder
from functions.ML_classification_check_funcs import ml_classification_check
from notebooks_dir._04_ML_classification._support._n01_funcs import (
    expected_crs_from_input,
    RF_models,
    sample_training_bands_for_RF,
    apply_RF_model_to_rstrs,
)

###############################################################################
## FULL WORKFLOW
###############################################################################
def RF_trainer_workflow(
    *,
    # Raster stack inputs
    stacked_raster_dir: Union[str, Path],
    stacked_raster_crs: str,
    stacked_raster_filename: str,
    all_raster_band_names: Sequence[str],
    feature_band_names: Sequence[str],
    years: Sequence[Union[int, str]],
    timeframes: Sequence[Union[str, int]],
    # Valid mask
    valid_band_available: bool = True,
    valid_band_name: str = "VALID_MASK",
    valid_value: int = 1,
    # Train/val vectors
    train_gpkg_path: Union[str, Path],
    val_gpkg_path: Union[str, Path],
    gpkg_label_col: str = "type",
    gpkg_year_col: str = "years",
    # RF setup
    rf_model_selection: str = "mdl1",
    hab_selection: str = "WD1",
    # Naming controls
    stack_description: str = "b2348",
    train_split_attempt: str = "atX",
    # Output dirs (parent folder only)
    classified_rasters_dir: Union[str, Path],
    # Processing / outputs
    nodata_out: int = 255,
    # Single overwrite switch
    overwrite: bool = False,
    # Warnings / checks
    warn_on_unused_bands: bool = True,
) -> pd.DataFrame:
    """
    Train one RandomForest classifier per timeframe on multiple training years, apply each
    timeframe-specific model to the same timeframe across all requested years, write classified
    rasters (with class-map tags), and compute a validation summary table.

    Overwrite semantics (single switch)
    ----------------------------------
    - overwrite=True:
        * Always (re)train the model for each timeframe and overwrite the model file.
        * Always (re)create classified rasters for all years/timeframes (overwrite existing rasters).

    - overwrite=False:
        * If the timeframe model does NOT exist yet: train it and save it (first run).
        * If the timeframe model exists: load it (no retraining).
        * For rasters: only create rasters that do not exist yet; never overwrite existing rasters.

    Validation / class-map decoding
    -------------------------------
    `ml_classification_check` may decode predicted raster class codes using a dataset tag
    named "CLASS-MAP" (JSON). To be compatible with both old and new conventions, this
    function writes *both* tags:
      - "CLASS-MAP"
      - "CLASS_MAP"

    Additionally, `class_map` is passed explicitly into `ml_classification_check` so that
    validation does not depend on tags being present.

    Validation metrics reported
    ---------------------------
    The returned summary dataframe contains one row per (clas_year, timeframe) and includes:

    Overall metrics:
      - n: number of validation samples used (after masking/cleaning)
      - acc: overall accuracy
      - macro_f1: macro-averaged F1
      - weighted_f1: support-weighted F1

    Per-class metrics (columns are appended after `weighted_f1` in a fixed order per class):
      - acc_<class>: per-class accuracy defined as recall:
            P(y_pred == class | y_true == class)
      - f1_<class>: per-class F1 score
      - prec_<class>: per-class precision
      - rec_<class>: per-class recall
      - n_<class>: per-class support (count of validation samples of that class)

    Per-class missing-data behavior
    -------------------------------
    If a class does not occur in the validation set for a specific (year, timeframe),
    then for that row:
      - acc_<class>, f1_<class>, prec_<class>, rec_<class>, n_<class> are set to None.

    Rounding
    --------
    All floating-point metrics are rounded to 2 decimals (acc, macro_f1, weighted_f1, and
    per-class acc/f1/prec/rec). Counts are integers when present.

    Label type normalization (metrics safety)
    ----------------------------------------
    Even with class decoding, label sources can differ in dtype (e.g. numbers vs strings).
    Before computing sklearn metrics, `y_true` and `y_pred` are normalized to cleaned strings.

    Returns
    -------
    pandas.DataFrame
        Validation summary results with one row per (clas_year, timeframe).
        Base columns:
          - train_year
          - clas_year
          - timeframe
          - n
          - acc
          - macro_f1
          - weighted_f1

        Followed by per-class columns in this repeated order for each class:
          - acc_<class>, f1_<class>, prec_<class>, rec_<class>, n_<class>
    """

    def _as_clean_str_labels(arr) -> list[str]:
        """Normalize labels to consistent string type for sklearn metrics."""
        s = pd.Series(list(arr), dtype="object")
        s = s.map(lambda v: None if v is None else str(v).strip())
        s = s.dropna()
        # convert numeric-looking strings like "1.0" -> "1"
        s = s.astype("string").str.replace(r"\.0$", "", regex=True)
        return s.tolist()

    def _sanitize_col_suffix(v: str) -> str:
        """Make a safe column suffix from a class label."""
        v = str(v).strip()
        v = v.replace(" ", "_").replace("/", "_").replace("\\", "_")
        return v

    def _round2_or_none(x):
        """Round floats to 2 decimals; preserve None; preserve NaN."""
        if x is None:
            return None
        try:
            if pd.isna(x):
                return np.nan
        except Exception:
            pass
        return float(np.round(float(x), 2))

    def _per_class_acc(
        y_true: list[str],
        y_pred: list[str],
        labels: list[str],
    ) -> dict[str, float | None]:
        """
        Per-class accuracy (recall): P(pred==c | true==c) for each class c.
        If a class has no validation data -> None.
        """
        yt = pd.Series(y_true, dtype="string")
        yp = pd.Series(y_pred, dtype="string")

        out: dict[str, float | None] = {}
        for lab in labels:
            m = (yt == lab)
            n = int(m.sum())
            key = f"acc_{_sanitize_col_suffix(lab)}"
            out[key] = (_round2_or_none((yp[m] == lab).mean()) if n > 0 else None)
        return out

    def _per_class_f1_prec_rec_n(
        y_true: list[str],
        y_pred: list[str],
        labels: list[str],
    ) -> dict[str, float | int | None]:
        """
        Per-class f1/precision/recall and n (support).
        If a class has no validation data (support=0) -> f1/prec/rec/n = None.
        """
        out: dict[str, float | int | None] = {}

        if len(y_true) == 0:
            for lab in labels:
                suf = _sanitize_col_suffix(lab)
                out[f"f1_{suf}"] = None
                out[f"prec_{suf}"] = None
                out[f"rec_{suf}"] = None
                out[f"n_{suf}"] = None
            return out

        p, r, f1s, s = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average=None,
            zero_division=0,  # overridden to None when support==0
        )

        for lab, pv, rv, fv, sv in zip(labels, p, r, f1s, s):
            suf = _sanitize_col_suffix(lab)
            sv = int(sv)
            if sv == 0:
                out[f"f1_{suf}"] = None
                out[f"prec_{suf}"] = None
                out[f"rec_{suf}"] = None
                out[f"n_{suf}"] = None
            else:
                out[f"f1_{suf}"] = _round2_or_none(fv)
                out[f"prec_{suf}"] = _round2_or_none(pv)
                out[f"rec_{suf}"] = _round2_or_none(rv)
                out[f"n_{suf}"] = sv

        return out

    stacked_raster_dir = Path(stacked_raster_dir)
    classified_rasters_dir = Path(classified_rasters_dir)
    classified_rasters_dir.mkdir(parents=True, exist_ok=True)

    out_root = classified_rasters_dir / str(stack_description) / str(hab_selection)
    out_root.mkdir(parents=True, exist_ok=True)

    model_dir = out_root / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    performance_dfs_dir = out_root / "performance_dfs"
    performance_dfs_dir.mkdir(parents=True, exist_ok=True)

    if train_split_attempt is None or str(train_split_attempt).strip() == "":
        warnings.warn(
            "train_split_attempt is empty/None. Models will be stored in model/_missing_attempt/. "
            "Provide train_split_attempt to avoid mixing attempts.",
            category=UserWarning,
        )
        attempt_dirname = "_missing_attempt"
    else:
        attempt_dirname = str(train_split_attempt).strip()

    attempt_model_dir = model_dir / attempt_dirname
    attempt_model_dir.mkdir(parents=True, exist_ok=True)

    # --- import gpkg ---
    train_gdf = import_gpkg_func(train_gpkg_path)
    val_all = import_gpkg_func(val_gpkg_path)

    # --- CRS check (train_gdf vs expected) ---
    exp_crs, exp_epsg, exp_name = expected_crs_from_input(stacked_raster_crs)
    if train_gdf.crs is None:
        raise ValueError("train_gdf.crs is None")
    if train_gdf.crs != exp_crs:
        raise ValueError(f"CRS mismatch: expected {exp_epsg} ({exp_name}), got {train_gdf.crs}")

    # --- optional warning: unused bands ---
    if warn_on_unused_bands:
        used = set(feature_band_names)
        if valid_band_available and valid_band_name is not None:
            used.add(valid_band_name)
        unused = [b for b in all_raster_band_names if b not in used]
        if unused:
            warnings.warn(
                f"Some raster bands are not used. Unused: {unused}. Used: {sorted(used)}",
                category=UserWarning,
            )

    # --- mask handling ---
    valid_mask_band = valid_band_name if valid_band_available else None

    # --- models ---
    models_dict = RF_models()
    if rf_model_selection not in models_dict:
        raise ValueError(f"Unknown rf_model_selection='{rf_model_selection}'. Options: {list(models_dict)}")
    base_model = models_dict[rf_model_selection]

    # --- validation setup ---
    if "years" not in val_all.columns:
        raise ValueError("Validation GDF must have a column 'years' for year filtering.")
    val_all = val_all.copy()
    val_all["years_str"] = val_all["years"].astype(str).str.strip()

    years_int = [int(y) for y in years]
    results: list[dict] = []

    all_class_labels: list[str] | None = None

    def _append_summary_row(
        *,
        train_year_,
        clas_year_,
        timeframe_,
        n=None,
        acc=None,
        macro_f1=None,
        weighted_f1=None,
        **extra,
    ):
        row = {
            "train_year": str(train_year_),
            "clas_year": str(clas_year_),
            "timeframe": str(timeframe_).upper(),
            "n": n,
            "acc": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        }
        row.update(extra)
        results.append(row)

    for tf in timeframes:
        tf_u = str(tf).upper()

        # --- Check for missing training rasters ---
        missing_years = []
        for y in train_gdf[gpkg_year_col].dropna().unique():
            yr_path = sentinel_path_builder(stacked_raster_dir, int(y), tf, stacked_raster_filename)
            if not yr_path.exists():
                missing_years.append(int(y))

        if len(missing_years) == len(train_gdf[gpkg_year_col].dropna().unique()):    
            print(f"Skip timeframe {tf_u}: no training rasters found for any year in train_gdf.")
            for y in years_int:
                _append_summary_row(train_year_="multi", clas_year_=y, timeframe_=tf_u)
            continue
        elif missing_years:
            print(f"Warning: missing training rasters for years {missing_years} in timeframe {tf_u}. Will skip those samples.")

        # --- Build train_years_str from train_gdf ---
        train_years_str = f"{int(train_gdf[gpkg_year_col].min())}-{int(train_gdf[gpkg_year_col].max())}"

        model_name = (
            f"RF_model__{stack_description}__{hab_selection}__train{train_years_str}__{tf_u}__"
            f"{attempt_dirname}__{rf_model_selection}__{stacked_raster_crs}.joblib"
        )
        model_path = attempt_model_dir / model_name

        # --- Load existing or train new ---
        if model_path.exists() and not overwrite:
            loaded_artifact = joblib.load(model_path)
            if not isinstance(loaded_artifact, dict) or "model" not in loaded_artifact:
                raise ValueError(f"Model artifact at {model_path} is not the expected dict with key 'model'.")
            rf_model = loaded_artifact["model"]

            artifact_feature_bands = loaded_artifact.get("feature_bands", None)
            if artifact_feature_bands is not None and list(artifact_feature_bands) != list(feature_band_names):
                raise ValueError(
                    f"Feature band mismatch for loaded model {model_path}.\n"
                    f"Loaded: {artifact_feature_bands}\n"
                    f"Requested: {list(feature_band_names)}"
                )

            valid_mask_band_to_use = loaded_artifact.get("valid_mask_band", valid_mask_band)
            valid_value_to_use = loaded_artifact.get("valid_value", valid_value)

            class_map_dict = loaded_artifact.get("class_map", None)
            if class_map_dict is None:
                le_classes = loaded_artifact.get("le_classes", None)
                if le_classes is None:
                    raise ValueError(
                        f"Loaded model artifact {model_path} has no 'class_map' or 'le_classes'. "
                        "Cannot decode predictions reliably."
                    )
                class_map_dict = {str(i): cls for i, cls in enumerate(le_classes)}

            n_classes = len(class_map_dict)
            if n_classes >= nodata_out:
                raise ValueError(
                    f"Loaded model has too many classes ({n_classes}) for nodata_out={nodata_out}. "
                    f"Need n_classes <= {nodata_out-1}."
                )

            if all_class_labels is None:
                all_class_labels = [str(v) for v in class_map_dict.values()]

            print(f"Loaded existing model for timeframe {tf_u}: {model_path}")

        else:
            if model_path.exists() and overwrite:
                print(f"Overwriting model for timeframe {tf_u}: {model_path}")
            elif (not model_path.exists()) and overwrite:
                print(f"Training new model for timeframe {tf_u} (overwrite=True): {model_path}")
            else:
                print(f"Training new model for timeframe {tf_u} (model missing): {model_path}")

            rf_model = clone(base_model)

            X_train, y_train, _train_df, _class_map_df, le = sample_training_bands_for_RF(
                gdf=train_gdf,
                stacked_raster_dir=stacked_raster_dir,
                stacked_raster_filename=stacked_raster_filename,
                timeframe=tf,
                year_col=gpkg_year_col,
                label_col=gpkg_label_col,
                band_names=list(all_raster_band_names),
                valid_mask_band=valid_mask_band,
                valid_value=valid_value,
                feature_bands=list(feature_band_names),
                le=None,
                assume_polygons=True,
            )

            # Save training data for inspection
            train_inspect_path = model_path.parent / f"training_data_{tf_u}_inspect.csv"
            _train_df.to_csv(train_inspect_path, index=True)
            print(f"Saved training data to: {train_inspect_path}")
            print(f" -> Samples: {len(_train_df)}, Classes: {_train_df[gpkg_label_col].nunique()}")
            print(f" -> Class distribution:\n{_train_df[gpkg_label_col].value_counts()}")

            if len(X_train) == 0:
                print(f"No valid training samples for timeframe {tf_u}. Skipping model training.")
                for y in years_int:
                    _append_summary_row(train_year_=train_years_str, clas_year_=y, timeframe_=tf_u)
                continue

            n_classes = len(le.classes_)
            if n_classes >= nodata_out:
                raise ValueError(
                    f"Too many classes ({n_classes}) for nodata_out={nodata_out}. "
                    f"Need n_classes <= {nodata_out-1} (codes 0..{nodata_out-1})."
                )

            rf_model.fit(X_train, y_train)
            class_map_dict = {str(i): cls for i, cls in enumerate(le.classes_)}

            if all_class_labels is None:
                all_class_labels = [str(v) for v in class_map_dict.values()]

            train_years_used = sorted(train_gdf[gpkg_year_col].dropna().unique().astype(int).tolist())

            joblib.dump(
                {
                    "model": rf_model,
                    "le_classes": le.classes_.tolist(),
                    "class_map": class_map_dict,
                    "feature_bands": list(feature_band_names),
                    "all_band_names": list(all_raster_band_names),
                    "valid_mask_band": valid_mask_band,
                    "valid_value": valid_value,
                    "train_years": train_years_used,
                    "train_year_range": train_years_str,
                    "timeframe": tf_u,
                    "crs_code": stacked_raster_crs,
                    "stack_description": stack_description,
                    "hab_selection": hab_selection,
                    "train_split_attempt": attempt_dirname,
                },
                model_path,
            )
            print(f"Saved multi-year model (years={train_years_used}): {model_path}")

            valid_mask_band_to_use = valid_mask_band
            valid_value_to_use = valid_value

        # --- Apply to all years (same timeframe) + validate ---
        for y in years_int:
            in_stack_path = sentinel_path_builder(stacked_raster_dir, y, tf, stacked_raster_filename)
            if not in_stack_path.exists():
                print(f"Skip (missing): {in_stack_path}")
                _append_summary_row(train_year_=train_years_str, clas_year_=y, timeframe_=tf_u)
                continue

            out_dir = out_root / f"{int(y)}_{tf_u}"
            out_dir.mkdir(parents=True, exist_ok=True)

            out_name = (
                f"RF_out__{stack_description}__{hab_selection}_{int(y)}_{tf_u}_{attempt_dirname}"
                f"__{stacked_raster_crs}__rstr.tif"
            )
            out_path = out_dir / out_name

            if out_path.exists() and not overwrite:
                print(f"Skip (exists): {out_path}")
            else:
                print(f"Classifying: {in_stack_path} -> {out_path}")
                apply_RF_model_to_rstrs(
                    rf_model=rf_model,
                    in_stack_path=Path(in_stack_path),
                    out_class_path=Path(out_path),
                    expected_band_names=list(all_raster_band_names),
                    valid_mask_band=valid_mask_band_to_use,
                    valid_value=valid_value_to_use,
                    nodata=nodata_out,
                    feature_bands=tuple(feature_band_names),
                )

                # Write tags (CHANGED: use train_years_str)
                with rasterio.open(out_path, "r+") as dst:
                    class_map_json = json.dumps(class_map_dict)
                    dst.update_tags(
                        **{
                            "CLASS-MAP": class_map_json,
                            "CLASS_MAP": class_map_json,
                            "NODATA_VALUE": str(dst.nodata),
                            "TRAIN_YEAR_RANGE": train_years_str,  # CHANGED
                            "TIMEFRAME": tf_u,
                            "MODEL": rf_model_selection,
                            "HAB_DIVISION": hab_selection,
                            "CRS_CODE": stacked_raster_crs,
                            "TRAIN_SPLIT_ATTEMPT": str(attempt_dirname),
                        }
                    )

            # --- Validation (year-filtered) ---
            val_gdf = val_all[val_all["years_str"] == str(y)]
            if val_gdf.empty:
                print(f"Skip (no validation rows): year={y}, timeframe={tf_u}")
                _append_summary_row(train_year_=train_years_str, clas_year_=y, timeframe_=tf_u)
                continue

            y_true, y_pred, _df_cmp = ml_classification_check(
                gdf=val_gdf,
                raster_path=out_path,
                label_col=gpkg_label_col,
                assume_points=False,
                class_map=class_map_dict,
                nodata_value=nodata_out,
            )

            if len(y_true) == 0:
                print(f"Skip (no valid samples after sampling): year={y}, timeframe={tf_u}")
                _append_summary_row(train_year_=train_years_str, clas_year_=y, timeframe_=tf_u)
                continue

            # --- Normalize to strings for sklearn metrics robustness ---
            y_true_s = _as_clean_str_labels(y_true)
            y_pred_s = _as_clean_str_labels(y_pred)
            nmin = min(len(y_true_s), len(y_pred_s))
            y_true_s = y_true_s[:nmin]
            y_pred_s = y_pred_s[:nmin]

            if nmin == 0:
                print(f"Skip (empty after cleaning): year={y}, timeframe={tf_u}")
                _append_summary_row(train_year_=train_years_str, clas_year_=y, timeframe_=tf_u)
                continue

            labels_for_cols = all_class_labels or sorted(set(y_true_s))

            per_acc = _per_class_acc(y_true_s, y_pred_s, labels=labels_for_cols)
            per_prf = _per_class_f1_prec_rec_n(y_true_s, y_pred_s, labels=labels_for_cols)

            _append_summary_row(
                train_year_=train_years_str,
                clas_year_=y,
                timeframe_=tf_u,
                n=int(nmin),
                acc=_round2_or_none(accuracy_score(y_true_s, y_pred_s)),
                macro_f1=_round2_or_none(f1_score(y_true_s, y_pred_s, average="macro", zero_division=0)),
                weighted_f1=_round2_or_none(f1_score(y_true_s, y_pred_s, average="weighted", zero_division=0)),
                **per_acc,
                **per_prf,
            )

    # --- Summary df ordering ---
    summary_df = pd.DataFrame(results)

    base_cols = ["train_year", "clas_year", "timeframe", "n", "acc", "macro_f1", "weighted_f1"]

    # Stable ordering of classes (preferred)
    if all_class_labels is not None:
        class_labels = [str(c) for c in all_class_labels]
        suffixes = [_sanitize_col_suffix(c) for c in class_labels]
    else:
        # Infer suffixes from existing columns (already sanitized)
        suffixes = []
        for col in summary_df.columns:
            for prefix in ("acc_", "f1_", "prec_", "rec_", "n_"):
                if col.startswith(prefix):
                    suffixes.append(col[len(prefix) :])
        suffixes = sorted(set(suffixes))

    per_class_cols: list[str] = []
    for suf in suffixes:
        per_class_cols += [f"acc_{suf}", f"f1_{suf}", f"prec_{suf}", f"rec_{suf}", f"n_{suf}"]

    other_cols = [c for c in summary_df.columns if c not in base_cols and c not in per_class_cols]
    summary_df = summary_df[base_cols + per_class_cols + other_cols]

    # train_years_str in CSV filename
    train_years_for_filename = f"{int(train_gdf[gpkg_year_col].min())}-{int(train_gdf[gpkg_year_col].max())}"
    summary_csv_name = (
        f"RF_validation__{stack_description}__{hab_selection}_{train_split_attempt}__train{train_years_for_filename}__"
        f"{rf_model_selection}__{stacked_raster_crs}.csv"
    )
    summary_df.to_csv(performance_dfs_dir / summary_csv_name, index=False)

    return summary_df