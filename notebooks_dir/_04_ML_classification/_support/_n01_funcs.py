import warnings
import os
warnings.filterwarnings('ignore', message='.*sklearn.utils.parallel.delayed.*')

###############################################################################
## CHECK EXPECTED CRS
###############################################################################
import pyproj

def expected_crs_from_input(crs_code: str):
    """
    Map a short CRS code to an expected CRS object for validation.

    Parameters
    ----------
    crs_code : str
        One of:
        - "UTM32631" : WGS 84 / UTM zone 31N (EPSG:32631)
        - "RD"       : Amersfoort / RD New (EPSG:28992)
        - "WGS84"    : WGS 84 geographic (EPSG:4326)

    Returns
    -------
    expected_crs : pyproj.CRS
        CRS object you can compare to a GeoDataFrame's CRS.
    expected_epsg : int
        EPSG code (when defined).
    expected_name : str
        Human-readable CRS name.

    Raises
    ------
    ValueError
        If `crs_code` is not one of the supported values. Update the "expected_crs_from_input" function.
    """
    code = (crs_code or "").strip().upper()

    mapping = {
        "UTM32631": 32631,
        "RD": 28992,
        "WGS84": 4326,
    }

    if code not in mapping:
        raise ValueError(f"Unsupported crs_code='{crs_code}'. Supported: {list(mapping)}")

    epsg = mapping[code]
    crs = pyproj.CRS.from_epsg(epsg)
    return crs, epsg, crs.name


###############################################################################
## RF MODELS
###############################################################################
from sklearn.ensemble import RandomForestClassifier
import os

def RF_models():
    n_cores = max(1, os.cpu_count() - 2)
    return {
        "mdl1": RandomForestClassifier(
            n_estimators=500, 
            random_state=42, 
            n_jobs=n_cores,
            class_weight="balanced_subsample"
        ),
    }



###############################################################################
## TRAIN AND VAL GDF TO RASTER
###############################################################################
import warnings
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from functions.s2_import_funcs import sentinel_path_builder


def rowcol_func(transform, xs, ys):
    """Vectorized x/y -> (row, col) using an affine transform."""
    Tinv = ~transform
    xs = np.ascontiguousarray(xs, dtype=np.float64)
    ys = np.ascontiguousarray(ys, dtype=np.float64)
    cols_f, rows_f = Tinv * (xs, ys)
    return np.floor(rows_f).astype(np.int64), np.floor(cols_f).astype(np.int64)


def sample_training_bands_for_RF(
    gdf,
    raster_path=None,
    *,
    stacked_raster_dir=None,
    stacked_raster_filename=None,
    timeframe=None,
    year_col="years",
    # Original parameters
    label_col="type",
    band_names=["B02", "B03", "B04", "B08", "VALID_MASK"],
    valid_mask_band="VALID_MASK",
    valid_value=1,
    feature_bands=["B02", "B03", "B04", "B08"],
    le=None,
    assume_polygons=True,
):
    """
    Build Random-Forest training data by sampling a stacked raster at training geometries.

    **Multi-year training mode (NEW)**:
    If `stacked_raster_dir`, `stacked_raster_filename`, and `timeframe` are provided,
    the function will group `gdf` by the `year_col` column and sample each group from
    its corresponding year's raster. The samples are concatenated into a single training set.

    **Single-year mode (original)**:
    If `raster_path` is provided (and multi-year parameters are None), behaves as before:
    samples all rows from one raster.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Training geometries with a class label column (and optionally a year column).
    raster_path : str or pathlib.Path, optional
        Single raster to sample from (single-year mode).
    stacked_raster_dir : str or pathlib.Path, optional
        Base directory for multi-year rasters (e.g., ".../stacked_rasters/").
    stacked_raster_filename : str or pathlib.Path, optional
        Filename of the stacked raster (e.g., "S2_b2348_stack.tif").
    timeframe : str or int, optional
        Quarter (e.g., "Q1") or month (e.g., 11) for multi-year mode.
    year_col : str, default "years"
        Column in `gdf` containing the year for each sample (multi-year mode).
    label_col : str
        Column in `gdf` containing class labels (strings or ints).
    band_names : list[str]
        Names for each raster band (same order as in raster).
    valid_mask_band : str | int | None
        If str, interpreted as band name in `band_names`.
        If int, interpreted as 0-based band index.
        If None, no valid-mask filtering is applied.
    valid_value : int | float
        Value that indicates valid pixels in the valid mask band.
    feature_bands : list[str]
        Band names to use as RF features (subset of `band_names`).
    le : sklearn.preprocessing.LabelEncoder | None
        If provided, uses it (fits it if not already fit). If None, creates and fits a new one.
    assume_polygons : bool
        If True (default), samples centroids (polygon-style workflow).

    Returns
    -------
    X_train : numpy.ndarray, shape (n_samples, n_features)
        Feature values for `feature_bands`.
    y_train : numpy.ndarray, shape (n_samples,)
        Integer class numbers (LabelEncoder output).
    out_df : pandas.DataFrame
        Sample table indexed by the **original gdf index** (filtered to kept samples),
        containing all sampled bands, `label_col`, and `class_no`.
    class_map_df : pandas.DataFrame
        Two-column mapping: `class_no` -> `typology` (original label values).
    le : sklearn.preprocessing.LabelEncoder
        The fitted label encoder used to produce `class_no`.
    """

    # --- Determine mode ---
    multi_year_mode = (stacked_raster_dir is not None 
                       and stacked_raster_filename is not None 
                       and timeframe is not None)
    single_year_mode = (raster_path is not None)

    if multi_year_mode == single_year_mode:
        raise ValueError(
            "Provide either:\n"
            "  - raster_path (single-year mode), OR\n"
            "  - stacked_raster_dir + stacked_raster_filename + timeframe (multi-year mode)"
        )

    # --- Multi-year mode: loop by year and concatenate ---
    if multi_year_mode:
        if year_col not in gdf.columns:
            raise ValueError(f"Multi-year mode requires a '{year_col}' column in gdf.")

        gdf = gdf.copy()
        gdf["_year_int"] = gdf[year_col].astype(int)

        if le is None:
            le = LabelEncoder()
            le.fit(gdf[label_col].values)  # Fit on all years at once!
        elif not hasattr(le, "classes_"):
            le.fit(gdf[label_col].values)

        X_parts = []
        y_parts = []
        out_df_parts = []

        for year in sorted(gdf["_year_int"].unique()):
            gdf_year = gdf[gdf["_year_int"] == year].copy()

            # Build raster path for this year
            year_raster_path = sentinel_path_builder(
                base_dir=stacked_raster_dir,
                year=int(year),
                quarter_or_month=timeframe,
                filename=stacked_raster_filename,
            )

            if not year_raster_path.exists():
                warnings.warn(
                    f"Missing raster for year={year}, timeframe={timeframe}: {year_raster_path}. "
                    f"Skipping {len(gdf_year)} samples."
                )
                continue

            # Sample this year's subset (recursive call in single-year mode)
            X_y, y_y, out_df_y, _class_map_y, le = sample_training_bands_for_RF(
                gdf=gdf_year,
                raster_path=year_raster_path,
                label_col=label_col,
                band_names=band_names,
                valid_mask_band=valid_mask_band,
                valid_value=valid_value,
                feature_bands=feature_bands,
                le=le,
                assume_polygons=assume_polygons,
            )

            if len(X_y) > 0:
                X_parts.append(X_y)
                y_parts.append(y_y)
                out_df_parts.append(out_df_y)

        if not X_parts:
            # No samples collected
            out_df = pd.DataFrame(
                columns=[*band_names, label_col, "class_no"]
            ).set_index(pd.Index([], name=gdf.index.name))
            class_map_df = pd.DataFrame(columns=["class_no", "typology"])
            return (
                np.empty((0, len(feature_bands))),
                np.array([]),
                out_df,
                class_map_df,
                (le or LabelEncoder()),
            )

        X_train = np.vstack(X_parts)
        y_train = np.concatenate(y_parts)
        out_df = pd.concat(out_df_parts, axis=0)

        class_map_df = pd.DataFrame(
            {"class_no": np.arange(len(le.classes_), dtype=int), "typology": le.classes_}
        )

        return X_train, y_train, out_df, class_map_df, le

    # --- Single-year mode (original logic) ---
    # resolve valid mask band index (0-based)
    if isinstance(valid_mask_band, str):
        if valid_mask_band not in band_names:
            raise ValueError(f"valid_mask_band='{valid_mask_band}' not in band_names: {band_names}")
        valid_mask_band_idx0 = band_names.index(valid_mask_band)
    elif valid_mask_band is None:
        valid_mask_band_idx0 = None
    else:
        valid_mask_band_idx0 = int(valid_mask_band)

    # read raster
    with rasterio.open(raster_path) as src:
        arr = src.read(masked=True)  # (bands, h, w)
        transform = src.transform
        h, w = src.height, src.width
        nodata = src.nodata

    if np.ma.isMaskedArray(arr):
        fill = nodata if nodata is not None else -9999
        arr = arr.filled(fill)

    # choose sampling geometries (default: centroids)
    geom = gdf.geometry
    if assume_polygons:
        geom = geom.centroid
    else:
        first_valid = geom.dropna()
        if len(first_valid) > 0 and (not first_valid.iloc[0].geom_type.lower().endswith("point")):
            geom = geom.centroid

    xs = geom.x.to_numpy(np.float64)
    ys = geom.y.to_numpy(np.float64)

    # x/y -> row/col
    rows, cols = rowcol_func(transform, xs, ys)
    inb = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)

    idx_inb = gdf.index.to_numpy()[inb]
    rows_inb, cols_inb = rows[inb], cols[inb]
    y_lab_inb = gdf.loc[idx_inb, label_col].to_numpy()

    if len(idx_inb) == 0:
        out_df = pd.DataFrame(
            columns=[*band_names, label_col, "class_no"]
        ).set_index(pd.Index([], name=gdf.index.name))
        class_map_df = pd.DataFrame(columns=["class_no", "typology"])
        return (
            np.empty((0, len(feature_bands))),
            np.array([]),
            out_df,
            class_map_df,
            (le or LabelEncoder()),
        )

    # sample all raster bands at points
    samp = arr[:, rows_inb, cols_inb].T  # (n, bands)

    keep = np.isfinite(samp).all(axis=1)
    if nodata is not None:
        keep &= ~(samp == nodata).any(axis=1)
    if valid_mask_band_idx0 is not None:
        keep &= (samp[:, valid_mask_band_idx0] == valid_value)

    idx_keep = idx_inb[keep]
    samp_keep = samp[keep]
    y_lab_keep = y_lab_inb[keep]

    out_df = pd.DataFrame(samp_keep, columns=band_names, index=idx_keep)
    out_df[label_col] = y_lab_keep

    # label encoding
    if le is None:
        le = LabelEncoder()
        y_train = le.fit_transform(y_lab_keep)
    else:
        if not hasattr(le, "classes_"):
            y_train = le.fit_transform(y_lab_keep)
        else:
            y_train = le.transform(y_lab_keep)

    out_df["class_no"] = y_train

    class_map_df = pd.DataFrame(
        {"class_no": np.arange(len(le.classes_), dtype=int), "typology": le.classes_}
    )

    missing = [b for b in feature_bands if b not in out_df.columns]
    if missing:
        raise ValueError(f"Missing feature bands in raster/band_names: {missing}")

    X_train = out_df.loc[:, list(feature_bands)].to_numpy()

    return X_train, y_train, out_df, class_map_df, le



###############################################################################
## APPLY RF TRAINING ON ALL RASTERS
###############################################################################
from pathlib import Path
import rasterio
import numpy as np

def apply_RF_model_to_rstrs(
    rf_model,
    in_stack_path: Path,
    out_class_path: Path,
    expected_band_names=["B02","B03","B04","B08","VALID_MASK"],
    *,
    valid_mask_band="VALID_MASK",
    valid_value: int = 1,
    nodata: int = 0,
    block_size: int = 512,
    feature_bands=["B02","B03","B04","B08"]  # can also be a tuple of names
) -> Path:
    """
    Apply a trained scikit-learn RandomForest model to a stacked multi-band raster and
    write a 1-band classified raster (tiled, block-wise).

    The function reads the raster in blocks, extracts `feature_bands` pixels for locations
    that are valid according to `valid_mask_band`, predicts classes with `rf_model`, and
    writes the predicted class codes to `out_class_path`. Pixels that are not valid (or not
    processed) are written as `nodata`.

    Parameters
    ----------
    rf_model
        A fitted scikit-learn estimator supporting `predict(X)` where X is (n_samples, n_features).
        Typically a `RandomForestClassifier`.
    in_stack_path : pathlib.Path
        Input raster path. Must be a multi-band stack containing at least the bands listed in
        `feature_bands` and optionally a validity mask band.
    out_class_path : pathlib.Path
        Output path for the classified raster (single band).
    expected_band_names : tuple[str, ...]
        Expected band names/order for the input raster. If the raster has band descriptions
        (`src.descriptions`) for all bands, they must match this exactly or a ValueError is raised.
        If descriptions are missing, the function falls back to this tuple for name->index mapping.
    valid_mask_band : str | int | None
        Validity mask band selector:
        - str: band name resolved via raster band descriptions (or `expected_band_names` fallback)
        - int: 1-based raster band number (rasterio convention)
        - None: disables validity masking (all pixels processed)
    valid_value : int
        Pixel value in the validity mask band that indicates "valid" pixels to classify.
    nodata : int
        The nodata value written to the OUTPUT classified raster and used to fill pixels that are
        not classified (e.g., invalid mask pixels). Also stored in the output raster metadata as
        the dataset nodata value.
    block_size : int
        Tile/block size (in pixels) used for processing and for the output tiling metadata.
    feature_bands : tuple[str | int, ...]
        Bands used as model features. Each entry can be:
        - str: band name (resolved via descriptions / fallback mapping)
        - int: 1-based raster band number

    Returns
    -------
    pathlib.Path
        The `out_class_path` that was written.

    Raises
    ------
    ValueError
        - If input band descriptions exist but do not match `expected_band_names`.
        - If `valid_mask_band` (name) is not found.
        - If any `feature_bands` (names) are not found.

    Notes
    -----
    - This function does not currently treat input raster nodata specially when predicting.
      If your feature bands contain nodata pixels, those values will be passed into the model
      unless you mask them out via `valid_mask_band` or add additional filtering.
    - Output dtype is `uint8`. Ensure your predicted class codes fit in 0..255.
    """
    # Warning suppression
    import warnings
    warnings.filterwarnings(
        "ignore",
        message=r".*sklearn\.utils\.parallel\.delayed.*",
        category=UserWarning,
    )

    # Start of the function
    out_class_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(in_stack_path) as src:
        # --- band name source (prefer descriptions, fallback to expected_band_names) ---
        desc = list(src.descriptions) if src.descriptions else []
        have_desc = len(desc) == src.count and any(d is not None for d in desc)

        if have_desc and all(d is not None for d in desc):
            if tuple(desc) != tuple(expected_band_names):
                raise ValueError(
                    f"Unexpected band order in {in_stack_path}\n"
                    f"Expected: {expected_band_names}\n"
                    f"Got:      {tuple(desc)}"
                )
            band_list = desc  # use actual names from file
        else:
            print(
                f"Warning: raster band descriptions missing in {in_stack_path}. "
                "Falling back to expected_band_names for name->band mapping."
            )
            band_list = list(expected_band_names)

        # --- resolve valid mask band to 1-based band number for rasterio ---
        if isinstance(valid_mask_band, str):
            if valid_mask_band not in band_list:
                raise ValueError(f"valid_mask_band '{valid_mask_band}' not found. Available: {band_list}")
            vm_band_1 = band_list.index(valid_mask_band) + 1
        elif valid_mask_band is None:
            vm_band_1 = None
        else:
            vm_band_1 = int(valid_mask_band)

        # --- resolve feature bands to 1-based band numbers ---
        feature_band_nums_1 = []
        for b in feature_bands:
            if isinstance(b, str):
                if b not in band_list:
                    raise ValueError(f"Feature band '{b}' not found. Available: {band_list}")
                feature_band_nums_1.append(band_list.index(b) + 1)
            else:
                feature_band_nums_1.append(int(b))

        profile = src.profile.copy()
        profile.update(
            count=1,
            dtype=rasterio.uint8,
            nodata=nodata,
            compress="lzw",
            tiled=True,
            blockxsize=min(block_size, src.width),
            blockysize=min(block_size, src.height),
        )

        with rasterio.open(out_class_path, "w", **profile) as dst:
            for _, window in src.block_windows(1):
                arr_feats = src.read(feature_band_nums_1, window=window)  # (4, h, w)

                if vm_band_1 is not None:
                    vm = src.read(vm_band_1, window=window)  # (h, w)
                    valid = (vm == valid_value)
                else:
                    valid = np.ones((arr_feats.shape[1], arr_feats.shape[2]), dtype=bool)

                out = np.full((arr_feats.shape[1], arr_feats.shape[2]), nodata, dtype=np.uint8)

                if valid.any():
                    feats = np.moveaxis(arr_feats, 0, -1)  # (h, w, 4)
                    X = feats[valid].reshape(-1, len(feature_bands))
                    pred_enc = rf_model.predict(X)
                    out[valid] = pred_enc.astype(np.uint8)

                dst.write(out, 1, window=window)

    return out_class_path