################################################################################
## ML CLASSIFICATION CHECK
################################################################################
from pathlib import Path
import json
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol


def ml_classification_check(
    gdf,
    raster_path,
    label_col,
    *,
    pred_col="y_pred",
    raster_band=1,
    nodata_value=None,
    class_map=None,
    assume_points=False,
    warn_multiband: bool = True,
):
    """
    Sample predicted class values from a classified raster at vector locations and
    return paired ground-truth vs prediction arrays + a comparison DataFrame.

    Key assumptions / requirements
    ------------------------------
    - CRS must match: `gdf` must be in the *same CRS* as the raster. No reprojection
      is performed. If they differ, this function raises a ValueError.
    - It samples one band (`raster_band`, 1-based).
    - Nodata is required: either the raster must have `nodata` set in metadata, or you must
      pass `nodata_value`. Otherwise the function raises an error.
    - Geometries:
        * If `assume_points=True`, geometries are assumed to be Point-like and have `.x/.y`.
        * If `assume_points=False`, non-Point geometries are replaced by their centroid.
      The function does not buffer polygons or do zonal statistics—only point sampling.
    - Memory: the selected raster band is read fully into memory. For very large rasters,
      this can be expensive.
    - Class decoding (optional):
        * If `class_map` is None, the function tries to read raster dataset tag "CLASS-MAP" (JSON).
        * If decoding is performed, any sampled raster values not present in the map are dropped.

    Notes
    -----
    If a multi-band raster is provided, this function will still read and sample only
    `raster_band`. All other bands are ignored (a warning is printed).

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing geometries and a ground-truth label column `label_col`.
        Must have a non-null `.crs`.
    raster_path : str or pathlib.Path
        Path to a raster readable by rasterio.
    label_col : str
        Column name in `gdf` containing ground-truth labels.
    pred_col : str, optional
        Column name to use for predictions in the returned DataFrame. Default "y_pred".
    raster_band : int, optional
        1-based raster band index to sample. Default 1.
    nodata_value : int or float, optional
        Overrides the raster's internal nodata. If None, uses `src.nodata`.
        If both are None -> raises ValueError.
    class_map : dict or None, optional
        Mapping from raster class codes to class labels, e.g. {"0": "WD1", "1": "OW"}.
        If None, the function tries to read raster dataset tag "CLASS-MAP" (JSON). If still None,
        no decoding is done and raw raster values are returned.
    assume_points : bool, optional
        If True, treat geometries as points. If False, non-point geometries are replaced by
        centroids. Default False.
    warn_multiband : bool, optional
        If True, print a warning when the raster has multiple bands and only one is read. Default True.

    Returns
    -------
    y_true : numpy.ndarray
        Ground-truth labels for the kept samples.
    y_pred : numpy.ndarray
        Predictions at sampled locations. If class decoding occurred, these are labels
        (often strings). Otherwise these are raw raster values (often integers).
    out : pandas.DataFrame
        DataFrame with paired columns: `{label_col}` and `{pred_col}`.

    Raises
    ------
    ValueError
        If `gdf.crs` is missing or does not match the raster CRS.
        If no nodata is available from either `nodata_value` or raster metadata.
        If raster tag "CLASS-MAP" exists but cannot be parsed as JSON.
    """
    raster_path = Path(raster_path)

    if gdf is None or len(gdf) == 0:
        out = pd.DataFrame(columns=[label_col, pred_col])
        return np.array([]), np.array([]), out

    # Require CRS on gdf (fail fast)
    if getattr(gdf, "crs", None) is None:
        raise ValueError("gdf.crs is None. Set the GeoDataFrame CRS before sampling (e.g., gdf.set_crs(...)).")

    with rasterio.open(raster_path) as src:
        # CRS check (fail fast)
        if src.crs is None:
            raise ValueError("Raster CRS is None; cannot validate CRS match.")
        if gdf.crs != src.crs:
            raise ValueError(f"CRS mismatch: gdf.crs={gdf.crs!s} vs raster.crs={src.crs!s}. Reproject gdf first.")

        # Warn (but do not fail) on multiband rasters; keep behavior unchanged
        if warn_multiband and src.count > 1:
            read_band = int(raster_band)
            ignored = [b for b in range(1, src.count + 1) if b != read_band]
            ignored_str = ", ".join(map(str, ignored)) if ignored else "(none)"
            print(
                f"Warning: raster has {src.count} bands. "
                f"ml_classification_check() will read band {read_band} and ignore band(s): {ignored_str}."
            )

        pred = src.read(raster_band)
        transform = src.transform
        h, w = src.height, src.width

        nodata = src.nodata if nodata_value is None else nodata_value
        if nodata is None:
            raise ValueError(
                "No nodata value available. Provide `nodata_value=...` or ensure the raster has nodata metadata."
            )

        if class_map is None:
            tags = src.tags()  # dataset-level tags
            if "CLASS-MAP" in tags and tags["CLASS-MAP"]:
                try:
                    class_map = json.loads(tags["CLASS-MAP"])
                except Exception as e:
                    raise ValueError(f'Raster tag "CLASS-MAP" exists but could not be parsed as JSON: {e}') from e

    geom = gdf.geometry
    if not assume_points:
        first_valid = geom.dropna()
        if len(first_valid) == 0:
            out = pd.DataFrame(columns=[label_col, pred_col])
            return np.array([]), np.array([]), out

        if not first_valid.iloc[0].geom_type.lower().endswith("point"):
            geom = geom.centroid

    xs = geom.x.to_numpy(np.float64)
    ys = geom.y.to_numpy(np.float64)

    rows, cols = rowcol(transform, xs, ys)
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)

    inb = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    if not np.any(inb):
        out = pd.DataFrame(columns=[label_col, pred_col])
        return np.array([]), np.array([]), out

    rows_in, cols_in = rows[inb], cols[inb]
    y_true = gdf.loc[inb, label_col].to_numpy()
    y_pred = pred[rows_in, cols_in]

    keep = np.isfinite(y_pred) & (y_pred != nodata)
    y_true = y_true[keep]
    y_pred = y_pred[keep]

    if class_map is not None:
        s = pd.Series(y_pred)
        y_pred_dec = s.astype(str).map(class_map)
        ok = y_pred_dec.notna().to_numpy()
        y_true = y_true[ok]
        y_pred = y_pred_dec.to_numpy()

    out = pd.DataFrame({label_col: y_true, pred_col: y_pred})
    return y_true, y_pred, out