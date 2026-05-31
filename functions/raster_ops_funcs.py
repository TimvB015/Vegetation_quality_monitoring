from __future__ import annotations

################################################################################
## CLIP WITH RASTERIO
################################################################################
import os
from typing import Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask


def clip_with_rasterio_func(
    raster_path: str,
    gpkg_path: str,
    out_dir: Optional[str] = None,
    out_filename: Optional[str] = None,
    overwrite: bool = False,
    layer: Optional[str] = None,
    dissolve: bool = True,
    all_touched: bool = False,
    nodata: Optional[Union[int, float]] = -9999,
    blocksize: int = 512,
    include_mask_layer: bool = False,
    mask_as_alpha: bool = False,
) -> Tuple[str, Optional[int], str]:
    """
    Clip a raster to vector geometry from a GeoPackage using `rasterio.mask.mask`
    and write a cropped GeoTIFF.

    This function:
      1) reads geometries from a GeoPackage (optionally selecting a layer)
      2) drops empty/None geometries and attempts to repair invalid geometries
      3) reprojects geometries to the raster CRS if needed
      4) optionally dissolves all features into one clip geometry
      5) clips the raster with `rasterio.mask.mask(..., filled=False)` to preserve a
         pixel-validity mask
      6) fills pixels outside the clip geometry with `nodata`
      7) writes a tiled, DEFLATE-compressed GeoTIFF, optionally appending a uint8
         validity mask band

    Output details
    --------------
    - Output format: GeoTIFF (GTiff), tiled, DEFLATE-compressed.
    - Output extent: cropped to the geometry bounds (`crop=True`).
    - Output nodata:
        * If `nodata` is not None, that value is used and written to metadata.
        * If `nodata` is None, the source raster nodata is used (if present),
          otherwise falls back to -9999.
    - If `include_mask_layer=True`, an extra band is appended as the last band:
        * dtype: uint8
        * values: 1 = valid (inside clip geometry), 0 = invalid (outside)
      If `mask_as_alpha=True`, this last band is marked as an alpha band where supported.

    Parameters
    ----------
    raster_path : str
        Path to the input raster (e.g., GeoTIFF).
    gpkg_path : str
        Path to a GeoPackage containing clip geometries.
    out_dir : Optional[str], default None
        Output directory. If None, output is written next to `raster_path`.
    out_filename : Optional[str], default None
        Output filename (e.g., "my_clip.tif"). If not provided, defaults to
        "<input_basename>_clipped.tif".
    overwrite : bool, default False
        If False and the output already exists, nothing is written and the function
        returns status "exists".
    layer : Optional[str], default None
        GeoPackage layer name to read. If None, the default layer is used.
    dissolve : bool, default True
        If True, dissolve all features into a single geometry before clipping (often faster
        and avoids seams between multiple polygons).
    all_touched : bool, default False
        Passed to `rasterio.mask.mask`. If True, include all pixels touched by geometries
        (not only those whose center lies within the geometry).
    nodata : Optional[Union[int, float]], default -9999
        Nodata value used to fill pixels outside the clip geometry and written into
        output metadata. If explicitly set to None, the function attempts to use the
        source raster nodata; if still None, it falls back to -9999.
    blocksize : int, default 512
        Internal tile size for the output GeoTIFF (`blockxsize`/`blockysize`).
        Commonly best as a multiple of 16 (512 is a safe default).
    include_mask_layer : bool, default False
        If True, append a uint8 validity mask band (1=inside clip geometry, 0=outside)
        as the last band.
    mask_as_alpha : bool, default False
        If True and `include_mask_layer=True`, mark the appended mask band as an alpha band
        (color interpretation), where supported by the driver/software.

    Returns
    -------
    (out_path, epsg, status) : Tuple[str, Optional[int], str]
        out_path : str
            Output raster path.
        epsg : Optional[int]
            EPSG code derived from the input raster CRS, or None if unavailable.
        status : str
            "written" if a new file was written, or "exists" if the output already existed
            and `overwrite=False`.

    Raises
    ------
    FileNotFoundError
        If `raster_path` or `gpkg_path` does not exist.
    ValueError
        If the GeoPackage contains no usable geometries, if the raster CRS is missing,
        or if the vector CRS is missing.
    """

    if not os.path.exists(raster_path):
        raise FileNotFoundError(f"Raster not found: {raster_path}")
    if not os.path.exists(gpkg_path):
        raise FileNotFoundError(f"GeoPackage not found: {gpkg_path}")

    # Determine output path
    if out_filename:
        if out_dir is None:
            out_dir = os.path.dirname(raster_path)
        os.makedirs(out_dir, exist_ok=True)
        final_out = os.path.join(out_dir, out_filename)
    else:
        base = os.path.splitext(os.path.basename(raster_path))[0]
        filename = f"{base}_clipped.tif"
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            final_out = os.path.join(out_dir, filename)
        else:
            final_out = os.path.join(os.path.dirname(raster_path), filename)

    if os.path.exists(final_out) and not overwrite:
        with rasterio.open(raster_path) as src:
            src_crs = src.crs
            epsg = int(src_crs.to_epsg()) if (src_crs and src_crs.to_epsg() is not None) else None
        return final_out, epsg, "exists"

    vec = gpd.read_file(gpkg_path, layer=layer) if layer else gpd.read_file(gpkg_path)
    if vec.empty:
        raise ValueError("GeoPackage contains no features.")

    # parentheses for boolean ops
    vec = vec[(~vec.geometry.is_empty) & (vec.geometry.notna())].copy()
    if vec.empty:
        raise ValueError("No non-empty geometries found in GeoPackage.")

    # Fix invalid geometries (only where invalid)
    invalid = ~vec.geometry.is_valid
    if invalid.any():
        try:
            vec.loc[invalid, "geometry"] = vec.loc[invalid, "geometry"].make_valid()
        except Exception:
            vec.loc[invalid, "geometry"] = vec.loc[invalid, "geometry"].buffer(0)

        vec = vec[(~vec.geometry.is_empty) & (vec.geometry.notna())].copy()
        vec = vec[vec.geometry.is_valid].copy()
        if vec.empty:
            raise ValueError("No valid geometries remain after attempting to fix invalid geometries.")

    with rasterio.open(raster_path) as src:
        src_crs = src.crs
        if src_crs is None:
            raise ValueError("Raster has no CRS defined.")
        if vec.crs is None:
            raise ValueError("GeoPackage has no CRS defined.")
        if vec.crs != src_crs:
            vec = vec.to_crs(src_crs)

        epsg = int(src_crs.to_epsg()) if (src_crs.to_epsg() is not None) else None

        if dissolve:
            try:
                geom = vec.geometry.union_all()
            except Exception:
                geom = vec.unary_union
            geoms = [geom.__geo_interface__]
        else:
            geoms = [g.__geo_interface__ for g in vec.geometry]

        out_ma, out_transform = mask(
            src,
            geoms,
            crop=True,
            all_touched=all_touched,
            filled=False,
        )

        if not isinstance(out_ma, np.ma.MaskedArray):
            out_ma = np.ma.array(out_ma)

        out_nodata = nodata if nodata is not None else src.nodata
        if out_nodata is None:
            out_nodata = -9999

        # 1=valid inside clip (across all bands), 0=invalid
        if np.isscalar(out_ma.mask):
            valid_mask_2d = (0 if out_ma.mask else 1) * np.ones(
                (out_ma.shape[1], out_ma.shape[2]), dtype=np.uint8
            )
        else:
            valid_mask_2d = (~np.any(out_ma.mask, axis=0)).astype(np.uint8)

        out_image = out_ma.filled(out_nodata)

        out_count = int(out_image.shape[0]) + (1 if include_mask_layer else 0)

        out_meta = src.meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": int(out_image.shape[1]),
                "width": int(out_image.shape[2]),
                "transform": out_transform,
                "compress": "DEFLATE",
                "tiled": True,
                "blockxsize": int(blocksize),
                "blockysize": int(blocksize),
                "count": out_count,
                "nodata": out_nodata,
            }
        )

        with rasterio.open(final_out, "w", **out_meta) as dst:
            for b in range(out_image.shape[0]):
                dst.write(out_image[b, :, :], b + 1)

            if include_mask_layer:
                mask_band_index = out_image.shape[0] + 1
                dst.write(valid_mask_2d, mask_band_index)

                if mask_as_alpha:
                    try:
                        ci = list(dst.colorinterp)
                        ci[mask_band_index - 1] = rasterio.enums.ColorInterp.alpha
                        dst.colorinterp = tuple(ci)
                    except Exception:
                        pass

    return final_out, epsg, "written"


def get_default_alpha_legend_kwargs(cell_figsize: tuple[float, float]) -> dict:
    """Get default kwargs for alpha legend based on cell size."""
    return {
        'cell_figsize': cell_figsize,
        'box_height': 0.5,
        'font_size': 8,
        'text_color': 'black',
        'tile_size': 0.025,
        'tile_spacing': 0.01,
        'show_class_names': True,
        'class_name_font_size': 8,
        'row_spacing_pts': 12.0,
        'column_spacing': 0.33,
        'class_name_overrides': None,
        'stability_labels': None,
        'horizontal_offset_pts': 0.0,
    }


###############################################################################
## CLIP WITH RASTERIO (RETURN IN-MEMORY)
###############################################################################
from pathlib import Path
from typing import Optional, Tuple, Union, Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.io import DatasetReader
from rasterio.mask import mask


RasterOrPath = Union[str, Path, DatasetReader]
GpkgOrPath = Union[str, Path, gpd.GeoDataFrame]
Number = Union[int, float]


def clip_with_rasterio_and_return(
    raster_or_path: RasterOrPath,
    gpkg_or_path: GpkgOrPath,
    layer: Optional[str] = None,
    dissolve: bool = True,
    all_touched: bool = False,
    nodata: Optional[Number] = -9999,
    blocksize: int = 512,
    include_mask_layer: bool = False,
    mask_as_alpha: bool = False,
) -> Tuple[np.ndarray, dict, Optional[int]]:
    """
    Clip a raster to vector geometry using `rasterio.mask.mask` and return the result
    in memory (no file is written).

    Inputs may be either paths or already-loaded objects:
    - `raster_or_path`: a rasterio dataset (open) or a path to a raster file
    - `gpkg_or_path`: a GeoDataFrame or a path to a GeoPackage (optionally with `layer`)

    The function:
      1) loads/validates vector geometries (drops empty/None, repairs invalid)
      2) reprojects vectors to the raster CRS if needed
      3) optionally dissolves vectors into one geometry
      4) clips using `rasterio.mask.mask(..., filled=False)` to preserve a mask
      5) fills pixels outside the clip geometry with `nodata`
      6) optionally appends a uint8 validity mask band (1=inside, 0=outside)

    Parameters
    ----------
    raster_or_path
        Rasterio dataset (opened) or path (str/Path) to the input raster.
    gpkg_or_path
        GeoDataFrame or path (str/Path) to a GeoPackage containing clip geometries.
    layer
        GeoPackage layer name (only used when `gpkg_or_path` is a path).
    dissolve
        If True, dissolve all features into a single clip geometry.
    all_touched
        Passed to `rasterio.mask.mask`. If True, include all touched pixels.
    nodata
        Output nodata value. If None, uses source nodata if present; otherwise -9999.
    blocksize
        Kept for signature parity with the “write” variant; not used for in-memory output.
    include_mask_layer
        If True, append an extra uint8 mask band (1=valid, 0=invalid) as last band.
    mask_as_alpha
        Kept for signature parity with the “write” variant; no file is written, so
        alpha band metadata is not applied.

    Returns
    -------
    (out_image, out_meta, epsg) : (np.ndarray, dict, Optional[int])
        out_image : np.ndarray
            Array of shape (count, rows, cols). If `include_mask_layer=True`, the last
            band is the uint8 validity mask.
        out_meta : dict
            Rasterio-style metadata for the clipped raster (driver removed), including:
            height, width, transform, crs, dtype, count, nodata.
        epsg : Optional[int]
            EPSG code derived from raster CRS, or None if unavailable.

    Notes
    -----
    - If you pass an already-open raster dataset, this function will NOT close it.
      If a path is passed, the dataset is opened internally and closed automatically.
    - `mask_as_alpha` is only meaningful when writing a file (color interpretation);
      for an in-memory array you already have the mask band.
    """
    # --- Load vector data ---
    if isinstance(gpkg_or_path, gpd.GeoDataFrame):
        vec = gpkg_or_path.copy()
    else:
        gpkg_path = str(gpkg_or_path)
        vec = gpd.read_file(gpkg_path, layer=layer) if layer else gpd.read_file(gpkg_path)

    if vec.empty:
        raise ValueError("Vector data contains no features.")

    vec = vec[(~vec.geometry.is_empty) & (vec.geometry.notna())].copy()
    if vec.empty:
        raise ValueError("No non-empty geometries found.")

    invalid = ~vec.geometry.is_valid
    if invalid.any():
        try:
            vec.loc[invalid, "geometry"] = vec.loc[invalid, "geometry"].make_valid()
        except Exception:
            vec.loc[invalid, "geometry"] = vec.loc[invalid, "geometry"].buffer(0)

        vec = vec[(~vec.geometry.is_empty) & (vec.geometry.notna())].copy()
        vec = vec[vec.geometry.is_valid].copy()
        if vec.empty:
            raise ValueError("No valid geometries remain after attempting to fix invalid geometries.")

    # --- Open raster if needed ---
    def _process(src: DatasetReader) -> Tuple[np.ndarray, dict, Optional[int]]:
        src_crs = src.crs
        if src_crs is None:
            raise ValueError("Raster has no CRS defined.")
        if vec.crs is None:
            raise ValueError("Vector data has no CRS defined.")

        vec_in = vec
        if vec_in.crs != src_crs:
            vec_in = vec_in.to_crs(src_crs)

        epsg = int(src_crs.to_epsg()) if (src_crs.to_epsg() is not None) else None

        if dissolve:
            try:
                geom = vec_in.geometry.union_all()
            except Exception:
                geom = vec_in.unary_union
            geoms = [geom.__geo_interface__]
        else:
            geoms = [g.__geo_interface__ for g in vec_in.geometry]

        out_ma, out_transform = mask(
            src,
            geoms,
            crop=True,
            all_touched=all_touched,
            filled=False,
        )

        if not isinstance(out_ma, np.ma.MaskedArray):
            out_ma = np.ma.array(out_ma)

        out_nodata = nodata if nodata is not None else src.nodata
        if out_nodata is None:
            out_nodata = -9999

        # 1=valid inside clip (across all bands), 0=invalid
        if np.isscalar(out_ma.mask):
            valid_mask_2d = (0 if out_ma.mask else 1) * np.ones(
                (out_ma.shape[1], out_ma.shape[2]), dtype=np.uint8
            )
        else:
            valid_mask_2d = (~np.any(out_ma.mask, axis=0)).astype(np.uint8)

        out_image = out_ma.filled(out_nodata)

        if include_mask_layer:
            out_image = np.concatenate([out_image, valid_mask_2d[None, :, :]], axis=0)

        out_meta = src.meta.copy()
        # make this “in-memory friendly”
        out_meta.pop("driver", None)
        out_meta.update(
            {
                "height": int(out_image.shape[1]),
                "width": int(out_image.shape[2]),
                "transform": out_transform,
                "count": int(out_image.shape[0]),
                "nodata": out_nodata,
            }
        )
        return out_image, out_meta, epsg

    if hasattr(raster_or_path, "read") and hasattr(raster_or_path, "crs"):
        # assume rasterio dataset-like
        return _process(raster_or_path)  # type: ignore[arg-type]
    else:
        raster_path = str(raster_or_path)
        with rasterio.open(raster_path) as src:
            return _process(src)



###############################################################################
## COMPARE RASTER ALIGNMENT
###############################################################################
import numpy as np
import rasterio

def compare_alignment_tifs(tifs, tol=0.0):
    """
    Compare alignment for an arbitrary list of rasters.

    Parameters
    ----------
    tifs : list[str] | tuple[str, ...]
        Iterable of raster paths (must contain at least 2).
    tol : float
        tol=0.0  -> strict equality for transform/res/bounds
        tol>0.0  -> allow small float differences in transform/res/bounds

    Returns
    -------
    bool
        True if ALL rasters align exactly (relative to the first raster), else False.
    """
    if not isinstance(tifs, (list, tuple)) or len(tifs) < 2:
        raise ValueError("tifs must be a list/tuple of at least 2 raster paths")

    def info(p):
        with rasterio.open(p) as ds:
            return {
                "path": p,
                "crs": ds.crs,
                "transform": ds.transform,
                "width": ds.width,
                "height": ds.height,
                "count": ds.count,
                "dtype": ds.dtypes,
                "bounds": ds.bounds,
                "res": ds.res,
            }

    rasters = [info(p) for p in tifs]
    ref = rasters[0]

    def eq_transform(a, b):
        if tol == 0.0:
            return a == b
        return np.allclose(np.array(a), np.array(b), atol=tol, rtol=0)

    def eq_tuple(a, b):
        if tol == 0.0:
            return a == b
        return np.allclose(np.array(a, dtype=float), np.array(b, dtype=float), atol=tol, rtol=0)

    all_ok = True
    per_raster_results = []

    for r in rasters[1:]:
        same = {}
        same["crs"] = (r["crs"] == ref["crs"])
        same["shape"] = (r["width"] == ref["width"] and r["height"] == ref["height"])
        same["res"] = eq_tuple(r["res"], ref["res"])
        same["bounds"] = eq_tuple(r["bounds"], ref["bounds"])
        same["transform"] = eq_transform(r["transform"], ref["transform"])

        aligns = all(same.values())
        all_ok = all_ok and aligns

        per_raster_results.append((r["path"], same, aligns))

    # --- printing ---
    print("=== Reference Raster (0) ===")
    for k in ["path", "crs", "width", "height", "res", "bounds", "transform"]:
        print(f"{k}: {ref[k]}")

    for i, (path, same, aligns) in enumerate(per_raster_results, start=1):
        r = next(x for x in rasters if x["path"] == path)
        print(f"\n=== Raster {i} ===")
        for k in ["path", "crs", "width", "height", "res", "bounds", "transform"]:
            print(f"{k}: {r[k]}")

        print("\n--- Comparison vs reference ---")
        for k, v in same.items():
            print(f"{k}: {v}")
        print("RESULT:", "ALIGNS" if aligns else "DOES NOT ALIGN")

    print("\nFINAL RESULT:", "ALL ALIGN EXACTLY" if all_ok else "NOT ALL ALIGN EXACTLY")
    return all_ok



###############################################################################
## STICH MULTIPLE RASTERS
###############################################################################
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.merge import merge


def stich_multiple_rasters_func(
    sources,
    out_path,
    nodata=None,
    method="first",
    compress="deflate",
    tiled=True,
    blocksize=512,
    nodata_mode="legacy",
    check_grid=False,         
    check_alignment=True,
    res_tol=1e-9,
    bigtiff="IF_SAFER",
):
    """
    Build a mosaic from multiple in-memory rasters (numpy stacks).

    Parameters
    ----------
    sources : list
        List of tuples: (stack, transform, crs)
        where stack has shape (bands, H, W).
    out_path : str
        Output GeoTIFF path.
    nodata : number or None
        Nodata value for merge + output. If None, tries to infer NaN nodata for float stacks.
    method : str
        Merge method passed to rasterio.merge.merge (e.g., "first", "last", "max").
    compress : str
        GeoTIFF compression (e.g., "deflate", "lzw").
    tiled : bool
        Write tiled GeoTIFF.
    blocksize : int
        Tile size (blockxsize=blockysize=blocksize).
    check_alignment : bool
        If True, checks that rasters are aligned on the same pixel grid (origin alignment).
    res_tol : float
        Tolerance for comparing resolutions/transforms.

    Returns
    -------
    out_path, out_transform, out_crs
    """

    def _is_north_up(transform, tol=res_tol):
        return abs(transform.b) <= tol and abs(transform.d) <= tol

    def _same_resolution(t1, t2, tol=res_tol):
        return abs(t1.a - t2.a) <= tol and abs(t1.e - t2.e) <= tol

    def _aligned_on_grid(t_ref, t, tol=res_tol):
        px = t_ref.a
        py = t_ref.e  # typically negative
        if abs(px) <= tol or abs(py) <= tol:
            return False
        dx = (t.c - t_ref.c) / px
        dy = (t.f - t_ref.f) / py
        return (abs(dx - round(dx)) <= 1e-6) and (abs(dy - round(dy)) <= 1e-6)

    def _infer_nodata_effective():
        """
        Auto mode heuristic:
          - if nodata is a number -> use it
          - if nodata is None:
              - if any float stack contains NaN -> nodata_effective = np.nan
              - else -> nodata_effective = None (no nodata masking)
        """
        if nodata is not None:
            return nodata

        any_nan = False
        for stk, _, _ in sources:
            if np.issubdtype(stk.dtype, np.floating) and np.isnan(stk).any():
                any_nan = True
                break
        return np.nan if any_nan else None

    def mem_ds_from_stack(stack, transform, crs, nodata_value):
        if stack.ndim != 3:
            raise ValueError(f"Expected stack with shape (bands,H,W). Got {stack.shape}")
        mem = MemoryFile()
        ds = mem.open(
            driver="GTiff",
            height=stack.shape[1],
            width=stack.shape[2],
            count=stack.shape[0],
            dtype=stack.dtype,
            crs=crs,
            transform=transform,
            nodata=nodata_value,
        )
        ds.write(stack)
        return mem, ds

    if len(sources) == 0:
        raise ValueError("sources is empty")

    # Decide nodata behavior (legacy vs auto)
    if nodata_mode not in ("legacy", "auto"):
        raise ValueError("nodata_mode must be 'legacy' or 'auto'")

    nodata_effective = nodata if nodata_mode == "legacy" else _infer_nodata_effective()

    # Basic validation (same CRS + band count) - kept
    ref_bands = sources[0][0].shape[0]
    ref_crs = sources[0][2]
    ref_tr = sources[0][1]

    for i, (stk, tr, crs) in enumerate(sources):
        if stk.shape[0] != ref_bands:
            raise ValueError(f"Source {i} band count {stk.shape[0]} != {ref_bands}")
        if crs != ref_crs:
            raise ValueError(
                f"Source {i} CRS {crs} != {ref_crs}. Reproject first (or extend function)."
            )

    # Optional stronger grid checks (off by default for backwards compatibility)
    if check_grid:
        if not _is_north_up(ref_tr):
            raise ValueError("Reference raster has rotation/shear in transform (not north-up).")

        for i, (stk, tr, crs) in enumerate(sources):
            if not _is_north_up(tr):
                raise ValueError(f"Source {i} has rotation/shear in transform (not north-up).")
            if not _same_resolution(ref_tr, tr):
                raise ValueError(
                    f"Source {i} resolution differs from reference. "
                    f"ref(a,e)=({ref_tr.a},{ref_tr.e}) vs src(a,e)=({tr.a},{tr.e})."
                )
            if check_alignment and not _aligned_on_grid(ref_tr, tr):
                raise ValueError(
                    f"Source {i} is not aligned to the same pixel grid as reference (origin mismatch)."
                )

    mems, dss = [], []
    try:
        for stack, tr, crs in sources:
            mem, ds = mem_ds_from_stack(stack, tr, crs, nodata_value=nodata_effective)
            mems.append(mem)
            dss.append(ds)

        mosaic, out_transform = merge(dss, method=method, nodata=nodata_effective)
        out_crs = dss[0].crs

        profile = dss[0].profile.copy()
        profile.update(
            driver="GTiff",
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            count=mosaic.shape[0],
            dtype=mosaic.dtype,
            crs=out_crs,
            transform=out_transform,
            nodata=nodata_effective,
            compress=compress,
            tiled=tiled,
            BIGTIFF=bigtiff,
        )

        if tiled:
            bs = int(blocksize)
            bs = max(16, bs)
            bs = min(bs, mosaic.shape[2], mosaic.shape[1])
            profile.update(blockxsize=bs, blockysize=bs)

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(mosaic)

        return out_path, out_transform, out_crs

    finally:
        for ds in dss:
            try:
                ds.close()
            except Exception:
                pass
        for mem in mems:
            try:
                mem.close()
            except Exception:
                pass



################################################################################
## STACKING SINGLEBAND RASTER LAYERS
################################################################################
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import numpy as np
import rasterio
from rasterio.transform import Affine


def stack_rasters_func(
    raster_paths: Sequence[Union[str, Path]],
    out_dir: Union[str, Path],
    out_name: str,
    band_names: Optional[Sequence[str]] = None,
    strict_dtype: bool = True,
    tolerance: float = 0.0,
    rstr_classes: Optional[Mapping[Union[str, int], str]] = None,
    overwrite: bool = False,
) -> Path:
    """
    Stack multiple aligned (single-band) rasters into one multiband GeoTIFF.

    The first raster in `raster_paths` is used as the reference grid. All other
    rasters must match the reference on:
      - CRS and raster shape (width/height) exactly
      - transform, bounds and resolution either exactly or within `tolerance`

    Optionally, a raster-class mapping (`rstr_classes`) is written as a *dataset-level*
    metadata tag named "CLASS-MAP" containing JSON (keys and values are strings).

    Notes on metadata scope
    -----------------------
    Raster bands in rasterio/GDAL are 1-based (valid indices: 1..count). Dataset-level
    tags are written without a band index and apply to the whole file.

    Parameters
    ----------
    raster_paths : Sequence[str | Path]
        Input rasters to stack. Each input must be single-band and aligned to
        the same grid as the first raster.
    out_dir : str | Path
        Output directory. Created if it does not exist.
    out_name : str
        Output filename. If it does not end in '.tif' or '.tiff', '.tif' is appended.
    band_names : Sequence[str], optional
        Band descriptions to set on the output bands (e.g., years). If provided,
        its length must equal `len(raster_paths)`.
    strict_dtype : bool, default True
        If True, all rasters must have the same dtype as the reference raster.
        If False, dtypes may differ, but writing still uses the reference profile's dtype.
    tolerance : float, default 0.0
        Absolute tolerance used when comparing transform, bounds, and resolution.
        Use 0.0 for strict equality.
    rstr_classes : Mapping[str|int, str], optional
        Mapping from raster class codes to class labels, e.g. {"1": "Remaining", "2": "Wet Nature"}.
        Written to the output as dataset tag "CLASS-MAP" (JSON).
    overwrite : bool, default False
        If False and the output file already exists, the function prints a warning
        and returns the existing file path (skips writing).
        If True and the output file exists, it is deleted first and then rewritten.

    Returns
    -------
    Path
        Path to the created (or already existing) multiband GeoTIFF.

    Raises
    ------
    ValueError
        If `raster_paths` is empty; if any input raster is not single-band; if
        band_names has the wrong length; if rasters are misaligned beyond the
        specified tolerance; or if dtype differs and `strict_dtype=True`.
    """
    if not raster_paths:
        raise ValueError("raster_paths is empty.")

    raster_paths = [Path(p) for p in raster_paths]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / out_name
    if out_path.suffix.lower() not in {".tif", ".tiff"}:
        out_path = out_path.with_suffix(".tif")

    if out_path.exists() and not overwrite:
        print(f"Warning: output exists and overwrite=False; skipping: {out_path}")
        return out_path

    if band_names is not None and len(band_names) != len(raster_paths):
        raise ValueError("band_names length must match raster_paths length.")

    def _affine_close(a: Affine, b: Affine, tol: float) -> bool:
        if tol == 0.0:
            return a == b
        return all(abs(x - y) <= tol for x, y in zip(a, b))

    def _tuple_close(a, b, tol: float) -> bool:
        if tol == 0.0:
            return a == b
        return all(abs(x - y) <= tol for x, y in zip(a, b))

    def _nodata_equal(a, b) -> bool:
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        try:
            if np.isnan(a) and np.isnan(b):
                return True
        except TypeError:
            pass
        return a == b

    # Read reference metadata
    with rasterio.open(raster_paths[0]) as ref:
        if ref.count != 1:
            raise ValueError(f"Reference raster must be single-band. Got count={ref.count}")

        ref_crs = ref.crs
        ref_w, ref_h = ref.width, ref.height
        ref_transform = ref.transform
        ref_bounds = ref.bounds
        ref_res = ref.res
        ref_dtype = ref.dtypes[0]
        ref_nodata = ref.nodata

        profile = ref.profile.copy()
        profile.update(count=len(raster_paths), driver="GTiff")

    # Validate all inputs
    for p in raster_paths:
        with rasterio.open(p) as src:
            if src.count != 1:
                raise ValueError(f"{p} must be single-band. Got count={src.count}")

            if src.crs != ref_crs:
                raise ValueError(f"{p} CRS differs from reference.")

            if (src.width, src.height) != (ref_w, ref_h):
                raise ValueError(f"{p} width/height differs from reference.")

            if not _affine_close(src.transform, ref_transform, tolerance):
                raise ValueError(f"{p} transform differs from reference beyond tolerance={tolerance}.")

            if not _tuple_close(tuple(src.bounds), tuple(ref_bounds), tolerance):
                raise ValueError(f"{p} bounds differ from reference beyond tolerance={tolerance}.")

            if not _tuple_close(tuple(src.res), tuple(ref_res), tolerance):
                raise ValueError(f"{p} resolution differs from reference beyond tolerance={tolerance}.")

            if not _nodata_equal(src.nodata, ref_nodata):
                raise ValueError(f"{p} nodata differs from reference.")

            if strict_dtype and src.dtypes[0] != ref_dtype:
                raise ValueError(f"{p} dtype differs from reference ({src.dtypes[0]} != {ref_dtype}).")

    # Prepare CLASS-MAP tag (optional)
    class_map_json = None
    if rstr_classes:
        classes_norm = {str(k): str(v) for k, v in dict(rstr_classes).items()}
        class_map_json = json.dumps(classes_norm, ensure_ascii=False)

    # If overwriting, remove existing file before writing (helps on Windows / driver quirks)
    if out_path.exists() and overwrite:
        out_path.unlink()

    # Write output
    with rasterio.open(out_path, "w", **profile) as dst:
        for i, p in enumerate(raster_paths, start=1):
            with rasterio.open(p) as src:
                dst.write(src.read(1), i)
            if band_names is not None:
                dst.set_band_description(i, str(band_names[i - 1]))

        if class_map_json is not None:
            dst.update_tags(**{"CLASS-MAP": class_map_json})

    return out_path



################################################################################
## STACK MULTIBAND (RGB) RASTERS
################################################################################
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling
from rasterio.io import MemoryFile


def stack_multiband_rasters_func(
    raster_paths: Sequence[Union[str, Path]],
    out_dir: Union[str, Path],
    out_name: str,
    bands_to_select: Sequence[int],
    raster_names: Sequence[str],
    band_labels: Optional[Sequence[str]] = None,
    strict_dtype: bool = True,
    tolerance: float = 0.0,
    rstr_classes: Optional[Mapping[Union[str, int], str]] = None,
    overwrite: bool = False,
    auto_align: bool = True,
    resampling_method: Resampling = Resampling.bilinear,
) -> Path:
    """
    Stack selected bands from multiple aligned multiband rasters into one multiband GeoTIFF.

    Each input raster contributes the specified bands, and output band names are constructed
    as "{raster_name}_{band_label}" (e.g., "2020_Red", "2020_Green", "2020_Blue").

    The first raster in `raster_paths` is used as the reference grid. All other
    rasters must match the reference on:
      - CRS and raster shape (width/height) exactly
      - transform, bounds and resolution either exactly or within `tolerance`

    If `auto_align=True`, misaligned rasters are automatically reprojected to match
    the reference grid.

    Optionally, a raster-class mapping (`rstr_classes`) is written as a *dataset-level*
    metadata tag named "CLASS-MAP" containing JSON (keys and values are strings).

    Parameters
    ----------
    raster_paths : Sequence[str | Path]
        Input multiband rasters to stack. Each input must be aligned to the same grid.
    out_dir : str | Path
        Output directory. Created if it does not exist.
    out_name : str
        Output filename. If it does not end in '.tif' or '.tiff', '.tif' is appended.
    bands_to_select : Sequence[int]
        1-based band indices to extract from each input raster (e.g., [1, 2, 3] for RGB).
    raster_names : Sequence[str]
        Overarching names for each input raster (e.g., ["2020", "2021", "2022"]).
        Length must equal `len(raster_paths)`.
    band_labels : Sequence[str], optional
        Labels for the selected bands (e.g., ["Red", "Green", "Blue"]).
        Length must equal `len(bands_to_select)`.
        If None, uses the band numbers (e.g., "Band1", "Band2", "Band3").
    strict_dtype : bool, default True
        If True, all rasters must have the same dtype as the reference raster.
        If False, dtypes may differ, but writing still uses the reference profile's dtype.
    tolerance : float, default 0.0
        Absolute tolerance used when comparing transform, bounds, and resolution.
        Use 0.0 for strict equality. Only used when auto_align=False.
    rstr_classes : Mapping[str|int, str], optional
        Mapping from raster class codes to class labels, e.g. {"1": "Remaining", "2": "Wet Nature"}.
        Written to the output as dataset tag "CLASS-MAP" (JSON).
    overwrite : bool, default False
        If False and the output file already exists, the function prints a warning
        and returns the existing file path (skips writing).
        If True and the output file exists, it is deleted first and then rewritten.
    auto_align : bool, default True
        If True, automatically reproject rasters that don't match the reference grid.
        If False, raises ValueError for misaligned rasters.
    resampling_method : Resampling, default Resampling.bilinear
        Resampling method to use when auto_align=True.
        Options: Resampling.nearest, Resampling.bilinear, Resampling.cubic, etc.

    Returns
    -------
    Path
        Path to the created (or already existing) multiband GeoTIFF.

    Raises
    ------
    ValueError
        If `raster_paths` is empty; if any requested band doesn't exist in an input raster;
        if raster_names or band_labels have the wrong length; if rasters are misaligned
        beyond the specified tolerance (when auto_align=False); or if dtype differs and `strict_dtype=True`.
    """
    if not raster_paths:
        raise ValueError("raster_paths is empty.")

    if len(raster_names) != len(raster_paths):
        raise ValueError("raster_names length must match raster_paths length.")

    if not bands_to_select:
        raise ValueError("bands_to_select is empty.")

    if band_labels is not None and len(band_labels) != len(bands_to_select):
        raise ValueError("band_labels length must match bands_to_select length.")

    raster_paths = [Path(p) for p in raster_paths]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / out_name
    if out_path.suffix.lower() not in {".tif", ".tiff"}:
        out_path = out_path.with_suffix(".tif")

    if out_path.exists() and not overwrite:
        print(f"Warning: output exists and overwrite=False; skipping: {out_path}")
        return out_path

    # Create default band labels if not provided
    if band_labels is None:
        band_labels = [f"Band{b}" for b in bands_to_select]

    def _affine_close(a: Affine, b: Affine, tol: float) -> bool:
        if tol == 0.0:
            return a == b
        return all(abs(x - y) <= tol for x, y in zip(a, b))

    def _tuple_close(a, b, tol: float) -> bool:
        if tol == 0.0:
            return a == b
        return all(abs(x - y) <= tol for x, y in zip(a, b))

    def _nodata_equal(a, b) -> bool:
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        try:
            if np.isnan(a) and np.isnan(b):
                return True
        except TypeError:
            pass
        return a == b

    def _needs_alignment(src, ref_crs, ref_w, ref_h, ref_transform, ref_bounds, ref_res, tol):
        """Check if a raster needs alignment to the reference grid."""
        if src.crs != ref_crs:
            return True
        if (src.width, src.height) != (ref_w, ref_h):
            return True
        if not _affine_close(src.transform, ref_transform, tol):
            return True
        if not _tuple_close(tuple(src.bounds), tuple(ref_bounds), tol):
            return True
        if not _tuple_close(tuple(src.res), tuple(ref_res), tol):
            return True
        return False

    def _align_raster_in_memory(src, ref_profile, bands_to_extract, resampling):
        """Align a raster to reference grid in memory and return data arrays."""
        ref_crs = ref_profile['crs']
        ref_transform = ref_profile['transform']
        ref_width = ref_profile['width']
        ref_height = ref_profile['height']
        
        aligned_bands = []
        for band_idx in bands_to_extract:
            src_band = src.read(band_idx)
            dst_band = np.empty((ref_height, ref_width), dtype=src.dtypes[band_idx - 1])
            
            reproject(
                source=src_band,
                destination=dst_band,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=resampling
            )
            aligned_bands.append(dst_band)
        
        return aligned_bands

    # Read reference metadata
    with rasterio.open(raster_paths[0]) as ref:
        if ref.count < max(bands_to_select):
            raise ValueError(
                f"Reference raster has {ref.count} bands but band {max(bands_to_select)} was requested."
            )

        ref_crs = ref.crs
        ref_w, ref_h = ref.width, ref.height
        ref_transform = ref.transform
        ref_bounds = ref.bounds
        ref_res = ref.res
        # Use dtype of first selected band as reference
        ref_dtype = ref.dtypes[bands_to_select[0] - 1]
        ref_nodata = ref.nodata

        ref_profile = {
            'crs': ref_crs,
            'transform': ref_transform,
            'width': ref_w,
            'height': ref_h,
        }

        profile = ref.profile.copy()
        total_bands = len(raster_paths) * len(bands_to_select)
        profile.update(count=total_bands, driver="GTiff")

    # Validate all inputs
    for p in raster_paths:
        with rasterio.open(p) as src:
            if src.count < max(bands_to_select):
                raise ValueError(
                    f"{p} has {src.count} bands but band {max(bands_to_select)} was requested."
                )

            needs_align = _needs_alignment(
                src, ref_crs, ref_w, ref_h, ref_transform, ref_bounds, ref_res, tolerance
            )

            if needs_align and not auto_align:
                if src.crs != ref_crs:
                    raise ValueError(f"{p} CRS differs from reference.")
                if (src.width, src.height) != (ref_w, ref_h):
                    raise ValueError(f"{p} width/height differs from reference.")
                if not _affine_close(src.transform, ref_transform, tolerance):
                    raise ValueError(f"{p} transform differs from reference beyond tolerance={tolerance}.")
                if not _tuple_close(tuple(src.bounds), tuple(ref_bounds), tolerance):
                    raise ValueError(f"{p} bounds differ from reference beyond tolerance={tolerance}.")
                if not _tuple_close(tuple(src.res), tuple(ref_res), tolerance):
                    raise ValueError(f"{p} resolution differs from reference beyond tolerance={tolerance}.")

            if not _nodata_equal(src.nodata, ref_nodata):
                print(f"Warning: {p} nodata differs from reference. Using reference nodata value.")

            if strict_dtype:
                for band_idx in bands_to_select:
                    if src.dtypes[band_idx - 1] != ref_dtype:
                        raise ValueError(
                            f"{p} band {band_idx} dtype differs from reference "
                            f"({src.dtypes[band_idx - 1]} != {ref_dtype})."
                        )

    # Prepare CLASS-MAP tag (optional)
    class_map_json = None
    if rstr_classes:
        classes_norm = {str(k): str(v) for k, v in dict(rstr_classes).items()}
        class_map_json = json.dumps(classes_norm, ensure_ascii=False)

    # If overwriting, remove existing file before writing
    if out_path.exists() and overwrite:
        out_path.unlink()

    # Write output
    with rasterio.open(out_path, "w", **profile) as dst:
        out_band_idx = 1
        for raster_path, raster_name in zip(raster_paths, raster_names):
            with rasterio.open(raster_path) as src:
                needs_align = _needs_alignment(
                    src, ref_crs, ref_w, ref_h, ref_transform, ref_bounds, ref_res, tolerance
                )

                if needs_align and auto_align:
                    print(f"Aligning {raster_path.name} to reference grid...")
                    aligned_bands = _align_raster_in_memory(
                        src, ref_profile, bands_to_select, resampling_method
                    )
                    
                    for band_data, band_label in zip(aligned_bands, band_labels):
                        dst.write(band_data, out_band_idx)
                        band_description = f"{raster_name}_{band_label}"
                        dst.set_band_description(out_band_idx, band_description)
                        out_band_idx += 1
                else:
                    # No alignment needed, read directly
                    for band_num, band_label in zip(bands_to_select, band_labels):
                        data = src.read(band_num)
                        dst.write(data, out_band_idx)
                        band_description = f"{raster_name}_{band_label}"
                        dst.set_band_description(out_band_idx, band_description)
                        out_band_idx += 1

        if class_map_json is not None:
            dst.update_tags(**{"CLASS-MAP": class_map_json})

    return out_path



################################################################################
## RASTER TO INT16 CONVERSION
################################################################################
from pathlib import Path
import numpy as np
import rasterio


def convert_raster_to_int16(
    in_path: str | Path,
    out_path: str | Path | None = None,
    *,
    overwrite: bool = False,
    nodata: int | None = None,
    copy_nodata_if_valid: bool = True,
    clamp: bool = True,
) -> Path:
    """
    Convert a raster to int16.

    - If input is already int16 and out_path is None: returns in_path.
    - If `nodata` is provided: sets it on output (must fit in int16).
    - If `nodata` is None:
        - if copy_nodata_if_valid=True and input nodata fits in int16, it is copied
        - otherwise nodata is removed.
    - If clamp=True: values outside int16 range are clipped to [-32768, 32767].

    Returns the output path (or in_path if no conversion was needed).
    """
    in_path = Path(in_path)

    with rasterio.open(in_path) as src:
        if src.dtypes[0] == "int16" and out_path is None:
            return in_path

        if out_path is None:
            out_path = in_path.with_name(f"{in_path.stem}_i16{in_path.suffix}")
        out_path = Path(out_path)

        if out_path.exists() and not overwrite:
            raise FileExistsError(f"Output exists: {out_path}")

        meta = src.meta.copy()
        meta.update(dtype="int16")

        i16_min, i16_max = np.iinfo(np.int16).min, np.iinfo(np.int16).max

        # Decide nodata
        if nodata is not None:
            if not (i16_min <= nodata <= i16_max):
                raise ValueError(f"Provided nodata {nodata} does not fit in int16")
            meta.update(nodata=int(nodata))
        else:
            src_nodata = src.nodata
            if copy_nodata_if_valid and src_nodata is not None and i16_min <= src_nodata <= i16_max:
                meta.update(nodata=int(src_nodata))
            else:
                meta.pop("nodata", None)

        with rasterio.open(out_path, "w", **meta) as dst:
            for b in range(1, src.count + 1):
                arr = src.read(b)

                if clamp:
                    arr = np.clip(arr, i16_min, i16_max)

                arr = arr.astype(np.int16, copy=False)
                dst.write(arr, b)

            # Preserve internal mask if present
            try:
                dst.write_mask(src.dataset_mask())
            except Exception:
                pass

    return out_path



################################################################################
## ALIGN RASTER TO REFERENCE
################################################################################
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

def align_raster_to_reference(src_path, ref_path, out_path):
    """Align src_path to match the grid of ref_path."""
    with rasterio.open(ref_path) as ref:
        ref_profile = ref.profile.copy()
        ref_transform = ref.transform
        ref_width = ref.width
        ref_height = ref.height
        ref_crs = ref.crs
    
    with rasterio.open(src_path) as src:
        # Update profile to match reference
        out_profile = src.profile.copy()
        out_profile.update({
            'crs': ref_crs,
            'transform': ref_transform,
            'width': ref_width,
            'height': ref_height
        })
        
        with rasterio.open(out_path, 'w', **out_profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=Resampling.bilinear
                )