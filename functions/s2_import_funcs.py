from __future__ import annotations

################################################################################
## LOAD SENTINEL-2 IMAGES
################################################################################
from pathlib import Path
import zipfile
import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.coords import BoundingBox


def s2_raster_imp_func(
    zip_path: str | Path,
    bands: list[str] | tuple[str, ...],
    resolution_tag: str = "R10m",
) -> tuple[np.ndarray, rasterio.crs.CRS, Affine, BoundingBox]:
    """
    Load one or more Sentinel-2 JP2 bands directly from a zipped SAFE product and
    stack them into a 3D NumPy array, while also returning the georeferencing
    needed for mapping/overlay (CRS + affine transform + bounds).

    No resampling is performed. The function enforces that all requested bands
    are on the exact same grid (same CRS, transform, width, height). If they are
    not, it raises an error.

    Parameters
    ----------
    zip_path
        Path to the `.zip` containing a Sentinel-2 SAFE product.
    bands
        Band identifiers to load (e.g. ["B02", "B03", "B04", "B08"]).
        The returned stack follows this order.
    resolution_tag
        Resolution subfolder tag used to locate the JP2s inside the SAFE, e.g.
        "R10m" or "R20m". All requested bands must exist under this tag.

    Returns
    -------
    stack, crs, transform, bounds
        stack : numpy.ndarray
            Array with shape (n_bands, rows, cols), stacked in the same order as
            `bands`.
        crs : rasterio.crs.CRS
            CRS shared by all bands.
        transform : rasterio.transform.Affine
            Affine transform mapping pixel coordinates to CRS coordinates.
        bounds : rasterio.coords.BoundingBox
            Bounding box (left, bottom, right, top) in the returned CRS. Useful
            as `extent=` for plotting.

    Raises
    ------
    FileNotFoundError
        If `zip_path` does not exist, or if a requested band cannot be found in
        the zip under the given `resolution_tag`.
    ValueError
        If `zip_path` is not a `.zip`, or if any band does not match the
        reference grid (CRS/transform/width/height) and therefore cannot be
        stacked without resampling.
    """
    zip_path = Path(zip_path)
    bands = list(bands)

    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    if zip_path.suffix.lower() != ".zip":
        raise ValueError("zip_path must point to a .zip file")

    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()

    band_to_inside_path: dict[str, str] = {}
    for b in bands:
        candidates = [
            n for n in names
            if n.endswith(".jp2")
            and "IMG_DATA" in n
            and resolution_tag in n
            and f"_{b}_" in n
        ]
        candidates.sort()
        if not candidates:
            raise FileNotFoundError(
                f"Band {b} not found in zip for {resolution_tag}. "
                f"Check band name and resolution_tag."
            )
        band_to_inside_path[b] = candidates[0]

    ref_band = bands[0]
    ref_vsi = f"/vsizip/{zip_path.as_posix()}/{band_to_inside_path[ref_band]}"

    arrays: list[np.ndarray] = []
    with rasterio.open(ref_vsi) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_bounds = ref.bounds
        ref_width, ref_height = ref.width, ref.height
        ref_res = ref.res  # (xres, yres)

        arrays.append(ref.read(1))

        for b in bands[1:]:
            vsi = f"/vsizip/{zip_path.as_posix()}/{band_to_inside_path[b]}"
            with rasterio.open(vsi) as src:
                same_grid = (
                    src.crs == ref_crs
                    and src.transform == ref_transform
                    and src.width == ref_width
                    and src.height == ref_height
                )
                if not same_grid:
                    raise ValueError(
                        "Bands do not align to the same grid. "
                        f"Reference: band={ref_band}, res={ref_res}, shape=({ref_height},{ref_width}), "
                        f"crs={ref_crs}. "
                        f"Mismatch: band={b}, res={src.res}, shape=({src.height},{src.width}), crs={src.crs}. "
                        "Use bands from the same resolution folder (e.g. all R10m), "
                        "or resample explicitly outside this function."
                    )

                arrays.append(src.read(1))

    stack = np.stack(arrays, axis=0)
    return stack, ref_crs, ref_transform, ref_bounds



################################################################################
## LOAD SENTINEL-2 MOSAIC
################################################################################
import os
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.coords import BoundingBox
from rasterio.crs import CRS


def s2_mosaic_import_func(
    zip_path: str | Path,
    bands_to_import: list[str] | tuple[str, ...],
) -> tuple[np.ndarray, CRS, Affine, BoundingBox]:
    """
    Load one or more Sentinel-2 band GeoTIFFs (e.g. "B02.tif", "B03.tif", ...)
    directly from a .zip and stack them into a 3D NumPy array, while also
    returning georeferencing (CRS + affine transform + bounds).

    No resampling is performed. The function enforces that all requested bands
    are on the exact same grid (same CRS, transform, width, height). If they are
    not, it raises an error.

    Parameters
    ----------
    zip_path
        Path to the `.zip` that contains the band GeoTIFFs.
    bands_to_import
        Band identifiers to load, e.g. ["B02", "B03", "B04"].
        Values like "B02.tif" are also accepted. The returned stack follows this order.

    Returns
    -------
    stack, ref_crs, ref_transform, ref_bounds
        stack : numpy.ndarray
            Array with shape (n_bands, rows, cols), stacked in the same order as
            `bands_to_import`.
        ref_crs : rasterio.crs.CRS
            CRS shared by all bands.
        ref_transform : affine.Affine
            Affine transform mapping pixel coordinates to CRS coordinates.
        ref_bounds : rasterio.coords.BoundingBox
            Bounding box (left, bottom, right, top) in the returned CRS.

    Raises
    ------
    FileNotFoundError
        If `zip_path` does not exist, or if a requested band file is not found in the zip.
    ValueError
        If `zip_path` is not a `.zip`, or if any band does not match the reference grid
        (CRS/transform/width/height) and therefore cannot be stacked without resampling.
    rasterio.errors.RasterioIOError
        If a band GeoTIFF inside the zip cannot be opened/read.
    """
    zip_path = Path(zip_path)
    bands = list(bands_to_import)

    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    if zip_path.suffix.lower() != ".zip":
        raise ValueError("zip_path must point to a .zip file")
    if not bands:
        raise ValueError("bands_to_import must be a non-empty list/tuple, e.g. ['B02','B03'].")

    # Normalize requested filenames to "B02.tif" etc.
    req_files: list[str] = []
    for b in bands:
        if not isinstance(b, str) or not b.strip():
            raise ValueError(f"Invalid band value: {b!r}")
        b = b.strip()
        if not b.lower().endswith(".tif"):
            b = f"{b}.tif"
        req_files.append(b)

    # List zip members once
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()

    # Map requested band file -> first matching zip member (by basename, case-insensitive)
    band_to_inside_path: dict[str, str] = {}
    for f in req_files:
        f_l = f.lower()
        candidates = [n for n in names if os.path.basename(n).lower() == f_l]
        candidates.sort()
        if not candidates:
            raise FileNotFoundError(f"Band file '{f}' not found inside zip: {zip_path}")
        band_to_inside_path[f] = candidates[0]

    # Read reference band
    ref_file = req_files[0]
    ref_vsi = f"/vsizip/{zip_path.as_posix()}/{band_to_inside_path[ref_file]}"

    arrays: list[np.ndarray] = []
    with rasterio.open(ref_vsi) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_bounds = ref.bounds
        ref_width, ref_height = ref.width, ref.height

        arrays.append(ref.read(1))

        # Read remaining bands + enforce exact grid match
        for f in req_files[1:]:
            vsi = f"/vsizip/{zip_path.as_posix()}/{band_to_inside_path[f]}"
            with rasterio.open(vsi) as src:
                same_grid = (
                    src.crs == ref_crs
                    and src.transform == ref_transform
                    and src.width == ref_width
                    and src.height == ref_height
                )
                if not same_grid:
                    raise ValueError(
                        "Bands do not align to the same grid. "
                        f"Reference={ref_file} shape=({ref_height},{ref_width}) crs={ref_crs}. "
                        f"Mismatch={os.path.basename(band_to_inside_path[f])} "
                        f"shape=({src.height},{src.width}) crs={src.crs}."
                    )

                arrays.append(src.read(1))

    stack = np.stack(arrays, axis=0)
    return stack, ref_crs, ref_transform, ref_bounds



################################################################################
## SENTINEL-2 PATH BUILDER
################################################################################
from pathlib import Path
from typing import Union

def sentinel_path_builder(
    base_dir: Union[str, Path],
    year: int,
    quarter_or_month: Union[str, int],
    filename: Union[str, Path],
) -> Path:
    """
    Build a Sentinel raster path like:
      <base_dir>/<YEAR>_<Q# or MM>/<filename>

    Examples
    --------
    sentinel_path_builder(base, 2017, "q1", "S2_b2348_stack.tif")
      -> .../2017_Q1/S2_b2348_stack.tif

    sentinel_path_builder(base, 2017, 11, "S2_b2348_stack.tif")
      -> .../2017_11/S2_b2348_stack.tif
    """
    base_dir = Path(base_dir)
    filename = Path(filename).name  # ensure only the file name is appended

    if isinstance(quarter_or_month, str):
        s = quarter_or_month.strip()
        s_low = s.lower()

        # quarter like "q1", "Q2"
        if s_low.startswith("q") and s_low[1:].isdigit():
            q = int(s_low[1:])
            if q not in (1, 2, 3, 4):
                raise ValueError(f"Quarter must be Q1..Q4, got {quarter_or_month!r}")
            subfolder = f"{year}_Q{q}"
        # month like "11", "03"
        elif s_low.isdigit():
            m = int(s_low)
            if not (1 <= m <= 12):
                raise ValueError(f"Month must be 1..12, got {quarter_or_month!r}")
            subfolder = f"{year}_{m:02d}"
        else:
            raise ValueError(f"quarter_or_month must be like 'Q1' or 1..12, got {quarter_or_month!r}")

    else:
        m = int(quarter_or_month)
        if not (1 <= m <= 12):
            raise ValueError(f"Month must be 1..12, got {quarter_or_month!r}")
        subfolder = f"{year}_{m:02d}"

    return base_dir / subfolder / filename