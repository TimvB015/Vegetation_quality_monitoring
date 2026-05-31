from __future__ import annotations

from functions.raster_ops_funcs import (
    clip_with_rasterio_func,
    stich_multiple_rasters_func,
    convert_raster_to_int16,
)

###############################################################################
## HELPER FUNCTIONS
###############################################################################
import os
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import Affine

def _read_stack_from_zipped_tif(zip_path: Path, member_name: str) -> Tuple[np.ndarray, Affine, rasterio.crs.CRS]:
    """
    Read a GeoTIFF inside a zip into (stack, transform, crs) suitable for stich_multiple_rasters_func.
    Returns stack shaped (bands, H, W).
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(member_name) as f:
            data = f.read()

    with MemoryFile(data) as mem:
        with mem.open() as ds:
            arr = ds.read()  # (bands, H, W)
            return arr, ds.transform, ds.crs


def _find_band_members_in_zip(zip_path: Path, band_filenames: List[str]) -> Dict[str, str]:
    """
    Returns mapping band_filename -> member_name inside zip.
    Matches by basename (case-insensitive). Handles nested paths in zip.
    """
    want = {b.lower(): b for b in band_filenames}
    found: Dict[str, str] = {}

    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            base = Path(name).name.lower()
            if base in want:
                found[want[base]] = name

    return found



###############################################################################
## AUTOMATICALLY STITCH ALL OBSERVATIONS
###############################################################################
def stitch_sentinel_images_tree(
    in_root: str | Path,
    out_root: str | Path,
    band_filenames: List[str],
    nodata_mode: str = "auto",
    nodata=None,
    check_grid: bool = True,
):
    """
    Walks subfolders of in_root (arbitrary names), expects .zip tiles in each subfolder.
    For each subfolder and each band filename, reads that band from all zips and stitches them.
    Writes output to out_root/<subfolder>/<band>_stitched.tif

    Parameters
    ----------
    in_root : path
        Root folder like s2_mosaic/
    out_root : path
        Output root folder (will be created)
    band_filenames : list[str]
        REQUIRED. Band files to stitch inside each zip (e.g. ["B02.tif", ...] or ["VV.tif","VH.tif"]).
    nodata_mode, nodata, check_grid
        Passed to stich_multiple_rasters_func.
    """
    in_root = Path(in_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if not band_filenames or not isinstance(band_filenames, (list, tuple)):
        raise ValueError("band_filenames must be provided as a non-empty list, e.g. ['B02.tif','B03.tif']")

    # Only process direct subfolders; adjust to rglob if you want deeper recursion.
    subfolders = [p for p in in_root.iterdir() if p.is_dir()]
    if not subfolders:
        raise ValueError(f"No subfolders found in {in_root}")

    for sub in sorted(subfolders):
        zip_files = sorted(sub.glob("*.zip"))
        if not zip_files:
            continue

        out_sub = out_root / sub.name
        out_sub.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing: {sub.name} ({len(zip_files)} zip tiles)")

        for band in band_filenames:
            sources = []
            missing_in = []

            for zp in zip_files:
                members = _find_band_members_in_zip(zp, [band])
                if band not in members:
                    missing_in.append(zp.name)
                    continue

                stack, tr, crs = _read_stack_from_zipped_tif(zp, members[band])
                sources.append((stack, tr, crs))

            if not sources:
                print(f"  - {band}: skipped (not found in any zip)")
                continue

            if missing_in:
                print(
                    f"  - {band}: warning, missing in {len(missing_in)} zip(s): "
                    f"{missing_in[:3]}{'...' if len(missing_in) > 3 else ''}"
                )

            out_name = f"{Path(band).stem}_stitched.tif"
            out_path = out_sub / out_name

            stich_multiple_rasters_func(
                sources=sources,
                out_path=str(out_path),
                nodata=nodata,
                method="first",
                compress="deflate",
                tiled=True,
                blocksize=512,
                nodata_mode=nodata_mode,
                check_grid=check_grid,
            )

            print(f"  - {band}: wrote {out_path}")


if __name__ == "__main__":
    # Example usage
    # in_root = r"/path/to/s2_mosaic"
    # out_root = r"/path/to/s2_mosaic_stitched"
    # stitch_s2_mosaic_tree(in_root, out_root)
    pass



###############################################################################
## AUTOMATICALLY TRANSFER TREE TO INT16
###############################################################################
from pathlib import Path
from typing import List, Tuple
import rasterio


def convert_tree_to_int16(
    in_root: str | Path,
    out_root: str | Path,
    *,
    pattern: str = "*.tif",
    recursive: bool = True,
    overwrite: bool = False,
    nodata: int | None = None,
    copy_nodata_if_valid: bool = True,
    skip_if_int16: bool = True,
    clamp: bool = True,
) -> Tuple[List[Path], List[Path]]:
    """
    Convert GeoTIFFs in a directory tree to int16 and write them to a mirrored
    directory structure under `out_root`.

    Returns (written_paths, skipped_paths).
    """
    in_root = Path(in_root)
    out_root = Path(out_root)

    if not in_root.exists():
        raise FileNotFoundError(f"in_root not found: {in_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    paths = (in_root.rglob(pattern) if recursive else in_root.glob(pattern))
    paths = sorted(p for p in paths if p.is_file())

    written: List[Path] = []
    skipped: List[Path] = []

    for in_path in paths:
        rel = in_path.relative_to(in_root)
        out_path = out_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(in_path) as src:
            dtype0 = src.dtypes[0]

        if dtype0 == "int16" and skip_if_int16:
            skipped.append(in_path)
            continue

        out_written = convert_raster_to_int16(
            in_path=in_path,
            out_path=out_path,
            overwrite=overwrite,
            nodata=nodata,
            copy_nodata_if_valid=copy_nodata_if_valid,
            clamp=clamp,
        )
        written.append(Path(out_written))

    return written, skipped



###############################################################################
## AUTOMATICALLY CLIP ALL OBSERVATIONS
###############################################################################
from pathlib import Path
from typing import List, Optional, Union


def clip_raster_tree(
    in_root: str | Path,
    out_root: str | Path,
    gpkg_path: str | Path,
    *,
    pattern: str = "*_stitched.tif",
    layer: Optional[str] = None,
    dissolve: bool = True,
    all_touched: bool = False,
    nodata: Optional[Union[int, float]] = -9999,
    blocksize: int = 512,
    overwrite: bool = False,
    keep_filename: bool = True,
    suffix: str = "_clipped",
    skip_if_up_to_date: bool = False,
    include_mask_layer: bool = True,
    mask_as_alpha: bool = False,
    rename_from: str = "_stitched",
    rename_to: Optional[str] = None,
) -> List[Path]:
    """
    Recursively clip rasters under an input directory tree and write results to a
    mirrored output directory tree.

    The function searches `in_root` for rasters matching `pattern` (via `Path.rglob()`),
    groups them by their parent folder relative to `in_root`, and for each raster:
      - clips it to the geometries in `gpkg_path` (via `clip_with_rasterio_func`)
      - writes outputs to: `out_root/<same relative subfolder>/...`

    Output naming
    -------------
    - If `keep_filename=False`, the output name is:
        `<input_stem><suffix><input_suffix>`
    - If `keep_filename=True` (default), the output name is based on the input filename.
      Additionally, a simple substring replacement can be applied:
        `out_name = in_name.replace(rename_from, rename_to_or_suffix)`

      By default this means:
        `"_stitched"` is replaced by `suffix`

      Example:
        input : "S1_2020_stitched.tif"
        suffix: "_clip_ruim"
        output: "S1_2020_clip_ruim.tif"

    Nodata & mask behavior
    ----------------------
    - `nodata` defaults to **-9999** and is passed to `clip_with_rasterio_func`.
      Pixels outside the clip geometry are filled with this nodata value and it is
      written to output metadata.
    - `include_mask_layer` defaults to **True** and appends a uint8 validity mask band:
        * 1 = valid pixel (inside clip geometry)
        * 0 = invalid pixel (outside geometry / nodata)
    - If `mask_as_alpha=True`, the appended mask band is marked as an alpha band
      (where supported).

    Parameters
    ----------
    in_root : str | Path
        Root directory containing rasters to clip (subfolders allowed).
    out_root : str | Path
        Root directory where clipped rasters will be written. Subfolders are created
        to match the structure under `in_root`.
    gpkg_path : str | Path
        Path to a GeoPackage containing clip geometries.
    pattern : str, default "*_stitched.tif"
        Glob pattern used with `Path.rglob()` to select rasters to clip.
    layer : Optional[str], default None
        GeoPackage layer name to read. If None, the default layer is used.
    dissolve : bool, default True
        If True, dissolve all input features into a single geometry before clipping.
    all_touched : bool, default False
        Passed to `rasterio.mask.mask`. If True, include all pixels touched by the geometry.
    nodata : Optional[int | float], default -9999
        Output nodata value used to fill pixels outside the clip geometry and written
        into output metadata.
    blocksize : int, default 512
        Output GeoTIFF tile size (blockxsize/blockysize).
    overwrite : bool, default False
        If False and the output file exists, clipping is skipped for that file.
    keep_filename : bool, default True
        If True, keep the input filename (optionally applying `rename_from`/`rename_to`).
        If False, append `suffix` to the stem.
    suffix : str, default "_clipped"
        Suffix appended to the filename when `keep_filename=False`.
        Also used as the default replacement target when `keep_filename=True` and
        `rename_to=None`.
    skip_if_up_to_date : bool, default False
        If True, and output exists with modification time >= input modification time,
        skip clipping (unless `overwrite=True`).
        Note: this only checks timestamps; changing parameters like `include_mask_layer`
        or renaming settings may require `overwrite=True` to regenerate outputs.
    include_mask_layer : bool, default True
        If True, append a uint8 validity mask band (1=valid, 0=invalid) to each output.
    mask_as_alpha : bool, default False
        If True and `include_mask_layer=True`, mark the appended mask band as an alpha band
        (where supported).
    rename_from : str, default "_stitched"
        Substring in the input filename to replace when `keep_filename=True`.
        Use "" to disable replacement (not recommended); prefer setting `keep_filename=False`
        if you only want suffix-appending behavior.
    rename_to : Optional[str], default None
        Replacement string used when `keep_filename=True`.
        If None, `suffix` is used as the replacement value.

    Returns
    -------
    List[Path]
        List of output file paths (including files that already existed).
    """
    in_root = Path(in_root)
    out_root = Path(out_root)
    gpkg_path = Path(gpkg_path)

    if not in_root.exists():
        raise FileNotFoundError(f"in_root not found: {in_root}")
    if not gpkg_path.exists():
        raise FileNotFoundError(f"gpkg_path not found: {gpkg_path}")

    out_root.mkdir(parents=True, exist_ok=True)

    rasters = sorted(in_root.rglob(pattern))
    if not rasters:
        raise ValueError(f"No rasters found under {in_root} matching pattern '{pattern}'")

    groups: dict[Path, List[Path]] = {}
    for rp in rasters:
        rel_parent = rp.parent.relative_to(in_root)
        groups.setdefault(rel_parent, []).append(rp)

    outputs: List[Path] = []

    for rel_parent in sorted(groups.keys()):
        group = sorted(groups[rel_parent])
        label = rel_parent.as_posix() if rel_parent.as_posix() != "." else in_root.name
        print(f"\nProcessing: {label} ({len(group)} rasters)")

        for raster_path in group:
            out_dir = out_root / rel_parent
            out_dir.mkdir(parents=True, exist_ok=True)

            if keep_filename:
                # Rename by replacement (default: "_stitched" -> suffix)
                _rename_to = suffix if rename_to is None else rename_to
                out_name = raster_path.name.replace(rename_from, _rename_to)
            else:
                out_name = f"{raster_path.stem}{suffix}{raster_path.suffix}"

            out_path = out_dir / out_name

            if skip_if_up_to_date and out_path.exists() and not overwrite:
                if out_path.stat().st_mtime >= raster_path.stat().st_mtime:
                    print(f"  - {raster_path.name}: exists (up-to-date) -> {out_path}")
                    outputs.append(out_path)
                    continue

            final_out, epsg, status = clip_with_rasterio_func(
                raster_path=str(raster_path),
                gpkg_path=str(gpkg_path),
                out_dir=str(out_dir),
                out_filename=out_name,
                overwrite=overwrite,
                layer=layer,
                dissolve=dissolve,
                all_touched=all_touched,
                nodata=nodata,
                blocksize=blocksize,
                include_mask_layer=include_mask_layer,
                mask_as_alpha=mask_as_alpha,
            )

            print(
                f"  - {raster_path.name}: "
                f"{'exists' if status == 'exists' else 'wrote'} -> {final_out}"
            )
            outputs.append(Path(final_out))

    return outputs