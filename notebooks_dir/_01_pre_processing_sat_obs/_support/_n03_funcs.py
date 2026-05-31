from __future__ import annotations

###############################################################################
## RASTER OPERATIONS FUNC
###############################################################################
from pathlib import Path
from typing import Callable, List, Optional


def apply_raster_op_tree(
    in_root: str | Path,
    out_root: str | Path,
    operation: Callable[[str, str], str | Path],
    *,
    pattern: str = "*.tif",
    overwrite: bool = False,
    keep_filename: bool = True,
    suffix: str = "_op",
    skip_if_up_to_date: bool = False,
    rename_from: str = "_stitched",
    rename_to: Optional[str] = None,
) -> List[Path]:
    """
    Recursively apply a raster band operation to rasters under an input directory tree
    and write results to a mirrored output directory tree.

    Folder traversal + output naming logic is copied from `clip_raster_tree`.

    Parameters
    ----------
    in_root, out_root
        Input and output root folders.
    operation
        Function that performs the band operation.

        It is called as:
            operation(in_path: str, out_path: str) -> str | Path

        It must write the output raster to `out_path` and return the final output path.
        (Returning out_path is fine.)
    pattern
        rglob() pattern to find input rasters.
    overwrite, keep_filename, suffix, skip_if_up_to_date, rename_from, rename_to
        Same meaning as in your clip function (except clipping-specific params).

    Returns
    -------
    List[Path]
        List of output file paths (including files that already existed / were skipped).
    """
    in_root = Path(in_root)
    out_root = Path(out_root)

    if not in_root.exists():
        raise FileNotFoundError(f"in_root not found: {in_root}")

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

            if out_path.exists() and not overwrite:
                print(f"  - {raster_path.name}: exists -> {out_path}")
                outputs.append(out_path)
                continue

            final_out = operation(str(raster_path), str(out_path))
            final_out = Path(final_out)

            print(f"  - {raster_path.name}: wrote -> {final_out}")
            outputs.append(final_out)

    return outputs



###############################################################################
## RASTER BANDS FINDER
###############################################################################
import rasterio

def _find_band_index_by_description(src: rasterio.io.DatasetReader, contains: str) -> int:
    contains = contains.lower()
    for i, d in enumerate(src.descriptions, start=1):
        if d and contains in d.lower():
            return i
    raise ValueError(f"Could not find a band whose description contains '{contains}'. "
                     f"Descriptions: {src.descriptions}")



###############################################################################
## NDVI-calculator
###############################################################################
import numpy as np
import rasterio

def ndvi_op(in_path: str, out_path: str) -> str:
    with rasterio.open(in_path) as src:
        red_i = _find_band_index_by_description(src, "b04")  # e.g. "B04_clip_ruim"
        nir_i = _find_band_index_by_description(src, "b08")  # e.g. "B08_clip_ruim"

        red = src.read(red_i).astype("float32")
        nir = src.read(nir_i).astype("float32")

        ndvi = (nir - red) / (nir + red + 1e-6)

        profile = src.profile.copy()
        profile.update(count=1, dtype="float32", nodata=-9999)

        ndvi = np.where(np.isfinite(ndvi), ndvi, profile["nodata"]).astype("float32")

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(ndvi, 1)
            dst.set_band_description(1, "NDVI")

    return out_path



###############################################################################
## NDWI-calculator
###############################################################################
import rasterio
import numpy as np

def ndwi_op(in_path: str, out_path: str) -> str:
    """
    NDWI (McFeeters): (Green - NIR) / (Green + NIR)

    Expects band descriptions to include something like "B03" for green
    and "B08" for NIR (as in your stacks: B03_clip_ruim, B08_clip_ruim).
    """
    with rasterio.open(in_path) as src:
        green_i = _find_band_index_by_description(src, "b03")  # Green
        nir_i = _find_band_index_by_description(src, "b08")    # NIR

        green = src.read(green_i).astype("float32")
        nir = src.read(nir_i).astype("float32")

        ndwi = (green - nir) / (green + nir + 1e-6)

        profile = src.profile.copy()
        profile.update(count=1, dtype="float32", nodata=-9999)

        ndwi = np.where(np.isfinite(ndwi), ndwi, profile["nodata"]).astype("float32")

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(ndwi, 1)
            dst.set_band_description(1, "NDWI")

    return out_path