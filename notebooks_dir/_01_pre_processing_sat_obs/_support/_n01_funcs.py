from __future__ import annotations

###############################################################################
## GETTING THE PIXEL RASTER FROM THE ORIGINAL IMAGE
###############################################################################
import os
import numpy as np
import rasterio


def get_pixel_raster_func(
    raster_path: str,
    output_dir: str,
    output_name: str,
    fill_value,
    overwrite: bool = True,
):
    """
    Create a new GeoTIFF based on an input raster:
      - preserves georeferencing + metadata/profile
      - keeps ONLY the first band
      - fills the first band with `fill_value`

    Parameters
    ----------
    raster_path : str
        Path to input raster (any band count).
    output_dir : str
        Directory to write output to.
    output_name : str
        Output filename (with or without .tif extension).
    fill_value : int/float
        Value to assign to every pixel in band 1.
    overwrite : bool, default True
        If False and output exists, raise an error.

    Returns
    -------
    out_path : str
        Full path to the written raster.
    """
    os.makedirs(output_dir, exist_ok=True)

    if not output_name.lower().endswith((".tif", ".tiff")):
        output_name = f"{output_name}.tif"

    out_path = os.path.join(output_dir, output_name)

    if os.path.exists(out_path):
        if overwrite:
            os.remove(out_path)
        else:
            raise FileExistsError(f"Output already exists: {out_path}")

    with rasterio.open(raster_path) as src:
        profile = src.profile.copy()

        # Keep only first band, preserve georeferencing/metadata/profile
        profile.update(count=1)

        dtype = np.dtype(profile["dtype"])
        h, w = src.height, src.width

        # Validate fill_value is representable (mainly for integer rasters)
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            if not (info.min <= int(fill_value) <= info.max):
                raise ValueError(
                    f"fill_value={fill_value} outside range of {dtype} ({info.min}..{info.max})."
                )
        elif np.issubdtype(dtype, np.floating):
            if not np.isfinite(fill_value):
                raise ValueError(f"fill_value must be finite for float rasters; got {fill_value!r}.")
        else:
            raise TypeError(f"Unsupported dtype: {dtype}")

        # Create and write filled band (blockwise to support large rasters)
        with rasterio.open(out_path, "w", **profile) as dst:
            for _, window in src.block_windows(1):
                data = np.full((window.height, window.width), fill_value, dtype=dtype)
                dst.write(data, 1, window=window)

    print(
        f"get_pixel_raster_func complete. An empty raster (filled with {fill_value}) "
        f"with CRS {profile.get('crs')} was saved to: {out_path}"
    )
    return out_path



###############################################################################
## QUICKPLOT FUNC -> CHANGE LATER
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import plotting_extent

def quick_rgb_raster_plot_func(
    raster_path,
    geopackages=None,           # list of GeoDataFrames (or a single GeoDataFrame)
    rgb_idx=(0, 1, 2),          # indices in raster for R,G,B (adjust to your band order)
    stretch=(2, 98),            # percentile stretch (pmin, pmax)
    boundary_color="yellow",
    boundary_linewidth=2,
    figsize=(12, 12),
    title=None,
):
    """
    Quick plot of a (mosaic) raster with optional vector boundaries.
    Raises ValueError if any vector CRS != raster CRS.

    Parameters
    ----------
    raster_path : str
        Path to raster readable by rasterio.
    geopackages : GeoDataFrame or list[GeoDataFrame] or None
        One or multiple GeoDataFrames to overlay (boundary only).
    """
    if geopackages is None:
        geopackages = []
    elif not isinstance(geopackages, (list, tuple)):
        geopackages = [geopackages]

    with rasterio.open(raster_path) as src:
        arr = src.read()  # (bands, H, W)
        extent = plotting_extent(src)
        raster_crs = src.crs

    # CRS checks (no auto-reproject; error if mismatch)
    for i, gdf in enumerate(geopackages):
        if getattr(gdf, "crs", None) is None:
            raise ValueError(f"GeoDataFrame {i} has no CRS set (gdf.crs is None).")
        if gdf.crs != raster_crs:
            raise ValueError(
                f"CRS mismatch for GeoDataFrame {i}: gdf.crs={gdf.crs} vs raster_crs={raster_crs}. "
                "Reproject the GeoDataFrame to the raster CRS first."
            )

    fig, ax = plt.subplots(figsize=figsize)

    if arr.shape[0] >= 3:
        pmin, pmax = stretch
        rgb = np.stack([arr[rgb_idx[0]], arr[rgb_idx[1]], arr[rgb_idx[2]]], axis=-1).astype(np.float32)

        # percentile stretch per channel
        out = np.zeros_like(rgb, dtype=np.float32)
        for c in range(3):
            lo, hi = np.percentile(rgb[..., c], (pmin, pmax))
            if hi > lo:
                out[..., c] = (rgb[..., c] - lo) / (hi - lo)
        out = np.clip(out, 0, 1)

        ax.imshow(out, extent=extent)
    else:
        ax.imshow(arr[0], extent=extent, cmap="gray")

    for gdf in geopackages:
        gdf.boundary.plot(ax=ax, color=boundary_color, linewidth=boundary_linewidth)

    ax.set_title(title or "Raster quick plot")
    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")
    plt.tight_layout()
    return fig, ax



###############################################################################
## EXPORT PIXELS AS POLYGONS
###############################################################################
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape


def raster_pixels_to_gpkg(
    raster_path: str | Path,
    gpkg_path: str | Path,
    layer_name: str = "pixels",
    band: int = 1,
    drop_nodata: bool = True,
    use_mask: bool = True,
    chunk_size: int | None = None,
    driver: str = "GPKG",
) -> Path:
    """
    Convert a raster to per-pixel polygons and write them to a GeoPackage.

    Notes
    -----
    - This writes one polygon per pixel (not dissolved by value).
    - For large rasters this can be very big/slow. If you only want pixels
      matching a condition, pre-mask the raster first.

    Parameters
    ----------
    raster_path : path to input raster (e.g., GeoTIFF)
    gpkg_path : output .gpkg path
    layer_name : GeoPackage layer name
    band : raster band index (1-based)
    drop_nodata : if True, exclude nodata pixels
    use_mask : if True, use dataset mask for valid data
    chunk_size : optional tile size (in pixels) for writing in chunks
                 (None = process whole raster at once)
    driver : OGR driver (default "GPKG")

    Returns
    -------
    Path to written gpkg.
    """
    raster_path = Path(raster_path)
    gpkg_path = Path(gpkg_path)
    gpkg_path.parent.mkdir(parents=True, exist_ok=True)

    if gpkg_path.suffix.lower() != ".gpkg":
        raise ValueError("gpkg_path must end with .gpkg")

    def _write_gdf(gdf: gpd.GeoDataFrame, mode: str):
        # mode: "w" for first write, "a" for append
        gdf.to_file(gpkg_path, layer=layer_name, driver=driver, mode=mode)

    first_write = True

    with rasterio.open(raster_path) as src:
        crs = src.crs
        if crs is None:
            raise ValueError("Input raster has no CRS. Cannot write georeferenced polygons.")
        # If you specifically expect EPSG:32631, you can enforce/check it:
        # if crs.to_epsg() != 32631: raise ValueError(f"Expected EPSG:32631, got {crs}")

        nodata = src.nodata

        if chunk_size is None:
            data = src.read(band)
            mask = src.dataset_mask().astype(bool) if use_mask else np.ones(data.shape, dtype=bool)
            if drop_nodata and nodata is not None:
                mask &= (data != nodata)

            # shapes() yields polygons of connected regions with same value.
            # To force per-pixel polygons, we make every pixel unique by pairing (row,col,value)
            # BUT rasterio.features.shapes doesn't support multi-field values.
            # Instead: build per-pixel boxes directly (fast enough for moderate rasters).
            rows, cols = np.where(mask)
            if rows.size == 0:
                # write empty layer
                empty = gpd.GeoDataFrame({"value": [], "row": [], "col": []}, geometry=[], crs=crs)
                _write_gdf(empty, mode="w")
                return gpkg_path

            # compute bounds per pixel using transform
            t = src.transform
            # pixel (col,row) corner coords:
            # x_left = t.c + col*t.a + row*t.b ; y_top = t.f + col*t.d + row*t.e
            # assuming north-up rasters: t.b=t.d=0, t.a=px_w, t.e=-px_h
            # We still use rasterio.transform.xy for correctness.
            from rasterio.transform import xy
            from shapely.geometry import box

            geoms = []
            vals = data[rows, cols]
            for r, c, v in zip(rows, cols, vals):
                x_ul, y_ul = xy(t, r, c, offset="ul")
                x_lr, y_lr = xy(t, r, c, offset="lr")
                geoms.append(box(x_ul, y_lr, x_lr, y_ul))

            gdf = gpd.GeoDataFrame(
                {"value": vals.tolist(), "row": rows.tolist(), "col": cols.tolist()},
                geometry=geoms,
                crs=crs,
            )
            _write_gdf(gdf, mode="w")
            return gpkg_path

        # Chunked processing (recommended for large rasters)
        for row_off in range(0, src.height, chunk_size):
            h = min(chunk_size, src.height - row_off)
            for col_off in range(0, src.width, chunk_size):
                w = min(chunk_size, src.width - col_off)

                window = rasterio.windows.Window(col_off=col_off, row_off=row_off, width=w, height=h)
                data = src.read(band, window=window)
                mask = src.dataset_mask(window=window).astype(bool) if use_mask else np.ones(data.shape, dtype=bool)
                if drop_nodata and nodata is not None:
                    mask &= (data != nodata)

                rows, cols = np.where(mask)
                if rows.size == 0:
                    continue

                # convert window-local row/col to global
                rows_g = rows + row_off
                cols_g = cols + col_off

                from rasterio.transform import xy
                from shapely.geometry import box

                t = src.transform
                geoms = []
                vals = data[rows, cols]
                for r, c, v in zip(rows_g, cols_g, vals):
                    x_ul, y_ul = xy(t, r, c, offset="ul")
                    x_lr, y_lr = xy(t, r, c, offset="lr")
                    geoms.append(box(x_ul, y_lr, x_lr, y_ul))

                gdf = gpd.GeoDataFrame(
                    {"value": vals.tolist(), "row": rows_g.tolist(), "col": cols_g.tolist()},
                    geometry=geoms,
                    crs=crs,
                )

                _write_gdf(gdf, mode="w" if first_write else "a")
                first_write = False

    # If we never wrote anything (all nodata), write empty layer
    if first_write:
        empty = gpd.GeoDataFrame({"value": [], "row": [], "col": []}, geometry=[], crs="EPSG:32631")
        _write_gdf(empty, mode="w")

    return gpkg_path