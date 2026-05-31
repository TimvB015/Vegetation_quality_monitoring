from __future__ import annotations

################################################################################
## IMPORT RASTER FUNCTION
################################################################################
from pathlib import Path
import rasterio

def import_raster_func(path, band=None, masked=True, return_meta=True, return_src=False):
    """
    Import a raster for later visualization/calculations.

    Parameters
    ----------
    path : str | Path
        Path to raster file.
    band : int | None
        Band number to read (1-based). If None, reads all bands (shape: [count, rows, cols]).
    masked : bool
        If True, returns a numpy.ma.MaskedArray with NoData masked.
    return_meta : bool
        If True, also returns a metadata dict (transform, crs, bounds, etc.).
    return_src : bool
        If True, also returns an *open* rasterio dataset. You must close it yourself.

    Returns
    -------
    arr : numpy.ndarray or numpy.ma.MaskedArray
    meta : dict (optional)
    src  : rasterio.io.DatasetReader (optional, only if return_src=True)
    """
    path = Path(path)

    if return_src:
        src = rasterio.open(path)
        arr = src.read(masked=masked) if band is None else src.read(band, masked=masked)
        meta = None
        if return_meta:
            meta = {
                "path": str(path),
                "crs": src.crs,
                "transform": src.transform,
                "nodata": src.nodata,
                "bounds": src.bounds,
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "dtype": src.dtypes,
                "res": src.res,
                "profile": src.profile,
            }
        return (arr, meta, src) if return_meta else (arr, src)

    # Safe default: file is closed after reading, you keep only array + metadata
    with rasterio.open(path) as src:
        arr = src.read(masked=masked) if band is None else src.read(band, masked=masked)
        meta = None
        if return_meta:
            meta = {
                "path": str(path),
                "crs": src.crs,
                "transform": src.transform,
                "nodata": src.nodata,
                "bounds": src.bounds,
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "dtype": src.dtypes,
                "res": src.res,
                "profile": src.profile,
            }
    return (arr, meta) if return_meta else arr



################################################################################
## RETRIEVE NODATA VALUE
################################################################################
from typing import Optional, Union
import rasterio


def get_nodata_value(
    raster_path: str = None,
    raster_profile: dict = None,
) -> Optional[Union[int, float]]:
    """
    Automatically retrieve the nodata value from a raster file or profile.
    
    Parameters
    ----------
    raster_path : str, optional
        Path to the raster file
    raster_profile : dict, optional
        Raster profile dictionary (from rasterio or import_raster_func)
        Must contain 'nodata' key
    
    Returns
    -------
    nodata_value : int, float, or None
        The nodata value from the raster
        Returns None if no nodata value is set
    
    Raises
    ------
    ValueError
        If neither raster_path nor raster_profile is provided
    
    Examples
    --------
    >>> # From file path
    >>> nodata = get_nodata_value(raster_path="stacked.tif")
    >>> print(f"NoData value: {nodata}")
    NoData value: 255
    
    >>> # From profile (after loading with import_raster_func)
    >>> raster_data = import_raster_func("stacked.tif")
    >>> nodata = get_nodata_value(raster_profile=raster_data['profile'])
    >>> print(f"NoData value: {nodata}")
    NoData value: 255
    """
    
    if raster_path is None and raster_profile is None:
        raise ValueError("Either raster_path or raster_profile must be provided")
    
    # Option 1: Get from file path
    if raster_path is not None:
        with rasterio.open(raster_path) as src:
            nodata_value = src.nodata
            print(f"Retrieved nodata value from file: {nodata_value}")
            return nodata_value
    
    # Option 2: Get from profile dictionary
    if raster_profile is not None:
        nodata_value = raster_profile.get('nodata', None)
        print(f"Retrieved nodata value from profile: {nodata_value}")
        return nodata_value
    


################################################################################
## QUICK RASTER CHECK
################################################################################
from pathlib import Path
import rasterio


def print_raster_info(
        raster_path, 
        show_tags=True, 
        show_band_stats=False, 
        show_band_names=True,
):
    """
    Print key metadata for a raster.

    Parameters
    ----------
    raster_path : str | Path
    show_tags : bool
        If True, prints dataset-level tags.
    show_band_stats : bool
        If True, prints basic per-band stats (min/max/mean) by reading data (can be slow).
    show_band_names : bool
        If True, prints per-band names/descriptions (and common alternatives if missing).
    """
    raster_path = Path(raster_path)

    with rasterio.open(raster_path) as src:
        print(f"Path:        {raster_path}")
        print(f"Driver:      {src.driver}")
        print(f"CRS:         {src.crs}")
        print(f"Transform:   {src.transform}")
        print(f"Bounds:      {src.bounds}")
        print(f"Width/Height:{src.width} x {src.height} px")
        print(f"Count:       {src.count} band(s)")
        print(f"Dtype(s):    {src.dtypes}")
        print(f"Nodata:      {src.nodata}")
        print(f"Res (x,y):   {src.res}")
        print(f"Units:       {getattr(src.crs, 'linear_units', None) if src.crs else None}")
        print(f"Is tiled:    {src.is_tiled}")
        print(f"Block shapes:{src.block_shapes[:min(5, len(src.block_shapes))]}{' ...' if len(src.block_shapes) > 5 else ''}")
        print(f"Compress:    {src.profile.get('compress')}")
        print(f"Interleave:  {src.profile.get('interleave')}")
        print(f"Colorinterp: {tuple(src.colorinterp) if src.count else None}")

        if show_band_names:
            desc = src.descriptions 
            print("\nBand names / descriptions:")
            for i in range(1, src.count + 1):
                d = desc[i - 1] if desc else None
                d = d if (d is not None and str(d).strip() != "") else "(no description)"
                print(f"  Band {i}: {d}")

            has_band_tags = False
            band_tags_preview = {}
            for i in range(1, src.count + 1):
                bt = src.tags(i)
                bt_clean = {k: v for k, v in bt.items() if v not in ("", None)}
                if bt_clean:
                    has_band_tags = True
                    band_tags_preview[i] = bt_clean

            if has_band_tags:
                print("\nBand tags (per-band):")
                for i in range(1, src.count + 1):
                    bt = band_tags_preview.get(i)
                    if not bt:
                        continue
                    print(f"  Band {i}:")
                    for k in sorted(bt):
                        print(f"    {k}: {bt[k]}")

        if show_tags:
            tags = src.tags()
            print("\nTags:")
            for k in sorted(tags):
                print(f"  {k}: {tags[k]}")

        if show_band_stats:
            import numpy as np

            print("\nBand statistics (masked by nodata if set):")
            for b in range(1, src.count + 1):
                data = src.read(b, masked=True)
                mn = float(data.min()) if data.count() else None
                mx = float(data.max()) if data.count() else None
                mean = float(data.mean()) if data.count() else None
                print(f"  Band {b}: min={mn}, max={mx}, mean={mean}")



################################################################################
## Merge raster pixels and export to GPKG
################################################################################
import numpy as np
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape, Polygon, MultiPolygon

def raster_bordering_pixels_to_polygon(
    raster_path,
    band_index=1,
    connectivity=8,
    include_values=None,
    exclude_values=None,
    drop_nodata=True,
):
    """
    Polygonize a raster band into connected regions and return them as a GeoDataFrame.

    The function groups *bordering* (connected) pixels that share the same raster
    value into a single region geometry. Each output row represents one connected
    region. Geometries are coerced to ``shapely.geometry.MultiPolygon``.

    Internally this uses :func:`rasterio.features.shapes`, which performs the
    connected-component grouping based on the chosen connectivity (4- or 8-neighbour).

    Parameters
    ----------
    raster_path : str | os.PathLike
        Path to the raster dataset.
    band_index : int, default 1
        1-based band index to read from the raster.
    connectivity : int, default 8
        Pixel connectivity used to define bordering pixels. Must be ``4`` or ``8``.
        With 4-connectivity only edge-adjacent pixels are connected; with 8-connectivity
        edge- and corner-adjacent pixels are connected.
    include_values : iterable, optional
        If provided, only regions whose raster value is in this collection are kept
        (e.g. ``[1]`` for a binary raster). If ``None``, all values are eligible.
    exclude_values : iterable, optional
        Values to omit from the result (e.g. ``[0]``). Applied after ``include_values``.
    drop_nodata : bool, default True
        If ``True`` and the raster has a defined nodata value, nodata pixels are masked
        out and will not be polygonized.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with columns:

        - ``value``: the raster value of the region (typically ``int`` where possible)
        - ``region_id``: sequential integer identifier (0..n-1)
        - ``geometry``: MultiPolygon geometry in the raster CRS

        The CRS is taken from the source dataset.

    Raises
    ------
    rasterio.errors.RasterioIOError
        If the raster cannot be opened (e.g. path not found, unsupported format).
    IndexError
        If ``band_index`` is out of range for the dataset.
    ValueError
        If ``connectivity`` is not supported (must be 4 or 8), or if invalid values
        are passed to ``include_values``/``exclude_values`` such that they cannot be
        converted into a set.
    """
    exclude_values = set([] if exclude_values is None else exclude_values)
    include_values = None if include_values is None else set(include_values)

    with rasterio.open(raster_path) as src:
        band = src.read(band_index)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

    valid_mask = np.ones(band.shape, dtype=bool)
    if drop_nodata and nodata is not None:
        valid_mask &= (band != nodata)

    geoms, vals, region_ids = [], [], []
    rid = 0

    for geom_mapping, val in shapes(band, mask=valid_mask, transform=transform, connectivity=connectivity):
        v = int(val) if np.isfinite(val) and float(val).is_integer() else val

        if include_values is not None and v not in include_values:
            continue
        if v in exclude_values:
            continue

        g = shape(geom_mapping)
        if isinstance(g, Polygon):
            g = MultiPolygon([g])
        elif isinstance(g, MultiPolygon):
            pass
        else:
            polys = [p for p in getattr(g, "geoms", []) if isinstance(p, Polygon)]
            g = MultiPolygon(polys)

        geoms.append(g)
        vals.append(v)
        region_ids.append(rid)
        rid += 1

    return gpd.GeoDataFrame(
        {"value": vals, "region_id": region_ids},
        geometry=geoms,
        crs=crs
    )