from __future__ import annotations

################################################################################
## CHECK GPKG LAYERS FUNCTION
################################################################################
from pathlib import Path

def check_gpkg_layers_func(gpkg_path, engine="fiona", return_type="list"):
    """
    Inspect layers in a GeoPackage (.gpkg).

    Parameters
    ----------
    gpkg_path : str or path-like
        Path to the GeoPackage.
    engine : {"fiona", "pyogrio"}, default "fiona"
        Backend used to list layers.
    return_type : {"list", "dict"}, default "list"
        - "list": returns a list of layer names
        - "dict": returns a dict with keys:
            * "path": str
            * "layers": list[str]
            * "n_layers": int

    Returns
    -------
    list[str] or dict
        Layer names (or a small summary dict).

    Raises
    ------
    FileNotFoundError
        If gpkg_path does not exist.
    ValueError
        If the file extension is not .gpkg.
    """
    gpkg_path = Path(gpkg_path)

    if not gpkg_path.exists():
        raise FileNotFoundError(f"GeoPackage not found: {gpkg_path}")

    if gpkg_path.suffix.lower() != ".gpkg":
        raise ValueError(f"Expected a .gpkg file, got: {gpkg_path.suffix}")

    engine = engine.lower().strip()
    if engine == "fiona":
        import fiona
        layers = list(fiona.listlayers(str(gpkg_path)))
    elif engine == "pyogrio":
        import pyogrio
        layers = [name for name, _geom in pyogrio.list_layers(str(gpkg_path))]
    else:
        raise ValueError("engine must be 'fiona' or 'pyogrio'")

    if return_type == "list":
        return layers
    elif return_type == "dict":
        return {"path": str(gpkg_path), "layers": layers, "n_layers": len(layers)}
    else:
        raise ValueError("return_type must be 'list' or 'dict'")
    


################################################################################
## IMPORT GPKG
################################################################################
import geopandas as gpd

def import_gpkg_func(
    gdf_path,
    layer=None,
    index_col=None,
    reindex=None,
    cols_to_keep=None,
    crs=None,
):
    """
    Load a GeoDataFrame from the given path (e.g., GPKG, Shapefile, GeoJSON),
    with optional index handling, column selection, and CRS assignment/reprojection.

    Parameters
    ----------
    gdf_path : str or path-like
        Path to the file to read with geopandas.read_file.
    layer : str or int, optional
        Layer name or layer index to read (relevant for multi-layer datasets like GeoPackage).
        If None, geopandas.read_file uses its default behavior (often the first layer found),
        which can raise a warning if multiple layers are present.
    index_col : str, optional
        Name of a column to set as index after loading. If provided and not found in the
        GeoDataFrame, a ValueError is raised.
    reindex : str, optional
        Base name for creating a new index like 'reindex_1', 'reindex_2', ...
        If None, the existing index (possibly set by index_col) is kept.
    cols_to_keep : iterable of str, optional
        Column names to keep (geometry is always preserved if present).
    crs : any, optional
        Desired CRS for the returned GeoDataFrame. Accepts anything geopandas understands
        (e.g. "EPSG:28992", an EPSG int like 28992, or a pyproj.CRS).
        - If the input has no CRS, it is assigned with `set_crs(crs)`.
        - If the input has a CRS, the data are reprojected with `to_crs(crs)`.

    Behavior
    --------
    - Reads the dataset from gdf_path; if 'layer' is provided it is passed to read_file.
    - CRS handling (if crs is provided):
        * If gdf.crs is None: assign with set_crs(crs).
        * Else: reproject to crs using to_crs(crs).
    - Index handling:
        * If index_col is provided and exists, set it as the index.
        * Else if a column named 'index' exists, set it as index.
        * Else ensure the index is named 'index'.
    - cols_to_keep is applied after index handling; geometry is always kept if present.
    - Reindexing (if requested) is performed last and replaces the index with generated names.

    Raises
    ------
    ValueError
        If index_col is provided but not found in the GeoDataFrame columns.

    Returns
    -------
    geopandas.GeoDataFrame
        The loaded (and optionally reprojected/filtered/reindexed) GeoDataFrame.
    """
    if layer is None:
        gdf = gpd.read_file(gdf_path)
    else:
        gdf = gpd.read_file(gdf_path, layer=layer)

    # CRS handling
    if crs is not None:
        if gdf.crs is None:
            gdf = gdf.set_crs(crs)
            print("Input data has no CRS; assigned provided CRS without transformation.")
        elif gdf.crs != crs:
            gdf = gdf.to_crs(crs)
            print(f"Input data reprojected to {crs}.")

    geom_col = gdf.geometry.name if getattr(gdf, "geometry", None) is not None else "geometry"

    # Index handling
    if index_col is not None:
        if index_col not in gdf.columns:
            raise ValueError(
                f"Requested index_col '{index_col}' not found in GeoDataFrame columns: {list(gdf.columns)}"
            )
        gdf = gdf.set_index(index_col, drop=True)
        gdf.index.name = "index"
    else:
        if "index" in gdf.columns:
            gdf = gdf.set_index("index", drop=True)
        else:
            if gdf.index.name != "index":
                gdf.index.name = "index"

    # Column selection
    if cols_to_keep is not None:
        cols = [c for c in cols_to_keep if c in gdf.columns]
        if geom_col in gdf.columns and geom_col not in cols:
            cols.append(geom_col)
        gdf = gdf[cols]

    # Optional reindexing
    if reindex is not None:
        gdf.index = [f"{reindex}_{i+1}" for i in range(len(gdf))]
        gdf.index.name = "index"

    return gdf



################################################################################
## REPROJECT GPKG
################################################################################
import fiona
from pathlib import Path
from typing import Dict, Optional, Union

import geopandas as gpd


def reproject_gpkg_func(
    gdf: Union[str, Path, gpd.GeoDataFrame, Dict[str, gpd.GeoDataFrame]],
    output_epsg: Union[int, str],
    output_gpkg_path: Optional[Union[str, Path]] = None,
    *,
    layer: Optional[str] = None,
    overwrite: bool = True,
    dst_layer_suffix: str = "",
    require_crs: bool = True,
    fix_invalid: bool = True,
    return_gdf: bool = False,
) -> Union[
    gpd.GeoDataFrame,
    Dict[str, gpd.GeoDataFrame],
    Path,
    tuple[Path, gpd.GeoDataFrame],
    tuple[Path, Dict[str, gpd.GeoDataFrame]],
]:
    """
    Reproject GeoPackage/GeoDataFrame data to a target EPSG.

    Output handling
    ---------------
    - If `output_gpkg_path` is provided, the function writes a GeoPackage to that path.
    - If `output_gpkg_path` is None, no file is written and the reprojected GeoDataFrame(s)
      are returned directly.

    Overwrite behavior (only when writing)
    --------------------------------------
    The output GeoPackage is overwritten by default (`overwrite=True`). If `overwrite=True`
    and the output file already exists, it is deleted before writing.

    Supported input modes
    ---------------------
    1) GeoPackage path (str/Path)
       - Reads all layers in the input .gpkg
       - Reprojects and writes/returns all *spatial* layers (layers with a geometry column)
       - Non-spatial tables are skipped (GeoPandas does not reliably round-trip them)

    2) Single GeoDataFrame
       - Reprojects the GeoDataFrame and writes/returns it as one layer

    3) Dict[str, GeoDataFrame]
       - Reprojects each GeoDataFrame and writes/returns each as its own layer

    Parameters
    ----------
    gdf : str | pathlib.Path | geopandas.GeoDataFrame | dict[str, geopandas.GeoDataFrame]
        Input data:
        - Path to a .gpkg file, OR
        - a GeoDataFrame, OR
        - a dict mapping layer names to GeoDataFrames.
    output_epsg : int | str
        Target CRS EPSG code. Examples: 28992, "28992", or "EPSG:28992".
    output_gpkg_path : str | pathlib.Path | None, optional
        Output path for the resulting GeoPackage (.gpkg). If None, nothing is written.
    layer : str | None, optional
        Layer name to use when `gdf` is a single GeoDataFrame.
        If None, defaults to "layer".
    overwrite : bool, default True
        Only used when writing. If True, deletes an existing output file before writing.
        If False and the output exists, raises FileExistsError.
    dst_layer_suffix : str, default ""
        Optional suffix appended to each output layer name (when writing).
    require_crs : bool, default True
        If True, raises an error if an input layer has no CRS (`gdf.crs is None`).
    fix_invalid : bool, default True
        If True, attempts a simple geometry fix (`geometry.buffer(0)`) before reprojection.
        If the fix fails, reprojection still proceeds with the original geometries.
    return_gdf : bool, default False
        Only relevant when `output_gpkg_path` is provided.
        If False, returns only the output GeoPackage path.
        If True, also returns the reprojected GeoDataFrame(s).

    Returns
    -------
    If output_gpkg_path is None:
        geopandas.GeoDataFrame or dict[str, geopandas.GeoDataFrame]
            The reprojected data.

    If output_gpkg_path is provided:
        pathlib.Path
            If return_gdf=False: path to the written output GeoPackage.
        (pathlib.Path, geopandas.GeoDataFrame) or (pathlib.Path, dict[str, geopandas.GeoDataFrame])
            If return_gdf=True: (output_path, reprojected_data).

    Raises
    ------
    FileNotFoundError
        If the input GeoPackage path does not exist.
    FileExistsError
        If output exists, overwrite=False, and writing is requested.
    ValueError
        If no layers are found, no spatial layers are written, CRS is missing (when required),
        or input data is otherwise invalid.
    TypeError
        If `gdf` is not one of the supported input types.
    """
    # Normalize EPSG input
    if isinstance(output_epsg, str):
        s = output_epsg.strip().upper()
        output_epsg = int(s.split(":")[1]) if s.startswith("EPSG:") else int(s)

    do_write = output_gpkg_path is not None
    output_gpkg = Path(output_gpkg_path) if do_write else None

    if do_write:
        assert output_gpkg is not None
        output_gpkg.parent.mkdir(parents=True, exist_ok=True)

        # Handle output file existence
        if output_gpkg.exists():
            if overwrite:
                output_gpkg.unlink()
            else:
                raise FileExistsError(f"Output already exists (set overwrite=True): {output_gpkg}")

    def _validate_and_reproject(gdf: gpd.GeoDataFrame, layer_name: str) -> gpd.GeoDataFrame:
        if "geometry" not in gdf.columns or gdf.geometry is None:
            raise ValueError(f"Layer '{layer_name}' has no geometry column; cannot reproject.")

        if require_crs and gdf.crs is None:
            raise ValueError(
                f"Layer '{layer_name}' has no CRS defined. "
                "Assign the correct source CRS before reprojecting."
            )

        if fix_invalid:
            try:
                gdf = gdf.copy()
                gdf["geometry"] = gdf.geometry.buffer(0)
            except Exception:
                pass

        if gdf.crs is None:
            raise ValueError(f"Layer '{layer_name}' CRS is None; cannot safely transform without a source CRS.")

        return gdf.to_crs(epsg=output_epsg)

    def _write_layer(gdf: gpd.GeoDataFrame, layer_name: str) -> None:
        assert output_gpkg is not None
        gdf.to_file(output_gpkg, layer=layer_name, driver="GPKG")

    # Case 1: input is a gpkg path
    if isinstance(gdf, (str, Path)):
        input_gpkg = Path(gdf)
        if not input_gpkg.exists():
            raise FileNotFoundError(f"Input GeoPackage not found: {input_gpkg}")
        if input_gpkg.suffix.lower() != ".gpkg":
            raise ValueError(f"Input must be a .gpkg file: {input_gpkg}")

        layers = gpd.io.file.fiona.listlayers(str(input_gpkg))
        if not layers:
            raise ValueError(f"No layers found in GeoPackage: {input_gpkg}")

        returned: Dict[str, gpd.GeoDataFrame] = {}
        wrote_any = False

        for lyr in layers:
            gdf = gpd.read_file(input_gpkg, layer=lyr)

            # Skip non-spatial tables safely
            if "geometry" not in gdf.columns or gdf.geometry is None:
                continue

            out_layer = f"{lyr}{dst_layer_suffix}"
            out = _validate_and_reproject(gdf, lyr)

            if do_write:
                _write_layer(out, out_layer)
                wrote_any = True

            returned[out_layer] = out

        if not returned:
            raise ValueError(
                f"No spatial layers found in {input_gpkg}. "
                "If your GPKG contains only attribute tables, GeoPandas won't write them."
            )

        if not do_write:
            return returned

        assert output_gpkg is not None
        return (output_gpkg, returned) if return_gdf else output_gpkg

    # Case 2: input is a single GeoDataFrame
    if isinstance(gdf, gpd.GeoDataFrame):
        out_layer = f"{(layer or 'layer')}{dst_layer_suffix}"
        out = _validate_and_reproject(gdf, out_layer)

        if not do_write:
            return out

        _write_layer(out, out_layer)
        assert output_gpkg is not None
        return (output_gpkg, out) if return_gdf else output_gpkg

    # Case 3: input is a dict of named GeoDataFrames
    if isinstance(gdf, dict):
        if not gdf:
            raise ValueError("gdf dict is empty.")

        returned: Dict[str, gpd.GeoDataFrame] = {}

        for lyr, gdf in gdf.items():
            if not isinstance(gdf, gpd.GeoDataFrame):
                raise TypeError(f"Value for key '{lyr}' is not a GeoDataFrame.")

            out_layer = f"{lyr}{dst_layer_suffix}"
            out = _validate_and_reproject(gdf, lyr)

            if do_write:
                _write_layer(out, out_layer)

            returned[out_layer] = out

        if not do_write:
            return returned

        assert output_gpkg is not None
        return (output_gpkg, returned) if return_gdf else output_gpkg

    raise TypeError(
        "gdf must be one of: (str|Path to .gpkg), GeoDataFrame, or dict[str, GeoDataFrame]."
    )



################################################################################
## CLIP GPKG BY GPKG
################################################################################
import os
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd


GpkgOrPath = Union[str, Path, gpd.GeoDataFrame]


def clip_gpkg_with_gpkg(
    gpkg_or_path: GpkgOrPath,
    clip_gpkg_or_path: GpkgOrPath,
    layer: Optional[str] = None,
    clip_layer: Optional[str] = None,
    dissolve: bool = True,
    keep_geom_type: bool = True,
    out_path: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
    return_in_memory: bool = False,
) -> Optional[gpd.GeoDataFrame]:
    """
    Clip a vector dataset (GeoPackage/GeoDataFrame) to the extent of another vector
    dataset (GeoPackage/GeoDataFrame).

    Inputs can be either file paths or already-loaded GeoDataFrames.

    Behavior
    --------
    - If `out_path` is provided, the clipped result is written to that GeoPackage path.
    - If `overwrite=False` (default) and `out_path` already exists:
        * the existing file is read and a message is printed:
          "An already existing file is imported: <out_path>"
        * no clipping is performed
    - If `return_in_memory=True`, the clipped GeoDataFrame is returned.
      Otherwise returns None.

    Parameters
    ----------
    gpkg_or_path
        Input data to be clipped (GeoDataFrame or path to GeoPackage).
    clip_gpkg_or_path
        Clip geometry source (GeoDataFrame or path to GeoPackage).
    layer
        Layer name for `gpkg_or_path` when a path is provided.
    clip_layer
        Layer name for `clip_gpkg_or_path` when a path is provided.
    dissolve
        If True, dissolve clip geometries into a single geometry before clipping.
    keep_geom_type
        Passed to `geopandas.clip` (default True). Keeps only geometries of the same
        type as the input.
    out_path
        If provided, write the clipped output to this GeoPackage file path.
        (Note: this writes to a GeoPackage file; the layer name is set to `layer`
        if provided, otherwise "clipped".)
    overwrite
        If False (default) and `out_path` exists, the existing file is imported.
    return_in_memory
        If True, return the clipped GeoDataFrame (or the imported one if output exists).

    Returns
    -------
    Optional[geopandas.GeoDataFrame]
        The clipped (or imported) GeoDataFrame if `return_in_memory=True`, else None.

    Raises
    ------
    ValueError
        If inputs are empty, have missing CRS, or contain no valid geometries.
    """
    # --- if output exists and we don't overwrite: import existing ---
    if out_path is not None:
        out_path = Path(out_path)
        if out_path.exists() and not overwrite:
            print(f"An already existing file is imported: {out_path}")
            existing = gpd.read_file(out_path)  # reads default/first layer
            return existing if return_in_memory else None

    # --- load input gdf ---
    if isinstance(gpkg_or_path, gpd.GeoDataFrame):
        gdf = gpkg_or_path.copy()
    else:
        gdf = gpd.read_file(str(gpkg_or_path), layer=layer) if layer else gpd.read_file(str(gpkg_or_path))

    if isinstance(clip_gpkg_or_path, gpd.GeoDataFrame):
        clip_gdf = clip_gpkg_or_path.copy()
    else:
        clip_gdf = (
            gpd.read_file(str(clip_gpkg_or_path), layer=clip_layer)
            if clip_layer
            else gpd.read_file(str(clip_gpkg_or_path))
        )

    if gdf.empty:
        raise ValueError("Input gpkg contains no features.")
    if clip_gdf.empty:
        raise ValueError("Clip gpkg contains no features.")

    gdf = gdf[(~gdf.geometry.is_empty) & (gdf.geometry.notna())].copy()
    clip_gdf = clip_gdf[(~clip_gdf.geometry.is_empty) & (clip_gdf.geometry.notna())].copy()
    if gdf.empty:
        raise ValueError("Input gpkg contains no non-empty geometries.")
    if clip_gdf.empty:
        raise ValueError("Clip gpkg contains no non-empty geometries.")

    if gdf.crs is None:
        raise ValueError("Input gpkg has no CRS defined.")
    if clip_gdf.crs is None:
        raise ValueError("Clip gpkg has no CRS defined.")

    # Reproject clip geometries to input CRS if needed
    if clip_gdf.crs != gdf.crs:
        clip_gdf = clip_gdf.to_crs(gdf.crs)

    # Fix invalid geometries (minimal, safe approach)
    invalid_in = ~gdf.geometry.is_valid
    if invalid_in.any():
        try:
            gdf.loc[invalid_in, "geometry"] = gdf.loc[invalid_in, "geometry"].make_valid()
        except Exception:
            gdf.loc[invalid_in, "geometry"] = gdf.loc[invalid_in, "geometry"].buffer(0)
        gdf = gdf[(~gdf.geometry.is_empty) & (gdf.geometry.notna())].copy()
        gdf = gdf[gdf.geometry.is_valid].copy()

    invalid_clip = ~clip_gdf.geometry.is_valid
    if invalid_clip.any():
        try:
            clip_gdf.loc[invalid_clip, "geometry"] = clip_gdf.loc[invalid_clip, "geometry"].make_valid()
        except Exception:
            clip_gdf.loc[invalid_clip, "geometry"] = clip_gdf.loc[invalid_clip, "geometry"].buffer(0)
        clip_gdf = clip_gdf[(~clip_gdf.geometry.is_empty) & (clip_gdf.geometry.notna())].copy()
        clip_gdf = clip_gdf[clip_gdf.geometry.is_valid].copy()

    if gdf.empty:
        raise ValueError("No valid input geometries remain after fixing.")
    if clip_gdf.empty:
        raise ValueError("No valid clip geometries remain after fixing.")

    # Dissolve clip geometries if requested
    if dissolve:
        try:
            clip_geom = clip_gdf.geometry.union_all()
        except Exception:
            clip_geom = clip_gdf.unary_union
        clip_mask = gpd.GeoDataFrame(geometry=[clip_geom], crs=gdf.crs)
    else:
        clip_mask = clip_gdf[["geometry"]].copy()

    # Clip
    clipped = gpd.clip(gdf, clip_mask, keep_geom_type=keep_geom_type)

    # Write if requested
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # If you want the output layer name to be something else, adjust here:
        out_layer = layer if layer else "clipped"

        # Writing: GeoPackage driver
        clipped.to_file(out_path, layer=out_layer, driver="GPKG")

    return clipped if return_in_memory else None



################################################################################
## PIXELIZE GPKG
################################################################################
from pathlib import Path
import geopandas as gpd
import pandas as pd


def pixelize_gpkg_func(
    pixels_gpkg: str | Path,
    clip_gpkg: str | Path,
    out_gpkg: str | Path,
    out_layer: str = "pixelized",
    *,
    pixels_layer: str | None = None,
    clip_layer: str | None = None,
    id_col: str = "index",
    full_pixels_only: bool = True,
    predicate_full: str = "within",
) -> Path:
    """
    Output:
      - keeps all clip attributes (except it rewrites `id_col` to add _1, _2, ...)
      - removes ALL pixel attributes
      - never creates 'level_0'
    """
    pixels_gpkg = Path(pixels_gpkg)
    clip_gpkg = Path(clip_gpkg)
    out_gpkg = Path(out_gpkg)
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)

    pixels = gpd.read_file(pixels_gpkg, layer=pixels_layer)
    clip = gpd.read_file(clip_gpkg, layer=clip_layer)

    if pixels.empty:
        raise ValueError("pixels layer is empty.")
    if clip.empty:
        raise ValueError("clip layer is empty.")
    if pixels.crs is None or clip.crs is None:
        raise ValueError("Both inputs must have a CRS.")
    if pixels.crs != clip.crs:
        clip = clip.to_crs(pixels.crs)

    if "level_0" in clip.columns:
        clip = clip.drop(columns=["level_0"])

    if id_col not in clip.columns:
        raise ValueError(
            f"id_col='{id_col}' not found in clip layer columns. "
            f"Available: {list(clip.columns)}"
        )

    if full_pixels_only and predicate_full not in {"within", "contains"}:
        raise ValueError("predicate_full must be 'within' or 'contains'.")

    _ = pixels.sindex  # build once

    out_parts: list[gpd.GeoDataFrame] = []

    for _, clip_row in clip.iterrows():
        geom = clip_row.geometry
        if geom is None or geom.is_empty:
            continue

        # candidates by bbox + spatial index
        cand_idx = pixels.sindex.query(geom, predicate="intersects")
        cand_geom = pixels.iloc[cand_idx].geometry  # geometry only => no pixel attrs

        if cand_geom.empty:
            continue

        # select/intersect
        if full_pixels_only:
            if predicate_full == "within":
                geoms = cand_geom[cand_geom.within(geom)].values
            else:  # contains
                geoms = cand_geom[cand_geom.apply(geom.contains)].values
        else:
            inter = gpd.overlay(
                gpd.GeoDataFrame(geometry=cand_geom.values, crs=pixels.crs),
                gpd.GeoDataFrame(geometry=[geom], crs=pixels.crs),
                how="intersection",
                keep_geom_type=True,
            )
            geoms = inter.geometry.values

        if len(geoms) == 0:
            continue

        # replicate ALL clip attrs to each pixel row
        attrs = clip_row.drop(labels=["geometry"]).to_dict()
        out = gpd.GeoDataFrame(
            {k: [v] * len(geoms) for k, v in attrs.items()},
            geometry=geoms,
            crs=pixels.crs,
        )

        # rewrite the chosen column with _runningnumber
        base = str(attrs[id_col])
        out[id_col] = [f"{base}_{i}" for i in range(1, len(out) + 1)]

        out_parts.append(out)

    if out_parts:
        out_all = gpd.GeoDataFrame(
            pd.concat(out_parts, axis=0, ignore_index=True),  # ignore_index avoids any index-level columns
            crs=pixels.crs,
        )
    else:
        out_all = gpd.GeoDataFrame(columns=[c for c in clip.columns if c != "geometry"], geometry=[], crs=pixels.crs)

    out_all.to_file(out_gpkg, layer=out_layer, driver="GPKG")
    return out_gpkg



################################################################################
## RENAMING THE GDF INDEX
################################################################################
import geopandas as gpd


def prefix_index(
    gdf: gpd.GeoDataFrame,
    prefix: str,
    index_name: str | None = None,
    drop_old_index: bool = True,
) -> gpd.GeoDataFrame:
    """
    Prefix the GeoDataFrame index with `prefix`.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input GeoDataFrame.
    prefix : str
        Prefix to prepend to each index value (e.g. "lgn_plusOW_").
    index_name : str | None
        If provided, sets `gdf.index.name` to this value.
        If None, leaves index name unchanged.
        If "" (empty string), clears the index name.
    drop_old_index : bool
        If True, replaces the index with the prefixed index based on current index values.
        If False, keeps the existing index and adds a new column named `index_name` (or "index")
        containing the prefixed IDs.

    Returns
    -------
    GeoDataFrame
        Copy of the input with updated index (or a new column if drop_old_index=False).
    """
    out = gdf.copy()

    prefixed = [f"{prefix}{idx}" for idx in out.index]

    if drop_old_index:
        out.index = prefixed
        if index_name is not None:
            out.index.name = None if index_name == "" else index_name
    else:
        col = index_name if (index_name not in (None, "")) else "index"
        out[col] = prefixed

    return out