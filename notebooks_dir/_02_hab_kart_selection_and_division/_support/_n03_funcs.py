from __future__ import annotations

################################################################################
## REMOVE OVERLAP BETWEEN HABITAT MAPPINNG AND LGN MAP
################################################################################
import re
import pandas as pd
import geopandas as gpd
import shapely
from shapely.ops import unary_union
from shapely.geometry import box
import warnings


def remove_habitat_overlap_from_lgn(
    habitat_gdf: gpd.GeoDataFrame,
    lgn_gdf: gpd.GeoDataFrame,
    year: int,
    habitat_year_col: str = "years",
    make_valid: bool = True,
    explode_parts: bool = True,
    grid_size: float | None = None,
    batch_size: int = 5000,
) -> gpd.GeoDataFrame:
    """
    Subtract habitat polygons (for a selected year) from LGN polygons and return the edited LGN GeoDataFrame.

    Workflow (silent; no logging):
      1) Filter habitat features by `year` using `habitat_year_col`
      2) Align CRS (habitat -> LGN CRS)
      3) Drop missing/empty geometries
      4) Optionally repair invalid geometries (`shapely.make_valid` if available, else `buffer(0)`)
      5) If bounding boxes do not overlap: return LGN unchanged
      6) Spatial prefilter using `geopandas.sjoin(..., intersects)` to find intersecting candidates
      7) Union only the intersecting habitat subset
      8) Run `difference()` only on intersecting LGN features (in batches)
      9) Optionally explode multipart geometries
     10) Return the result (no file writing)

    Notes
    -----
    - This version intentionally does NOT use `sindex.query_bulk`.
    - `grid_size` applies precision snapping (CRS units). Use with care: too large distorts, too small may not help.

    Parameters
    ----------
    habitat_gdf, lgn_gdf : geopandas.GeoDataFrame
        Input polygon layers. `habitat_gdf` must contain `habitat_year_col`.
    year : int
        Target year used to select habitat features.
    habitat_year_col : str, default "years"
        Column in `habitat_gdf` indicating applicable year(s). Matching rules:
          - int: equals `year`
          - float: equals `year` after int-cast (e.g., 2018.0 -> 2018)
          - list/tuple/set: contains `year` (or string-equal)
          - string/other: contains `year` as a whole token (e.g. "2018;2019" matches 2018)
    make_valid : bool, default True
        Attempt to repair invalid geometries before overlay operations.
    explode_parts : bool, default True
        Explode multipart results into singlepart features.
    grid_size : float | None, default None
        Optional precision grid size in CRS units.
    batch_size : int, default 5000
        Number of LGN candidate features to process per difference batch.

    Returns
    -------
    geopandas.GeoDataFrame
        LGN GeoDataFrame with habitat overlap removed (same CRS as input LGN).

    Raises
    ------
    Warning
        If `habitat_year_col` is missing or no habitat features match `year`.
    """

    def _drop_missing_empty(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        geom = gdf.geometry
        non_missing = pd.notna(geom.to_numpy())
        non_empty = (~geom.is_empty).fillna(False).to_numpy()
        return gdf[non_missing & non_empty].copy()

    def _year_mask(series: pd.Series, y: int) -> pd.Series:
        if series.dtype.kind in "iu":
            return series.eq(y)
        if series.dtype.kind == "f":
            return series.fillna(-1).astype(int).eq(y)

        year_pat = re.compile(rf"(?<!\d){re.escape(str(y))}(?!\d)")

        def has_year(v):
            if v is None:
                return False
            if isinstance(v, (list, tuple, set)):
                return y in v or str(y) in {str(x) for x in v}
            return bool(year_pat.search(str(v)))

        return series.apply(has_year)

    def _spatial_join_intersects(left: gpd.GeoDataFrame, right: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        try:
            return gpd.sjoin(left, right, how="inner", predicate="intersects")
        except TypeError:
            return gpd.sjoin(left, right, how="inner", op="intersects")

    # ---- Copy inputs ----
    hab = habitat_gdf.copy()
    lgn = lgn_gdf.copy()

    if habitat_year_col not in hab.columns:
        raise ValueError(f"'{habitat_year_col}' not found in habitat_gdf columns: {list(hab.columns)}")

    # ---- Filter habitat to year ----
    hab = hab[_year_mask(hab[habitat_year_col], year)].copy()
    if hab.empty:
        warnings.warn(
            f"No habitat features matched year={year} in column '{habitat_year_col}'. "
            f"Returning LGN unchanged for this year.",
            category=UserWarning,
            stacklevel=2,
        )
        return lgn

    # ---- CRS align ----
    if hab.crs != lgn.crs:
        hab = hab.to_crs(lgn.crs)

    # ---- Early cleanup ----
    hab = _drop_missing_empty(hab)
    lgn = _drop_missing_empty(lgn)

    if hab.empty:
        warnings.warn(
            f"Habitat features for year={year} exist, but geometries are empty/missing after cleaning. "
            f"Returning LGN unchanged for this year.",
            category=UserWarning,
            stacklevel=2,
        )
        return lgn

    if lgn.empty:
        return lgn

    # ---- bbox short-circuit ----
    if not box(*hab.total_bounds).intersects(box(*lgn.total_bounds)):
        return lgn

    # ---- Optional validity repair ----
    if make_valid:
        try:
            from shapely import make_valid as _make_valid  # shapely >= 2
            hab["geometry"] = hab.geometry.map(_make_valid)
            lgn["geometry"] = lgn.geometry.map(_make_valid)
        except Exception:
            hab["geometry"] = hab.buffer(0)
            lgn["geometry"] = lgn.buffer(0)

        hab = _drop_missing_empty(hab)
        lgn = _drop_missing_empty(lgn)

        if hab.empty:
            warnings.warn(
                f"Habitat geometries for year={year} became empty/missing after make_valid/buffer(0). "
                f"Returning LGN unchanged for this year.",
                category=UserWarning,
                stacklevel=2,
            )
            return lgn

        if lgn.empty:
            return lgn

    # ---- Spatial prefilter via sjoin ----
    left = lgn[["geometry"]].copy()
    left["_lgn_i"] = left.index
    right = hab[["geometry"]].copy()
    right["_hab_i"] = right.index

    j = _spatial_join_intersects(left, right)
    lgn_idx = pd.Index(j["_lgn_i"].unique())
    hab_idx = pd.Index(j["_hab_i"].unique())

    if len(lgn_idx) == 0:
        return lgn

    hab_sub = hab.loc[hab_idx]
    lgn_cand = lgn.loc[lgn_idx]

    # ---- Union habitat candidates ----
    try:
        if grid_size is None:
            hab_union = shapely.union_all(hab_sub.geometry.to_numpy())
        else:
            hab_union = shapely.union_all(hab_sub.geometry.to_numpy(), grid_size=grid_size)
    except Exception:
        hab_union = unary_union(hab_sub.geometry)
        if grid_size is not None:
            hab_union = shapely.set_precision(hab_union, grid_size, mode="valid_output")

    if hab_union is None or hab_union.is_empty:
        return lgn

    # ---- Difference only intersecting LGN candidates ----
    if grid_size is not None:
        lgn_cand = lgn_cand.copy()
        lgn_cand["geometry"] = shapely.set_precision(lgn_cand.geometry, grid_size, mode="valid_output")

    idx_list = list(lgn_cand.index)
    for i in range(0, len(idx_list), batch_size):
        batch_idx = idx_list[i : i + batch_size]
        if grid_size is not None:
            lgn.loc[batch_idx, "geometry"] = lgn_cand.loc[batch_idx, "geometry"].difference(hab_union)
        else:
            lgn.loc[batch_idx, "geometry"] = lgn.loc[batch_idx, "geometry"].difference(hab_union)

    # ---- Final cleanup + explode ----
    lgn = _drop_missing_empty(lgn)
    if explode_parts:
        lgn = lgn.explode(index_parts=False)

    return lgn



################################################################################
## RESTRUCTURE THE UPDATED LGN OW GDF BEFORE EXPORT
################################################################################
from collections.abc import Iterable
import geopandas as gpd
from functions.gpkg_funcs import (
    prefix_index,
)

def _years_to_csv_string(years) -> str:
    """
    Convert years input to a comma-separated string without spaces.
    Examples:
      2018 -> "2018"
      "2018, 2019" -> "2018,2019"
      [2018, 2019] -> "2018,2019"
      ("2018", "2019") -> "2018,2019"
    """
    if years is None:
        raise ValueError("`years` must be provided (int/str or an iterable of years).")

    # string: normalize whitespace
    if isinstance(years, str):
        s = years.replace(" ", "")
        return s

    # scalar (int/float/etc.)
    if not isinstance(years, Iterable):
        return str(years)

    # iterable (list/tuple/set/np array/pd series), but not string (handled above)
    cleaned = []
    for y in years:
        if y is None:
            continue
        cleaned.append(str(y).strip())

    # remove empties and normalize potential "2018,2019" items inside iterables
    joined = ",".join([c for c in cleaned if c != ""])
    joined = joined.replace(" ", "")
    return joined

def tidy_lgn_output(
    gdf: gpd.GeoDataFrame,
    years,
    code_prefix: str = "LGN_OW",
    habitatType1_col: str = "habitatType1",
    habitatType1_value: str = "LGN_OW",
    habitatnaam_value: str = "Open water",
    habitatnaam_disp_col: str = "habitatnaam_1_disp",
    area_col: str = "bedekkingsOppervlakte1",
    pct_col: str = "bedekkingsPercentage1",
    index_prefix: str | None = None,
    index_name: str | None = None,
    apply_min_area: bool = False,
    min_area_m2: float = 100.0,
) -> gpd.GeoDataFrame:
    """
    Clean and standardize an LGN output GeoDataFrame.

    Writes:
    - `years` as a string:
        * single year -> "2018"
        * multiple years -> "2018,2019,2020" (comma-separated, no spaces)
      Input can be int, str, or an iterable of years; it will be normalized.
    - `habitatType1_col`, `habitatnaam_disp_col`
    - `pct_col` (default 100.0)
    - `area_col` (m²), computed from geometry (requires projected CRS in metres)

    Optional filtering:
    - If `apply_min_area=True`, rows with `{area_col} < min_area_m2` are removed.

    Index / code:
    - Index is set to `{code_prefix}_{i}` (no year embedded).
    - Optionally apply an extra index prefix via `index_prefix`.

    Final column order:
    ['years', 'habitatType1', 'habitatnaam_1_disp',
     'bedekkingsPercentage1', 'bedekkingsOppervlakte1', 'geometry']
    """
    out = gdf.drop(columns=["value", "region_id"], errors="ignore").copy()

    out["years"] = _years_to_csv_string(years)
    out[habitatType1_col] = habitatType1_value
    out[habitatnaam_disp_col] = habitatnaam_value

    if out.crs is None:
        raise ValueError("GeoDataFrame has no CRS set; cannot safely compute area in m².")
    if out.crs.is_geographic:
        raise ValueError(
            f"GeoDataFrame CRS is geographic ({out.crs}); reproject to a metric CRS (e.g. EPSG:28992) "
            "before computing area in m²."
        )

    geom_col = out.geometry.name
    out[area_col] = out.geometry.area

    if apply_min_area:
        out = out.loc[out[area_col] >= float(min_area_m2)].copy()

    out[pct_col] = 100.0

    out = out.reset_index(drop=True)
    out.index = [f"{code_prefix}_{i}" for i in range(1, len(out) + 1)]
    if index_name is not None:
        out.index.name = None if index_name == "" else index_name

    if index_prefix:
        out = prefix_index(out, prefix=index_prefix, index_name=out.index.name, drop_old_index=True)

    out = out[["years", habitatType1_col, habitatnaam_disp_col, pct_col, area_col, geom_col]].set_geometry(geom_col)
    return out



################################################################################
## ALL YEAR OVERLAPPING GEOMS
################################################################################
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union


def overlap_all_years_geoms(
    gdfs: list[gpd.GeoDataFrame],
    years_col: str = "years",
    out_years_col: str = "years_combined",
    keep_polygonal_only: bool = True,
) -> gpd.GeoDataFrame:
    """
    Compute the geometric overlap shared by ALL GeoDataFrames (e.g., all years),
    returning one row per resulting (multi)polygon part.

    What it does
    ------------
    1) For each input GeoDataFrame, dissolves/merges all its features into a single geometry.
       (This represents "area mapped in that year".)
    2) Intersects these per-GDF dissolved geometries across the entire stack.
       The result is the area present in EVERY input GeoDataFrame.
    3) Explodes the result into separate rows:
       - each disconnected polygon part becomes its own row
       - MultiPolygons are split into individual Polygon rows
    4) Adds a `out_years_col` attribute containing all years used, as a comma-separated string.

    Parameters
    ----------
    gdfs :
        List of GeoDataFrames to intersect. All must share the same CRS.
        Typically one GeoDataFrame per year.
    years_col :
        Column containing the year label in each GeoDataFrame. Each GeoDataFrame is expected
        to have exactly one unique non-null value in this column (e.g., "2018").
    out_years_col :
        Name of the output column that will contain all combined years as a string
        (e.g., "2018,2019,2020").
    keep_polygonal_only :
        If True, drops non-area intersections (LineString/Point) and keeps only polygonal
        area (Polygon/MultiPolygon). This prevents outputs like GEOMETRYCOLLECTION with
        points/lines when geometries only touch at edges/vertices.

    Returns
    -------
    GeoDataFrame
        A GeoDataFrame in the same CRS with:
        - one row per overlapping polygon part present in ALL inputs
        - column `out_years_col` filled with the combined years string
        If there is no shared area overlap, returns an empty GeoDataFrame (0 rows).
    """
    if not gdfs:
        raise ValueError("gdfs is empty")

    crs0 = gdfs[0].crs
    if crs0 is None:
        raise ValueError("First GeoDataFrame has no CRS")
    for i, gdf in enumerate(gdfs):
        if gdf.crs != crs0:
            raise ValueError(f"CRS mismatch at index {i}: {gdf.crs} != {crs0}")

    years: list[str] = []
    per_gdf_union = []

    for i, gdf in enumerate(gdfs):
        if gdf.empty:
            # If any input is empty, there can be no common overlap area
            return gpd.GeoDataFrame({out_years_col: []}, geometry=[], crs=crs0)

        if years_col not in gdf.columns:
            raise ValueError(f"Missing column '{years_col}' in gdf index {i}")

        uniq = list(gdf[years_col].dropna().unique())
        if len(uniq) != 1:
            raise ValueError(
                f"Expected exactly 1 unique '{years_col}' value in gdf index {i}, got {uniq}"
            )
        years.append(str(uniq[0]))

        # dissolve all features of that gdf into one geometry
        per_gdf_union.append(gdf.geometry.union_all())

    # intersection across all gdfs
    inter = per_gdf_union[0]
    for geom in per_gdf_union[1:]:
        inter = inter.intersection(geom)
        if inter.is_empty:
            break

    if inter.is_empty:
        return gpd.GeoDataFrame({out_years_col: []}, geometry=[], crs=crs0)

    # Keep only polygonal part (avoid GEOMETRYCOLLECTION with points/lines)
    if keep_polygonal_only:
        if isinstance(inter, GeometryCollection):
            polys = [g for g in inter.geoms if isinstance(g, (Polygon, MultiPolygon)) and not g.is_empty]
            inter = unary_union(polys) if polys else inter.buffer(0)
        elif not isinstance(inter, (Polygon, MultiPolygon)):
            inter = inter.buffer(0)

        if inter.is_empty:
            return gpd.GeoDataFrame({out_years_col: []}, geometry=[], crs=crs0)

    years_str = ",".join(years)

    out = gpd.GeoDataFrame({out_years_col: [years_str]}, geometry=[inter], crs=crs0)

    # explode so each disconnected polygon part becomes a separate row
    out = out.explode(index_parts=False).reset_index(drop=True)

    # Ensure each row repeats the years string (explode keeps it, but be explicit)
    out[out_years_col] = years_str

    return out



################################################################################
## MERGING ALL YEARS LGN WITH HABITAT KART
################################################################################
from pathlib import Path
import pandas as pd
import geopandas as gpd


def add_lgn_to_habitat_kart_and_save(
    habitat_kart_gdf: gpd.GeoDataFrame,
    lgn_gdf: gpd.GeoDataFrame,
    out_dir: str | Path,
    out_name: str,
    driver: str = "GPKG",
    layer: str | None = None,
    index: bool = True,
) -> gpd.GeoDataFrame:
    """
    - Checks columns are exactly equal (same names + same order), else raises ValueError
    - Checks CRS is the same, else raises ValueError
    - Concatenates rows (habitat_kart + lgn)
    - Saves to out_dir/out_name using the given driver
    - Returns the new GeoDataFrame

    Notes:
      * For GPKG, provide `layer` (defaults to stem of out_name).
      * For Shapefile, use driver="ESRI Shapefile" and out_name ending with .shp.
    """
    # --- 1) Columns must match exactly (names and order)
    if list(habitat_kart_gdf.columns) != list(lgn_gdf.columns):
        raise ValueError(
            "Columns are not exactly the same.\n"
            f"habitat_kart columns: {list(habitat_kart_gdf.columns)}\n"
            f"lgn_gdf columns:      {list(lgn_gdf.columns)}"
        )

    # --- 2) CRS must match
    if habitat_kart_gdf.crs != lgn_gdf.crs:
        raise ValueError(
            f"CRS mismatch.\n"
            f"habitat_kart CRS: {habitat_kart_gdf.crs}\n"
            f"lgn_gdf CRS:      {lgn_gdf.crs}"
        )

    # --- 3) Append rows
    combined = gpd.GeoDataFrame(
        pd.concat([habitat_kart_gdf, lgn_gdf], axis=0),
        crs=habitat_kart_gdf.crs,
    )

    # --- 4) Save
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / out_name

    if driver.upper() == "GPKG":
        layer = layer or out_path.stem
        combined.to_file(out_path, driver=driver, layer=layer, index=index)
    else:
        combined.to_file(out_path, driver=driver, index=index)

    # --- 5) Return
    return combined