################################################################################
## IDX df to GDF/GPKG FUNCTION
################################################################################
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union
import pandas as pd
import geopandas as gpd

def idx_df_to_gdf_func(
    idx_df: pd.DataFrame,
    source_gdf: gpd.GeoDataFrame,
    out_gpkg: Optional[Union[str, Path]] = None,
    *,
    idx_cols_to_include: Optional[Sequence[str]] = None,
    gdf_cols_to_keep: Optional[Sequence[str]] = None,
    years_colname: str = "years",
    type_colname: str = "type",
    single_layer_name: str = "combined",
) -> gpd.GeoDataFrame:
    """
    Build a GeoDataFrame by selecting features from `source_gdf` based on an index-mapping
    DataFrame (`idx_df`) and (optionally) write the result to a GeoPackage.

    Concept
    -------
    - `idx_df` rows represent "types" / classes (e.g., land-cover classes).
    - `idx_df` columns represent groups to process (typically years, seasons, scenarios, etc.).
    - Each cell in `idx_df` contains a list-like of ID codes that correspond to
      `source_gdf.index`.

    Year/column handling (IMPORTANT)
    --------------------------------
    This function does NOT copy a year column from `source_gdf`.

    Instead, the output column `years_colname` is created from the *idx_df column names*
    in which each ID appears:

    - If an ID appears in exactly one processed `idx_df` column, `years_colname` will
      equal that column name (as a string).
    - If an ID appears in multiple processed `idx_df` columns (for the same `type` row),
      `years_colname` becomes a comma-separated string of all matching column names,
      sorted lexicographically (e.g., "2017,2018,2020").

    Each ID is emitted at most once per `type` (duplicates across multiple years are
    collapsed into a single feature with aggregated `years_colname`).

    Column selection (`idx_cols_to_include`)
    ----------------------------------------
    - If `idx_cols_to_include` is None (default): all columns in `idx_df` are processed.
    - If provided: only those columns are processed; all other `idx_df` columns are ignored.

    Output columns
    --------------
    - If `gdf_cols_to_keep` is None (default): keep only geometry from `source_gdf`.
    - If `gdf_cols_to_keep` is provided: keep exactly those columns from `source_gdf`
      (use "geometry" as a placeholder for the active geometry column).
    - The function always adds:
        - `type_colname` (the `idx_df` row label)
        - `years_colname` (derived from matching `idx_df` column names)
    - The index is NOT reset; `source_gdf.index` values are preserved in the output.

    Missing IDs
    -----------
    ID codes present in `idx_df` but absent from `source_gdf.index` are ignored and a warning
    is printed (unique missing IDs).

    Optional writing
    ----------------
    If `out_gpkg` is provided, the resulting GeoDataFrame is written as a single layer
    named `single_layer_name` to a GeoPackage.

    Parameters
    ----------
    idx_df : pandas.DataFrame
        Mapping table where rows are types/classes and columns are years/groups.
        Each cell must be list-like of IDs that exist in `source_gdf.index`.
    source_gdf : geopandas.GeoDataFrame
        Source features indexed by ID.
    out_gpkg : str | pathlib.Path | None
        Optional output GeoPackage path.
    idx_cols_to_include : sequence[str] | None
        Which `idx_df` columns to process. None = all columns.
    gdf_cols_to_keep : sequence[str] | None
        Source columns to keep in the output. None = geometry only.
        Use "geometry" to refer to the active geometry column.
    years_colname : str
        Name of the output column containing aggregated year/group strings.
    type_colname : str
        Name of the output column containing the type/class (idx_df row label).
    single_layer_name : str
        Layer name used when writing to GeoPackage.

    Returns
    -------
    geopandas.GeoDataFrame
        Combined GeoDataFrame with selected features, including:
        - `type_colname`
        - `years_colname` (comma-separated groups/years derived from idx_df column names)
        - requested `gdf_cols_to_keep` (or geometry only by default)
    """
    # which idx_df columns to process
    if idx_cols_to_include is None:
        selected_cols = list(idx_df.columns)
    else:
        selected_cols = list(idx_cols_to_include)
        missing = [c for c in selected_cols if c not in idx_df.columns]
        if missing:
            raise KeyError(f"idx_cols_to_include contains columns not in idx_df: {missing}")

    geom_name = source_gdf.geometry.name

    # columns to take from source_gdf (note: years will be created, not copied)
    if gdf_cols_to_keep is None:
        source_cols_to_take = [geom_name]
    else:
        cols = [(geom_name if c == "geometry" else c) for c in gdf_cols_to_keep]
        cols_missing = [c for c in cols if c != geom_name and c not in source_gdf.columns]
        if cols_missing:
            raise KeyError(f"Requested gdf_cols_to_keep columns not found in source_gdf: {cols_missing}")
        # ensure geometry last
        source_cols_to_take = [c for c in cols if c != geom_name] + [geom_name]

    pieces: List[gpd.GeoDataFrame] = []
    missing_in_source: List[Any] = []

    # --- per type: build id -> years-set, then emit once
    for typ in idx_df.index:
        id_to_years = {}  # id -> set of year strings

        for col in selected_cols:
            cell = idx_df.at[typ, col]
            if cell is None or (isinstance(cell, float) and pd.isna(cell)):
                continue
            if not isinstance(cell, (list, tuple, set, pd.Index)):
                raise TypeError(f"Expected list-like in idx_df[{col!r}][{typ!r}], got {type(cell)}")

            for code in cell:
                if code in source_gdf.index:
                    id_to_years.setdefault(code, set()).add(str(col))
                else:
                    missing_in_source.append(code)

        if not id_to_years:
            continue

        codes = list(id_to_years.keys())
        sel = source_gdf.loc[codes, source_cols_to_take].copy()
        sel[type_colname] = typ
        sel[years_colname] = [",".join(sorted(id_to_years[c])) for c in sel.index]

        # keep geometry last
        cols = [c for c in sel.columns if c != geom_name] + [geom_name]
        sel = sel[cols].set_geometry(geom_name)
        pieces.append(sel)

    if pieces:
        out_gdf = gpd.GeoDataFrame(pd.concat(pieces, axis=0), crs=source_gdf.crs).set_geometry(geom_name)
        out_gdf = out_gdf[[c for c in out_gdf.columns if c != geom_name] + [geom_name]]
    else:
        out_cols = [c for c in source_cols_to_take if c != geom_name] + [type_colname, years_colname, geom_name]
        out_gdf = gpd.GeoDataFrame(columns=out_cols, crs=source_gdf.crs).set_geometry(geom_name)

    if out_gpkg is not None:
        out_gpkg = Path(out_gpkg)
        out_gpkg.parent.mkdir(parents=True, exist_ok=True)
        out_gdf.to_file(out_gpkg, layer=single_layer_name, driver="GPKG")

    if missing_in_source:
        # de-dup for nicer reporting
        missing_unique = list(dict.fromkeys(missing_in_source))
        print(
            f"Warning: {len(missing_unique)} unique codes not found in source_gdf.index "
            f"(showing first 10): {missing_unique[:10]}"
        )

    return out_gdf



################################################################################
## IDX PLOT FUNCTION
################################################################################
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def _cell_to_codes(cell):
    """Normalize a df-cell to a list of index codes (e.g., GI_...)."""
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    if isinstance(cell, (list, tuple, set, pd.Series)):
        return [str(x) for x in cell if pd.notna(x)]
    if isinstance(cell, str):
        s = cell.strip()
        if not s:
            return []
        # handle strings like "[GI_1, GI_2]"
        if s.startswith("[") and s.endswith("]"):
            inner = s[1:-1].strip()
            if not inner:
                return []
            return [tok.strip().strip("'").strip('"') for tok in inner.split(",") if tok.strip()]
        return [s]
    return [str(cell)]


def idx_to_gdf_plot_func(
    gdf: gpd.GeoDataFrame,
    idx_df: pd.DataFrame,
    year_cols,
    lut_df: pd.DataFrame,
    *,
    typology_col: str = "habitatnaam_1_disp",
    lut_set_col: str,     
    lut_color_col: str,   
    ax=None,
    figsize=(10, 10),
    title=None,
    edgecolor="black",
    linewidth=0.2,
    alpha=0.9,
    missing_codes="warn",      
    missing_colors="raise",    
    overlap="warn",            
    plot_unselected=False,
    unselected_color="#eeeeee",
):
    """
    Plot polygons from ``gdf`` colored by *resampled set* membership for one or more years.

    The data model is assumed to be:
    - ``gdf``: contains *all* polygons, indexed by a unique polygon code (ID).
      A typology attribute is stored in ``typology_col`` (default: ``"habitatnaam_1_disp"``).
    - ``idx_df``: maps resampled sets to polygon codes per year:
      ``idx_df.index`` = resampled set names, ``idx_df[year]`` = list-like (or string) of polygon codes.
    - ``lut_df``: lookup table mapping resampled set names to colors.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing all polygons. Must be indexed by unique polygon codes
        (no duplicate index values). Must contain ``typology_col``.
    idx_df : pandas.DataFrame
        DataFrame that maps resampled sets (index) to polygon codes per year (columns).
        Each cell for a given (resampled_set, year) must be either:
        - a list/tuple/set/Series of polygon codes, or
        - a single polygon code, or
        - a string representation (optionally like ``"[code1, code2]"``).
    year_cols : str | list[str]
        Year column name(s) in ``idx_df`` to use. When multiple years are provided,
        polygon codes are merged (union) across those years per resampled set.
    lut_df : pandas.DataFrame
        Lookup table that maps resampled set names to colors.
    typology_col : str, default "habitatnaam_1_disp"
        Name of the column in ``gdf`` containing the polygon typology/class.
        (The function stores it in the returned ``plot_gdf`` as ``"_typology"``.)
    lut_set_col : str
        Column name in ``lut_df`` containing resampled set names that correspond to ``idx_df.index``.
    lut_color_col : str
        Column name in ``lut_df`` containing colors for each resampled set
        (e.g., hex strings like ``"#RRGGBB"``).
    ax : matplotlib.axes.Axes | None, optional
        Axes to plot on. If None, a new figure and axes are created.
    figsize : tuple, default (10, 10)
        Figure size used when ``ax is None``.
    title : str | None, optional
        Plot title. If None, a title is derived from ``year_cols``.
    edgecolor : str, default "black"
        Edge color for plotted polygons.
    linewidth : float, default 0.2
        Line width for polygon edges.
    alpha : float, default 0.9
        Polygon fill opacity for selected polygons.
    missing_codes : {"ignore","warn","raise"}, default "warn"
        How to handle polygon codes referenced in ``idx_df`` that are not present in ``gdf.index``.
    missing_colors : {"ignore","warn","raise"}, default "raise"
        How to handle resampled sets that do not have a corresponding color in ``lut_df``.
        If not "raise", missing colors are replaced with gray (``"#999999"``) for plotting.
    overlap : {"ignore","warn","raise"}, default "warn"
        How to handle polygon codes that appear in multiple resampled sets (across selected years).
        The first encountered assignment is kept.
    plot_unselected : bool, default False
        If True, plots all polygons in ``gdf`` first in ``unselected_color``, and then overlays
        the selected polygons colored by resampled set.
    unselected_color : str, default "#eeeeee"
        Fill color used for unselected polygons when ``plot_unselected=True``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object.
    ax : matplotlib.axes.Axes
        The matplotlib Axes object used for plotting.
    plot_gdf : geopandas.GeoDataFrame
        Subset of ``gdf`` containing only polygons that were selected via ``idx_df`` for the
        given year(s), with additional columns:
        - ``"_resampled_set"``: assigned resampled set
        - ``"_color"``: color used for plotting
        - ``"_typology"``: copied from ``gdf[typology_col]``

    Raises
    ------
    ValueError
        If ``gdf`` has duplicate index values (polygon codes are not unique).
        If no polygon codes remain to plot after parsing/filtering.
        If ``overlap="raise"`` and at least one polygon code appears in multiple resampled sets.
    KeyError
        If ``typology_col`` is not present in ``gdf``.
        If any requested year column in ``year_cols`` is not present in ``idx_df``.
        If ``lut_set_col`` or ``lut_color_col`` is not present in ``lut_df``.
        If ``missing_colors="raise"`` and at least one resampled set used for plotting has no color.
        If ``missing_codes="raise"`` and at least one referenced polygon code is missing in ``gdf.index``.
    """

    if gdf.index.has_duplicates:
        raise ValueError("gdf index must be unique (polygon codes).")

    if typology_col not in gdf.columns:
        raise KeyError(f"typology_col='{typology_col}' not found in gdf.columns")

    if lut_set_col not in lut_df.columns or lut_color_col not in lut_df.columns:
        raise KeyError(f"lut_df must contain columns '{lut_set_col}' and '{lut_color_col}'")

    year_cols = [year_cols] if isinstance(year_cols, str) else list(year_cols)
    for y in year_cols:
        if y not in idx_df.columns:
            raise KeyError(f"Year column '{y}' not found in idx_df.columns")

    categories = list(idx_df.index)

    # Build category -> color mapping from lut
    lut_map = (lut_df[[lut_set_col, lut_color_col]]
               .dropna()
               .drop_duplicates(subset=[lut_set_col])
               .set_index(lut_set_col)[lut_color_col]
               .to_dict())

    missing_cats = [c for c in categories if c not in lut_map]
    if missing_cats:
        msg = f"{len(missing_cats)} resampled sets (idx_df.index) have no color in lut_df. Examples: {missing_cats[:5]}"
        if missing_colors == "raise":
            raise KeyError(msg)
        if missing_colors == "warn":
            print("Warning:", msg)

    # Assign each polygon code to a resampled set (union across selected years)
    code_to_cat = {}
    overlaps = []
    missing_codes_total = []

    for cat in categories:
        codes = []
        for y in year_cols:
            codes.extend(_cell_to_codes(idx_df.loc[cat, y]))

        # unique preserve order
        seen = set()
        codes = [c for c in codes if not (c in seen or seen.add(c))]

        # filter codes not present in gdf
        missing_codes = [c for c in codes if c not in gdf.index]
        if missing_codes:
            missing_codes_total.extend(missing_codes)
            if missing_codes == "raise":
                raise KeyError(f"Codes not found in gdf index. Examples: {missing_codes[:5]}")
            if missing_codes == "warn":
                print("Warning:", f"{len(missing_codes)} codes from '{cat}' not found in gdf index. Examples: {missing_codes[:5]}")
            codes = [c for c in codes if c in gdf.index]

        for code in codes:
            if code in code_to_cat and code_to_cat[code] != cat:
                overlaps.append((code, code_to_cat[code], cat))
                continue  # keep first
            code_to_cat[code] = cat

    if overlaps:
        msg = f"{len(overlaps)} codes appear in multiple resampled sets (kept first). Examples: {overlaps[:5]}"
        if overlap == "raise":
            raise ValueError(msg)
        if overlap == "warn":
            print("Warning:", msg)

    if not code_to_cat:
        raise ValueError("No codes to plot (after parsing/filtering).")

    # Build plot gdf
    plot_gdf = gdf.loc[list(code_to_cat.keys())].copy()
    plot_gdf["_resampled_set"] = [code_to_cat[c] for c in plot_gdf.index]
    plot_gdf["_color"] = plot_gdf["_resampled_set"].map(lut_map)
    plot_gdf["_typology"] = plot_gdf[typology_col]

    # If missing colors are allowed, set fallback
    if plot_gdf["_color"].isna().any():
        if missing_colors == "raise":
            missing = plot_gdf.loc[plot_gdf["_color"].isna(), "_resampled_set"].unique().tolist()
            raise KeyError(f"Some plotted sets have no color in lut_df: {missing[:10]}")
        if missing_colors == "warn":
            print("Warning:", "Some plotted sets have no color in lut_df; using gray for those.")
        plot_gdf["_color"] = plot_gdf["_color"].fillna("#999999")

    # Plot
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    if plot_unselected:
        gdf.plot(ax=ax, color=unselected_color, edgecolor="none", alpha=1.0)

    plot_gdf.plot(
        ax=ax,
        color=plot_gdf["_color"],
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
    )

    if title is None:
        title = " + ".join(map(str, year_cols))
    ax.set_title(title)
    ax.set_axis_off()

    # Legend in idx_df index order (only those that have a color or appear in lut_map)
    handles = []
    for cat in categories:
        if cat in lut_map:
            handles.append(Patch(facecolor=lut_map[cat], edgecolor="none", label=str(cat)))
        elif missing_colors != "raise":
            handles.append(Patch(facecolor="#999999", edgecolor="none", label=str(cat)))
    ax.legend(handles=handles, loc="lower left", frameon=True)

    return fig, ax, plot_gdf