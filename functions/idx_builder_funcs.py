################################################################################
##  TMP 0 IDX
################################################################################
import geopandas as gpd
import pandas as pd
from typing import Tuple


def tmp_0_idx_df_func(
    gdf: gpd.GeoDataFrame,
    group_col: str = "habitatnaam_1_disp",
    year_col: str = "years",
    year_sep: str = ",",
) -> Tuple[pd.DataFrame, int]:
    """
    Build a TMP-0 IDX table containing lists of GeoDataFrame index values
    per habitat (rows) and per year (columns).

    If `year_col` contains multiple years separated by commas (e.g. "2018,2019"),
    the feature index is appended to *each* of those years (after splitting).
    """

    gdf = gdf.copy()

    if gdf.empty:
        empty_df = pd.DataFrame()
        empty_df.index.name = "index"
        return empty_df, 0

    # Make years into a list per row, then explode to one year per row
    years = gdf[year_col].astype(str)

    # Split on commas, strip spaces, drop empties
    gdf["_year_str"] = (
        years
        .str.split(year_sep)
        .apply(lambda xs: [x.strip() for x in xs if x.strip() != ""])
    )

    gdf = gdf.explode("_year_str")
    gdf = gdf[gdf["_year_str"].notna() & (gdf["_year_str"] != "")]

    # Group and store original GeoDataFrame index values
    idx_df = (
        gdf.assign(_idx=gdf.index)
          .groupby([group_col, "_year_str"])["_idx"]
          .agg(list)
          .unstack(fill_value=[])
    )

    # Robust sorting: numeric-looking years first (numeric), then other labels
    def year_sort_key(x):
        s = str(x).strip()
        return (0, int(s)) if s.isdigit() else (1, s)

    idx_df = idx_df.reindex(columns=sorted(idx_df.columns, key=year_sort_key))
    idx_df.index.name = "index"

    stored_idx_count = sum(
        len(idx_list)
        for col in idx_df.columns
        for idx_list in idx_df[col]
    )

    return idx_df, stored_idx_count



################################################################################
##  TMP 1 IDX
################################################################################
import geopandas as gpd
import pandas as pd
from typing import Tuple


def tmp_1_idx_df_func(
    gdf: gpd.GeoDataFrame,
    group_col: str = "habitatnaam_1_disp",
    year_col: str = "years",
    year_sep: str = ",",
) -> Tuple[pd.DataFrame, int]:
    """
    Build a TMP-1 IDX table containing lists of GeoDataFrame index values
    per habitat (rows) and per year (columns), with an extra combined column
    '2010-2016' (years 2010..2016 merged).

    If `year_col` contains multiple years separated by commas (e.g. "2018,2019"),
    the feature index is appended to *each* of those years (after splitting).

    Returns
    -------
    idx_df : pd.DataFrame
        TMP-1 IDX table (with '2010-2016' and all year columns > 2016)
    stored_idx_count : int
        Total number of stored GeoDataFrame indices in idx_df
    """
    gdf = gdf.copy()

    if gdf.empty:
        empty_df = pd.DataFrame()
        empty_df.index.name = "index"
        return empty_df, 0

    # --- Added (same approach as TMP-0): split multi-year strings to one-year-per-row
    years = gdf[year_col].astype(str)

    gdf["_year_str"] = (
        years.str.split(year_sep)
             .apply(lambda xs: [x.strip() for x in xs if x.strip() != ""])
    )

    gdf = gdf.explode("_year_str")
    gdf = gdf[gdf["_year_str"].notna() & (gdf["_year_str"] != "")]
    # ---

    idx_df = (
        gdf.assign(_idx=gdf.index)
           .groupby([group_col, "_year_str"])["_idx"]
           .agg(list)
           .unstack(fill_value=[])
    )

    # --- Updated sorting: keep existing behaviour for numeric years,
    #     but avoid crashing on non-numeric labels
    def year_sort_key(x):
        s = str(x).strip()
        return (0, int(s)) if s.isdigit() else (1, s)

    idx_df = idx_df.reindex(columns=sorted(idx_df.columns, key=year_sort_key))
    # ---

    # Regrouping years 2010–2016 (kept, but made robust to non-numeric columns)
    group_years = [str(y) for y in range(2010, 2017) if str(y) in idx_df.columns]

    # Only treat digit columns as years for the "> 2016" selection
    after_years = [c for c in idx_df.columns if str(c).strip().isdigit() and int(str(c).strip()) > 2016]

    idx_df["2010-2016"] = (
        idx_df[group_years].apply(lambda row: sum(row, []), axis=1)
        if group_years else [[] for _ in range(len(idx_df))]
    )

    # Keep only combined column + years after 2016 (same as before)
    idx_df = idx_df[["2010-2016"] + after_years]
    idx_df.index.name = "index"

    stored_idx_count = sum(
        len(idx_list)
        for col in idx_df.columns
        for idx_list in idx_df[col]
    )

    return idx_df, int(stored_idx_count)



################################################################################
##  TMP 2 IDX
################################################################################
import geopandas as gpd
import pandas as pd
from typing import Tuple


def tmp2_idx_df_func(
    gdf: gpd.GeoDataFrame,
    group_col: str = "habitatnaam_1_disp",
    year_col: str = "years",
    year_sep: str = ",",
) -> Tuple[pd.DataFrame, int]:
    """
    Build a TMP-2 IDX table containing lists of GeoDataFrame index values
    per habitat (rows) and per year (columns), restricted to years 2017–2024.

    If `year_col` contains multiple years separated by commas (e.g. "2018,2019"),
    the feature index is appended to *each* of those years (after splitting).

    Returns
    -------
    idx_df : pd.DataFrame
        TMP-2 IDX table (columns: '2017'...'2024' if present; missing years filled with [])
    stored_idx_count : int
        Total number of stored GeoDataFrame indices in idx_df
    """
    gdf = gdf.copy()
    if gdf.empty:
        empty_df = pd.DataFrame()
        empty_df.index.name = "index"
        return empty_df, 0

    years_keep = list(range(2017, 2025))
    year_cols = [str(y) for y in years_keep]

    # --- Added: split multi-year strings to one-year-per-row (like TMP-0/TMP-1)
    years = gdf[year_col].astype(str)
    gdf["_year_str"] = (
        years.str.split(year_sep)
             .apply(lambda xs: [x.strip() for x in xs if x.strip() != ""])
    )
    gdf = gdf.explode("_year_str")
    gdf = gdf[gdf["_year_str"].notna() & (gdf["_year_str"] != "")]
    # ---

    # Normalize years to integers where possible (keeps TMP-2 selection logic intact)
    gdf["_year_int"] = pd.to_numeric(gdf["_year_str"], errors="coerce").astype("Int64")

    # Filter
    gdf = gdf[gdf["_year_int"].isin(years_keep)]
    if gdf.empty:
        empty_df = pd.DataFrame(index=pd.Index([], name="index"), columns=year_cols)
        return empty_df.astype(object), 0

    idx_df = (
        gdf.assign(_idx=gdf.index, _year_str=gdf["_year_int"].astype(str))
           .groupby([group_col, "_year_str"])["_idx"]
           .agg(list)
           .unstack(fill_value=[])
           .reindex(columns=year_cols, fill_value=[])
    )
    idx_df.index.name = "index"

    stored_idx_count = sum(len(lst) for col in idx_df.columns for lst in idx_df[col])
    return idx_df, int(stored_idx_count)



################################################################################
##  HT IDX
################################################################################
import pandas as pd

def HT_idx_df_func(
    gdf: pd.DataFrame,
    idx_df: pd.DataFrame,
    habitat_types_df: pd.DataFrame,
    habitat_col: str = 'habitatType1'
):
    """
    Filter an IDX DataFrame based on habitat types stored in the GDF.

    Returns
    -------
    filtered_idx_df : pd.DataFrame
        Like idx_df, but with indices removed if their habitat type is not allowed.
        Rows (AO) for which *all* AO×year cells become empty are dropped.

    total_saved : int
        Total number of saved indices across all AO × year cells (after filtering).
    """

    allowed_habitats = set(habitat_types_df.index)

    assert habitat_col in gdf.columns, (
        f"'{habitat_col}' not found in gdf columns"
    )

    filtered_idx_df = idx_df.copy(deep=True)
    saved_count_df = pd.DataFrame(
        0,
        index=idx_df.index,
        columns=idx_df.columns
    )

    for ao in idx_df.index:
        for year in idx_df.columns:
            idx_list = idx_df.at[ao, year]

            if not idx_list:
                filtered_idx_df.at[ao, year] = []
                saved_count_df.at[ao, year] = 0
                continue

            filtered_idx = [
                idx
                for idx in idx_list
                if (
                    idx in gdf.index and
                    gdf.at[idx, habitat_col] in allowed_habitats
                )
            ]

            filtered_idx_df.at[ao, year] = filtered_idx
            saved_count_df.at[ao, year] = len(filtered_idx)

    total_saved = int(saved_count_df.values.sum())

    def _cell_has_items(x) -> bool:
        if isinstance(x, (list, tuple, set, dict)):
            return len(x) > 0
        return bool(x) and not pd.isna(x)

    keep_mask = filtered_idx_df.apply(lambda col: col.map(_cell_has_items)).any(axis=1)
    filtered_idx_df = filtered_idx_df.loc[keep_mask].copy()

    return filtered_idx_df, total_saved



################################################################################
##  FC IDX
################################################################################
from typing import Tuple
import pandas as pd
from pandas.api.types import is_numeric_dtype

def FC_idx_df_func(
    gdf: pd.DataFrame,
    idx_df: pd.DataFrame,
    percentage_col: str = "bedekkingsPercentage1",
    min_percentage: float = 100
) -> Tuple[pd.DataFrame, int]:
    """
    Filter an IDX DataFrame by keeping only indices whose cover percentage
    (stored in `gdf[percentage_col]`) is >= `min_percentage`.

    Parameters
    ----------
    gdf : pd.DataFrame
        Source table containing the percentage column. Indices in `idx_df`'s
        lists are expected to correspond to `gdf.index`.
    idx_df : pd.DataFrame
        Table (same shape preserved) where each cell contains a list of indices
        referring to rows in `gdf`.
    percentage_col : str, default "bedekkingsPercentage1"
        Column in `gdf` containing cover percentages.
    min_percentage : float, default 100
        Minimum allowed percentage threshold. Indices are kept when
        `gdf.at[idx, percentage_col] >= min_percentage`.

    Returns
    -------
    filtered_idx_df : pd.DataFrame
        Same shape as `idx_df`, but with each cell's list filtered to only
        include indices meeting the threshold.
    total_count : int
        Total number of retained indices across the entire filtered IDX table.

    Raises
    ------
    AssertionError
        If `percentage_col` is not present in `gdf`.
    TypeError
        If `gdf[percentage_col]` is not a numeric dtype.
    """

    assert percentage_col in gdf.columns, f"'{percentage_col}' not found in gdf columns"

    if not is_numeric_dtype(gdf[percentage_col]):
        raise TypeError(
            f"Column '{percentage_col}' must be numeric to compare against "
            f"min_percentage; got dtype {gdf[percentage_col].dtype}."
        )

    filtered_idx_df = idx_df.copy(deep=True)
    total_count = 0

    for ao in idx_df.index:
        for year in idx_df.columns:
            idx_list = idx_df.at[ao, year]

            if not idx_list:
                filtered_idx_df.at[ao, year] = []
                continue

            filtered_idx = [
                idx
                for idx in idx_list
                if (
                    idx in gdf.index
                    and pd.notna(gdf.at[idx, percentage_col])
                    and gdf.at[idx, percentage_col] >= min_percentage
                )
            ]

            filtered_idx_df.at[ao, year] = filtered_idx
            total_count += len(filtered_idx)

    return filtered_idx_df, total_count



################################################################################
## GROUP IDX
################################################################################
import pandas as pd
from typing import Tuple, Set, Any


def group_idx_df_func(
    gdf: pd.DataFrame,
    idx_df: pd.DataFrame,
    habitat_types_df: pd.DataFrame,
    habitat_col: str = "habitatType1",
    group_col: str = None,
) -> Tuple[pd.DataFrame, int]:
    """
    Aggregate feature indices per group (or any other grouping column) for each
    time-slice/column in an index-list DataFrame.

    This function expects:
    - `idx_df` to contain, per column (e.g. year), a sequence of lists of feature IDs
      (strings such as "GI_123", "WI_5", "WI_hr_9", ...).
    - `gdf` to be indexed by those feature IDs and to provide a habitat type code in
      `habitat_col` (e.g. "H4030").
    - `habitat_types_df` to be indexed by habitat type codes and to provide a grouping
      label in `group_col` (e.g. "Dry Nature", "Wet Nature", ...).

    For every column in `idx_df`, all feature IDs in all lists are mapped:
        feature_id -> habitat_type (from `gdf[habitat_col]`)
        habitat_type -> group / grouping label (from `habitat_types_df[group_col]`)

    The output DataFrame has one row per group (unique non-null values of `group_col`)
    and the same columns as `idx_df`. Each cell contains a sorted list of unique
    feature IDs belonging to that group in that column.

    Parameters
    ----------
    gdf : pd.DataFrame
        DataFrame (often a GeoDataFrame) indexed by feature IDs. Must contain the
        column specified by `habitat_col`.
    idx_df : pd.DataFrame
        DataFrame where each column represents a time slice (e.g. years). Each
        cell is expected to be a list (or list-like) of feature IDs, or an empty
        list. Example cell value: ["GI_1", "GI_20"].
    habitat_types_df : pd.DataFrame
        Lookup table indexed by habitat type code. Must contain the column specified
        by `group_col` that defines the target grouping.
    habitat_col : str, default "habitatType1"
        Column name in `gdf` holding the habitat type code used to map into
        `habitat_types_df` (its index).
    group_col : str, default "WD_division"
        Column name in `habitat_types_df` holding the grouping label to aggregate by.

    Returns
    -------
    group_idx_df : pd.DataFrame
        DataFrame with index = unique non-null group labels from `habitat_types_df[group_col]`
        (as encountered via the mapping from `gdf`), columns equal to `idx_df.columns`,
        and each cell a sorted list of unique feature IDs that map to that group in
        that time slice.
    total_used : int
        Total number of unique feature IDs that were successfully mapped to a non-null
        group label and included in the result (counted across all columns).

    Raises
    ------
    KeyError
        If `habitat_col` is not present in `gdf.columns`, or `group_col` is not present
        in `habitat_types_df.columns`.
    TypeError
        If `idx_df` contains non-iterable, non-empty cells where a list of feature IDs
        is expected (e.g. an integer or float).
    """

    if habitat_col not in gdf.columns:
        raise KeyError(f"'{habitat_col}' not found in gdf.columns.")
    if group_col not in habitat_types_df.columns:
        raise KeyError(f"'{group_col}' not found in habitat_types_df.columns.")

    index_to_group = gdf[habitat_col].map(habitat_types_df[group_col])

    group_values = index_to_group.dropna().unique()
    group_idx_df = pd.DataFrame(index=group_values, columns=idx_df.columns, dtype=object)
    group_idx_df[:] = [[[] for _ in idx_df.columns] for _ in range(len(group_idx_df.index))]

    used_idx: Set[Any] = set()

    for col in idx_df.columns:
        for idx_list in idx_df[col]:
            if not idx_list:
                continue

            # If idx_list is not list-like, this loop will raise TypeError naturally.
            for gi in idx_list:
                if gi not in index_to_group.index:
                    continue

                group = index_to_group.loc[gi]
                if pd.isna(group):
                    continue

                group_idx_df.at[group, col].append(gi)
                used_idx.add(gi)

    for group in group_idx_df.index:
        for col in group_idx_df.columns:
            group_idx_df.at[group, col] = sorted(set(group_idx_df.at[group, col]))

    return group_idx_df, len(used_idx)