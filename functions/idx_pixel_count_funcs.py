################################################################################
## PIXEL COUNT
################################################################################
import pandas as pd
import numpy as np
from typing import Tuple


def pixel_count_df_func(
    gdf: pd.DataFrame,
    idx_df: pd.DataFrame,
    area_col: str = "bedekkingsOppervlakte1",
) -> Tuple[pd.DataFrame, int]:
    """
    Calculate pixel counts per AO * year using index lists from idx_df.

    Pixels are calculated per row as:
        floor(area_col / 100)
    and then summed per AO * year.

    Returns
    -------
    pixel_df : pd.DataFrame
        Same shape as idx_df, containing pixel counts.

    contributing_idx_count : int
        Number of unique indices from idx_df that actually
        contributed area (exist in gdf and were used).
    """

    assert area_col in gdf.columns, (
        f"'{area_col}' not found in gdf columns"
    )

    pixel_df = idx_df.copy(deep=True)

    contributing_idx = set()

    for ao in idx_df.index:
        for year in idx_df.columns:
            idx_list = idx_df.at[ao, year]

            if not idx_list:
                pixel_df.at[ao, year] = 0
                continue

            # Keep only valid indices
            valid_idx = [idx for idx in idx_list if idx in gdf.index]

            if not valid_idx:
                pixel_df.at[ao, year] = 0
                continue

            # Floor BEFORE summing
            pixel_count = int(
                np.floor(gdf.loc[valid_idx, area_col] / 100).sum()
            )

            pixel_df.at[ao, year] = pixel_count

            # Track contributing indices
            contributing_idx.update(valid_idx)

    return pixel_df, len(contributing_idx)



################################################################################
## PIXEL COUNT OVERVIEW DATAFRAME
################################################################################
import pandas as pd
import numpy as np
import warnings

def pixel_vis_func(df, years, title="Pixels per Year", row_order=None):
    """
    Displays a styled, scrollable table of pixel counts per year.
    The current index of df is used as the table index.

    Parameters
    ----------
    row_order : list-like, optional
        List of df.index labels in the desired display order.
        Rows listed here appear first in that order; remaining rows are appended.
        If there is any mismatch between provided and available labels, a warning is raised.
    """

    # --- Optional row ordering (by index labels) ---
    if row_order is not None:
        row_order = list(row_order)

        idx = df.index
        order_idx = pd.Index(row_order)

        missing_in_df = order_idx.difference(idx)      # specified but not in df
        not_specified = idx.difference(order_idx)      # in df but not specified

        if len(missing_in_df) > 0 or len(not_specified) > 0:
            warnings.warn(
                "Row-order mismatch detected. "
                f"Specified but not in df.index: {missing_in_df.tolist()}; "
                f"In df.index but not specified: {not_specified.tolist()}",
                UserWarning
            )

        # Keep only valid labels, preserve order, and append remaining rows
        valid_specified = [lab for lab in row_order if lab in idx]
        final_order = valid_specified + [lab for lab in idx if lab not in valid_specified]
        df = df.loc[final_order]

    # Prepare the DataFrame
    if isinstance(years, tuple) and len(years) == 2:
        year_cols = [str(y) for y in range(years[0], years[1] + 1)]
    else:
        year_cols = [str(y) for y in years]

    year_cols = [col for col in df.columns if str(col) in year_cols]
    plot_df = df[year_cols].copy()
    plot_df = np.floor(plot_df).astype(int)
    plot_df["Total pixels"] = plot_df.sum(axis=1)

    display_df = plot_df.astype(object)
    for j in range(len(year_cols)):
        display_df.iloc[:, j] = display_df.iloc[:, j].apply(
            lambda x: '999+' if x > 999 else (str(x) if x > 0 else "")
        )
    display_df["Total pixels"] = plot_df["Total pixels"].astype(str)

    # Use the current index of df as index for display_df
    display_df.index = df.index
    display_df.index.name = None

    def color_rows(row):
        rownum = row.name if isinstance(row.name, int) else list(display_df.index).index(row.name)
        color = "#ffffff" if rownum % 2 == 0 else "#dfdfdf"
        return ['background-color: {}'.format(color)] * len(row)

    def color_total_area(s):
        styled = []
        for val in s:
            try:
                v = int(val)
                if v <= 75:
                    styled.append("background-color: #d9534f; color: black")
                elif 75 < v <= 100:
                    styled.append("background-color: #f0ad4e; color: black")
                else:
                    styled.append("background-color: #5cb85c; color: black")
            except:
                styled.append("color: black")
        return styled

    styler = display_df.style
    styler = styler.apply(color_rows, axis=1)
    styler = styler.apply(color_total_area, subset=["Total pixels"])
    styler = styler.set_properties(subset=display_df.columns, **{
        "color": "black", "text-align": "center", "font-family": "Arial", "font-size": "13px"
    })
    styler = styler.set_properties(subset=["Total pixels"], **{"font-weight": "bold"})
    styler = styler.set_table_styles([
        {'selector': 'th', 'props': [('font-weight', 'bold'), ('background-color', '#aaaaaa'), ('color', 'black')]},
        {'selector': 'th.row_heading', 'props': [('font-style', 'italic'), ('font-weight', 'bold'), ('background-color', '#aaaaaa'), ('color', 'black')]},
        {'selector': 'caption', 'props': [
            ('caption-side', 'top'),
            ('font-size', '24px'),
            ('font-weight', 'bold'),
            ('color', 'black'),
            ('background-color', '#aaaaaa'),
            ('text-align', 'center'),
            ('padding', '0px')
        ]}
    ])
    styler = styler.set_caption(title)
    styler = styler.set_table_attributes('style="display:inline-block;overflow-x:auto;max-width:100%;"')

    return styler



################################################################################
##  RETURN AREA of UNIQUE Pixels
################################################################################
import pandas as pd
from typing import Optional


def unique_idx_area_func(
    gdf: pd.DataFrame,
    idx_df: pd.DataFrame,
    id_col: Optional[str] = None,
    area_col: str = "Shape_Area",
    decimals: int = 2,
    as_text: bool = True,
    divide_by: float = 100.0,
    unit: str = "pixels",
) -> Optional[str]:
    """
    Compute per-row total area based on unique IDs found in each `idx_df` row.

    For each row in `idx_df` (e.g., an AO/category), this function:
    1) Collects all indices listed in the row across all columns (e.g., years).
    2) Filters to indices that exist in `gdf.index`.
    3) Deduplicates *within the row* so each ID is counted at most once for that row.
       - If `id_col` is None, the `gdf.index` values are treated as the IDs.
       - If `id_col` is provided, uniqueness is determined by `gdf[id_col]`.
    4) Sums `gdf[area_col]` for those unique IDs.

    Output:
    - If `as_text=False` (default): prints one line per row: "<row_index>: <total_area>"
      and returns None.
    - If `as_text=True`: returns a single multiline string with one line per row and
      does not print.

    Parameters
    ----------
    gdf : pd.DataFrame
        Source table containing areas and IDs (either in the index or in `id_col`).
    idx_df : pd.DataFrame
        Table where each cell contains a list of indices into `gdf` (or empty/None).
        Totals are computed per `idx_df` row across all its columns.
    id_col : str | None, default None
        Column in `gdf` that contains the ID used for deduplication within each row.
        If None, `gdf.index` is used as the ID.
    area_col : str, default "bedekkingsOppervlakte1"
        Column in `gdf` containing the area to sum.
    decimals : int, default 2
        Number of decimal places used when formatting totals.
    as_text : bool, default False
        If True, return the multiline text instead of printing.

    Returns
    -------
    str | None
        Multiline string if `as_text=True`, else None.
    """
    assert area_col in gdf.columns, f"'{area_col}' not found in gdf columns"
    if id_col is not None:
        assert id_col in gdf.columns, f"'{id_col}' not found in gdf columns"

    lines = []
    for ao in idx_df.index:
        all_idx = []
        for col in idx_df.columns:
            idx_list = idx_df.at[ao, col]
            if idx_list:
                all_idx.extend(idx_list)

        valid_idx = [i for i in all_idx if i in gdf.index]

        if not valid_idx:
            total_area = 0.0
        else:
            if id_col is None:
                unique_idx = list(dict.fromkeys(valid_idx))
                total_area = float(gdf.loc[unique_idx, area_col].sum())
            else:
                sub = gdf.loc[valid_idx, [id_col, area_col]]
                sub_unique = sub.drop_duplicates(subset=id_col, keep="first")
                total_area = float(sub_unique[area_col].sum())

        value = total_area / divide_by
        lines.append(f"{ao}: {value:.{decimals}f} {unit}")

    text = "\n".join(lines)

    if as_text:
        return text

    print(text)
    return None



################################################################################
##  PIXEL COUNT OVERVIEW+DIFFERENCES DATAFRAME
################################################################################
def pixel_vis_plus_differences_func(df, years, title="Pixels per Year", row_order=None, idx_df=None, original_idx_df=None):
    """
    Displays a styled, scrollable table of pixel counts per year.
    The current index of df is used as the table index.

    Parameters
    ----------
    row_order : list-like, optional
        List of df.index labels in the desired display order.
        Rows listed here appear first in that order; remaining rows are appended.
        If there is any mismatch between provided and available labels, a warning is raised.
    idx_df : pd.DataFrame, optional
        The split index dataframe (training_df or validation_df) used to generate the pixel counts.
    original_idx_df : pd.DataFrame, optional
        The original index dataframe before splitting. Used to determine if data was
        available but not selected. If provided along with idx_df, cells with data in
        original_idx_df but empty in idx_df will be marked with [x].
    """

    # --- Optional row ordering (by index labels) ---
    if row_order is not None:
        row_order = list(row_order)

        idx = df.index
        order_idx = pd.Index(row_order)

        missing_in_df = order_idx.difference(idx)
        not_specified = idx.difference(order_idx)

        if len(missing_in_df) > 0 or len(not_specified) > 0:
            warnings.warn(
                "Row-order mismatch detected. "
                f"Specified but not in df.index: {missing_in_df.tolist()}; "
                f"In df.index but not specified: {not_specified.tolist()}",
                UserWarning
            )

        valid_specified = [lab for lab in row_order if lab in idx]
        final_order = valid_specified + [lab for lab in idx if lab not in valid_specified]
        df = df.loc[final_order]

    # Prepare the DataFrame
    if isinstance(years, tuple) and len(years) == 2:
        year_cols = [str(y) for y in range(years[0], years[1] + 1)]
    else:
        year_cols = [str(y) for y in years]

    year_cols = [col for col in df.columns if str(col) in year_cols]
    plot_df = df[year_cols].copy()
    plot_df = np.floor(plot_df).astype(int)
    plot_df["Total pixels"] = plot_df.sum(axis=1)

    display_df = plot_df.astype(object)
    
    # Format cells: show [x] for data available in original but not in split
    for j, col in enumerate(year_cols):
        for i, row_label in enumerate(display_df.index):
            val = display_df.iloc[i, j]
            
            # Check if we should mark this as [x]
            if (idx_df is not None and original_idx_df is not None and 
                col in idx_df.columns and col in original_idx_df.columns and
                row_label in idx_df.index and row_label in original_idx_df.index):
            
                split_list = idx_df.at[row_label, col]
                original_list = original_idx_df.at[row_label, col]

                # Check if original_list has data but split_list doesn't
                has_original = original_list is not None and len(original_list) > 0
                has_split = split_list is not None and len(split_list) > 0
                
                # Mark [x] if original had data but split doesn't
                if has_original and not has_split:
                    display_df.iloc[i, j] = '[-]'
                    continue
            
            # Otherwise, format normally
            if val > 999:
                display_df.iloc[i, j] = '999+'
            elif val > 0:
                display_df.iloc[i, j] = str(val)
            else:
                display_df.iloc[i, j] = ""
    
    display_df["Total pixels"] = plot_df["Total pixels"].astype(str)

    display_df.index = df.index
    display_df.index.name = None

    def color_rows(row):
        rownum = row.name if isinstance(row.name, int) else list(display_df.index).index(row.name)
        color = "#ffffff" if rownum % 2 == 0 else "#dfdfdf"
        return ['background-color: {}'.format(color)] * len(row)

    def color_total_area(s):
        styled = []
        for val in s:
            try:
                v = int(val)
                if v <= 75:
                    styled.append("background-color: #d9534f; color: black")
                elif 75 < v <= 100:
                    styled.append("background-color: #f0ad4e; color: black")
                else:
                    styled.append("background-color: #5cb85c; color: black")
            except:
                styled.append("color: black")
        return styled

    styler = display_df.style
    styler = styler.apply(color_rows, axis=1)
    styler = styler.apply(color_total_area, subset=["Total pixels"])
    styler = styler.set_properties(subset=display_df.columns, **{
        "color": "black", "text-align": "center", "font-family": "Arial", "font-size": "13px"
    })
    styler = styler.set_properties(subset=["Total pixels"], **{"font-weight": "bold"})
    styler = styler.set_table_styles([
        {'selector': 'th', 'props': [('font-weight', 'bold'), ('background-color', '#aaaaaa'), ('color', 'black')]},
        {'selector': 'th.row_heading', 'props': [('font-style', 'italic'), ('font-weight', 'bold'), ('background-color', '#aaaaaa'), ('color', 'black')]},
        {'selector': 'caption', 'props': [
            ('caption-side', 'top'),
            ('font-size', '24px'),
            ('font-weight', 'bold'),
            ('color', 'black'),
            ('background-color', '#aaaaaa'),
            ('text-align', 'center'),
            ('padding', '0px')
        ]}
    ])
    styler = styler.set_caption(title)
    styler = styler.set_table_attributes('style="display:inline-block;overflow-x:auto;max-width:100%;"')

    return styler