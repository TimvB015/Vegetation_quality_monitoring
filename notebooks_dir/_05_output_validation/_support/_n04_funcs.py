from __future__ import annotations

import matplotlib
import PIL
PIL.Image.MAX_IMAGE_PIXELS = None  # Remove PIL size limit
matplotlib.rcParams['agg.path.chunksize'] = 100000

# Also try setting these:
import matplotlib.pyplot as plt
plt.rcParams['path.simplify'] = True
plt.rcParams['path.simplify_threshold'] = 0.111111

################################################################################
## GENERAL HELPER FUNCTIONS
################################################################################
# COLOR HANDLING
from functions.color_coding_handling import(
    hex_to_rgb,
    rgb_to_hsv,
    hsv_to_rgb,
)

# NORMALIZE YEAR INPUT
def normalize_year(y, normalize=True):
    """
    Convert year to int if it's a numeric string.
    
    Parameters
    ----------
    y : str | int | float
        Year value to normalize
    normalize : bool, default True
        If True, convert numeric strings to int. If False, return as string.
    
    Returns
    -------
    int | str
        Integer year if normalize=True and y is numeric, otherwise string.
    
    Examples
    --------
    >>> normalize_year("2017")
    2017
    >>> normalize_year(" 2017 ")
    2017
    >>> normalize_year("2017", normalize=False)
    '2017'
    >>> normalize_year("2017a")
    '2017a'
    """
    y_str = str(y).strip()
    return int(y_str) if (normalize and y_str.isdigit()) else y_str


# READ INPUT CRS
def epsg_to_crs(epsg_val):
    """
    Normalize EPSG/CRS values to standard 'EPSG:XXXXX' format.
    
    Parameters
    ----------
    epsg_val : str | int | None
        EPSG code or CRS identifier in various formats
    
    Returns
    -------
    str | None
        Standardized 'EPSG:XXXXX' string, or None if input is None
    
    Examples
    --------
    >>> epsg_to_crs("UTM32631")
    'EPSG:32631'
    >>> epsg_to_crs("32631")
    'EPSG:32631'
    >>> epsg_to_crs(32631)
    'EPSG:32631'
    >>> epsg_to_crs("EPSG:32631")
    'EPSG:32631'
    >>> epsg_to_crs("epsg:4326")
    'EPSG:4326'
    >>> epsg_to_crs(None)
    None
    """
    if epsg_val is None:
        return None
    s = str(epsg_val).strip()
    if s.upper().startswith("EPSG:"):
        return s.upper()
    if s.upper().startswith("UTM"):
        s = s[3:]
    return f"EPSG:{s}"



################################################################################
## EXTRACT CLASSIFICATION RESULTS PER PERIOD
################################################################################
def filter_by_timeframe(df, period):
    """
    Filter dataframe to return only rows matching the requested timeframe period.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing a 'timeframe' column
    period : str
        The timeframe period to filter for (e.g., 'Q1', 'Q2', 'Q3', 'Q4')
    
    Returns:
    --------
    pandas.DataFrame
        Filtered dataframe with only rows matching the requested period
    """
    return df[df['timeframe'] == period]



################################################################################
## FILTER DICTIONARIES BY KEYS
################################################################################
def filter_dict_by_keys(
    source_dict: dict,
    keys: list[str],
    dict_name: str = "dictionary",
    allow_missing: bool = False,
) -> dict:
    """
    Filter a dictionary to only include specified keys.
    
    Parameters
    ----------
    source_dict : dict
        The dictionary to filter
    keys : list[str]
        Keys to include in the filtered dictionary
    dict_name : str
        Name of the dictionary for error messages (e.g., "background_config")
    allow_missing : bool
        If True, silently skip keys not in source_dict.
        If False, raise ValueError for missing keys.
    
    Returns
    -------
    dict
        Filtered dictionary containing only the specified keys
    
    Raises
    ------
    ValueError
        If allow_missing=False and any requested keys are not in source_dict
    
    Examples
    --------
    >>> config = {"A": 1, "B": 2, "C": 3}
    >>> filter_dict_by_keys(config, ["A", "C"], "config")
    {'A': 1, 'C': 3}
    
    >>> filter_dict_by_keys(config, ["A", "D"], "config")
    ValueError: Invalid keys in config: ['D']. Available keys: ['A', 'B', 'C']
    
    >>> filter_dict_by_keys(config, ["A", "D"], "config", allow_missing=True)
    {'A': 1}
    """
    if not allow_missing:
        invalid_keys = [key for key in keys if key not in source_dict]
        if invalid_keys:
            raise ValueError(
                f"Invalid keys in {dict_name}: {invalid_keys}. "
                f"Available keys: {list(source_dict.keys())}"
            )
    
    return {
        key: source_dict[key] 
        for key in keys 
        if key in source_dict
    }



################################################################################
## REMOVE TYPOLOGIES FROM COLOR DF
################################################################################
def remove_typologies_from_color_df(
    color_df: pd.DataFrame,
    typologies_to_remove: list[str],
) -> pd.DataFrame:
    """
    Remove specific typologies from a color DataFrame.
    
    Parameters
    ----------
    color_df : pd.DataFrame
        DataFrame with columns 'type' and 'color'
    typologies_to_remove : list[str]
        List of typology names to remove, e.g., ["Open water", "Remaining"]
    
    Returns
    -------
    pd.DataFrame
        Color DataFrame with specified typologies removed
    
    Examples
    --------
    >>> filtered_df = remove_typologies_from_color_df(
    ...     color_df=colors_df,
    ...     typologies_to_remove=["Open water", "Remaining"],
    ... )
    """
    # Create a copy and filter out specified typologies
    df = color_df[~color_df['type'].isin(typologies_to_remove)].copy()
    
    return df



################################################################################
## BUILD WINDOWS INFORMATION GDF
################################################################################
import geopandas as gpd
import pandas as pd
from typing import Mapping, Any, Union, Optional
from pathlib import Path


def build_windows_gdf(
    background_info_list: list[Mapping[str, Any]],
    default_crs: Optional[str] = None,
):
    """
    Build a background GeoDataFrame for plotting from location configuration.

    Parameters
    ----------
    background_info_list : list[Mapping[str, Any]]
        List of mappings describing the backgrounds per location.

        Required keys per item:
          - location : str
              Human-readable location name.
          - epsg : str
              EPSG/CRS token (e.g. `"UTM32631"`).
          - plot_window : str or Path or GeoDataFrame
              Path to a gpkg file or a GeoDataFrame containing the clip geometry.
          
        Optional keys per item:
          - crs : str
              EPSG code (e.g., "EPSG:28992" or "EPSG:32631").
              If not provided, will try to construct from `epsg` key or use default_crs.

    default_crs : str, optional
        Fallback CRS if individual items don't specify one.
        Example: "EPSG:28992" or "EPSG:32631"

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame indexed by `location` with columns:
          - epsg : str
              EPSG/CRS token for that row.
          - geometry : geometry
              The plot window/clip geometry for that location.
        
        The GeoDataFrame will have a valid `.crs` set.
    """

    required_keys = {"location", "epsg", "plot_window"}

    for i, info in enumerate(background_info_list):
        missing = sorted(required_keys - set(info))
        if missing:
            raise KeyError(f"Item {i} missing required keys: {missing}")

    # Process each item and load geometries
    processed_data = []
    crs_list = []
    
    for i, info in enumerate(background_info_list):
        plot_window = info["plot_window"]
        
        # Load geometry if it's a file path (string or Path object)
        if isinstance(plot_window, (str, Path)):
            window_gdf = gpd.read_file(plot_window)
            geom = window_gdf.geometry.iloc[0]
            source_crs = window_gdf.crs
        elif isinstance(plot_window, gpd.GeoDataFrame):
            geom = plot_window.geometry.iloc[0]
            source_crs = plot_window.crs
        else:
            # Assume it's already a geometry object
            geom = plot_window
            source_crs = None
        
        # Determine CRS for this item
        if "crs" in info:
            item_crs = info["crs"]
        elif source_crs is not None:
            item_crs = str(source_crs)
        elif info["epsg"].startswith("EPSG:"):
            item_crs = info["epsg"]
        elif info["epsg"].upper().startswith("UTM"):
            # Try to parse UTM zone (e.g., "UTM32631" -> "EPSG:32631")
            try:
                zone_code = info["epsg"].replace("UTM", "")
                item_crs = f"EPSG:{zone_code}"
            except:
                item_crs = default_crs
        else:
            item_crs = default_crs
        
        if item_crs is None:
            raise ValueError(
                f"Item {i} (location='{info['location']}'): "
                f"Could not determine CRS. Please provide 'crs' key or default_crs parameter."
            )
        
        crs_list.append(item_crs)
        
        processed_data.append({
            "location": info["location"],
            "epsg": info["epsg"],
            "geometry": geom
        })
    
    # Check if all items have the same CRS
    unique_crs = set(crs_list)
    if len(unique_crs) > 1:
        print(f"Warning: Mixed CRS detected: {unique_crs}. Using first CRS: {crs_list[0]}")
        # You could optionally reproject all to a common CRS here
    
    # Create GeoDataFrame with CRS
    gdf = gpd.GeoDataFrame(processed_data, geometry="geometry", crs=crs_list[0])
    
    return gdf



################################################################################
## BUILD AOI GDF
################################################################################
import warnings
import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Mapping, Any

from functions.gpkg_funcs import import_gpkg_func


def aoi_overlays_to_gdf(
    aoi_overlays: list[Mapping[str, Any]],
    *,
    crs=None,
) -> gpd.GeoDataFrame:
    """
    Expand AOI overlay specifications into a GeoDataFrame.

    The function reads each overlay geometry from file and expands the configuration
    into one row per (overlay description × year × feature). Any extra keys in the
    overlay dict (e.g., styling settings) are copied to output columns.

    Parameters
    ----------
    aoi_overlays : list[Mapping[str, Any]]
        List of overlay configuration mappings. Each item must contain:
          - description : str
              Overlay identifier; becomes the output index values (index name = "index").
          - gpkg_path : str | pathlib.Path
              Path to the GeoPackage (or other vector file supported by GeoPandas).
          - years : iterable[str | int]
              Year values to expand into separate rows. Numeric strings (e.g., "2017")
              are automatically converted to int.

        Optional keys:
          - epsg : Any
              CRS token used when `crs` is not provided. Accepted forms include
              "EPSG:32631", 32631, "32631", or "UTM32631" (converted to "EPSG:32631").
              If neither `crs` nor `epsg` is provided, a warning is raised and CRS is
              left to whatever is read from the file.
          - type : str
              Human-readable overlay label/category; defaults to None.
          - layer : str | int
              Layer passed through to `import_gpkg_func`.
          - any other keys
              Copied to output columns (e.g. facecolor, edgecolor, linewidth, linestyle).

    crs : Any, optional (keyword-only)
        If provided, overrides any overlay-level `epsg` and is passed to `import_gpkg_func`
        for all overlays.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with:
          - index: overlay `description` values (index name = "index")
          - columns: includes at least `years`, `type`, and `geometry`, plus any additional
            keys from the overlay dicts.
        Internal read-only keys `gpkg_path`, `layer`, and `epsg` are not included in the output.
        The output columns are ordered with `years` first and `type` second.

    Notes
    -----
    - Row expansion:
        rows = (#overlays) × (#years per overlay) × (#features in imported layer)
      If the imported layer contains multiple features, each feature is repeated for
      every requested year.
    - CRS handling:
        `crs` argument (if set) > overlay dict `epsg` (if set) > CRS as read from file
        (with a warning when neither is provided).
    - Year normalization:
        Digit-only year values are automatically converted to int (e.g., "2017" -> 2017).
    """

    rows = []
    for i, ov in enumerate(aoi_overlays):
        for k in ("description", "gpkg_path", "years"):
            if k not in ov:
                raise KeyError(f"Overlay {i} missing required key: {k!r}")

        description = str(ov["description"])
        gpkg_path = Path(ov["gpkg_path"])
        layer = ov.get("layer", None)

        # CRS resolution: function arg `crs` wins, else use dict `epsg`, else warn+None
        overlay_crs = crs
        if overlay_crs is None:
            overlay_crs = epsg_to_crs(ov.get("epsg", None))
            if overlay_crs is None:
                warnings.warn(
                    f"Overlay {description!r} has no 'epsg' and function arg `crs` is None. "
                    f"Falling back to CRS as read from file: {gpkg_path}",
                    UserWarning,
                )

        gdf = import_gpkg_func(gpkg_path, layer=layer, crs=overlay_crs)

        years_list = [normalize_year(y) for y in ov["years"]]

        meta = dict(ov)
        meta["description"] = description
        meta["years"] = years_list
        meta.setdefault("type", None)
        meta.setdefault("layer", None)

        # Remove epsg from meta - it's no longer needed after import
        meta.pop("epsg", None)

        for _, feat in gdf.iterrows():
            geom = feat.geometry
            for y in years_list:
                row = dict(meta)
                row["years"] = y
                row["geometry"] = geom
                rows.append(row)

    # Build output; drop internal columns
    out = gpd.GeoDataFrame(pd.DataFrame(rows), geometry="geometry")
    out = out.set_index("description")
    out.index.name = "index"
    out = out.drop(columns=[c for c in ("gpkg_path", "layer", "epsg") if c in out.columns])

    # Ensure a CRS is set on the output (prefer `crs`, else first non-null `epsg`, else keep as-is)
    if crs is not None:
        out = out.set_crs(crs, allow_override=True)
    else:
        # try to set from first overlay epsg if present
        for ov in aoi_overlays:
            c = epsg_to_crs(ov.get("epsg", None))
            if c is not None:
                out = out.set_crs(c, allow_override=True)
                break

    preferred = ["years", "type"]
    cols = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
    return out[cols]



################################################################################
## BUILD ALL INFORMATION GDF
################################################################################
import warnings
from pathlib import Path
from typing import Mapping, Any

import numpy as np
import pandas as pd
import geopandas as gpd


def all_pixels_gpkg_processing(
    all_pixels_gdf: gpd.GeoDataFrame,
    habitat_reference_df: pd.DataFrame,
    *,
    hab_selection: str,
    epsg: str | int | None = "EPSG:32631",
    years_col: str = "years",
    index_name: str = "index",
) -> gpd.GeoDataFrame:
    """
    Build a plot-ready overlay GeoDataFrame from pixel data and habitat reference tables.

    This function:
      1) Takes all_pixels_gdf with columns: years, habitatType1, geometry, etc.
      2) Joins with habitat_reference_df to map habitatType1 -> division & color
      3) Filters out rows where division is not assigned (NaN)
      4) Expands multi-year cells into one row per year
      5) Adds styling columns (facecolor/edgecolor/linewidth/linestyle)

    Parameters
    ----------
    all_pixels_gdf : gpd.GeoDataFrame
        Must contain columns: years, habitatType1, geometry.
    habitat_reference_df : pd.DataFrame
        Indexed by 'habitatType', contains columns like:
        - {hab_selection}_division (e.g., 'WD1_division')
        - {hab_selection}_color (e.g., 'WD1_color')
        Rows where the division column is NaN will be excluded from the output.
    hab_selection : str
        Division scheme to use (e.g., 'WD1', 'WD', 'WD2').
    epsg : str | int | None, default "EPSG:32631"
        Target CRS (e.g., 'EPSG:32631', 32631, "UTM32631").
        Accepts various EPSG formats and normalizes to 'EPSG:XXXXX'.
    years_col : str, default "years"
        Name of the years column in all_pixels_gdf.
    index_name : str, default "index"
        Name for the output GeoDataFrame index.

    Returns
    -------
    gpd.GeoDataFrame
        all_information_gdf : gpd.GeoDataFrame
            Plot-ready GeoDataFrame with columns:
            - years : int | str
                Single year per row (expanded from years_col).
            - type : str
                Division type from habitat_reference_df.
            - facecolor : str | None
                Color for the feature (None if not assigned in reference).
            - geometry : geometry
                Feature geometry.
            
            Indexed by index_name, with one row per (feature × year).
            Rows where the division is not assigned are excluded.

    Raises
    ------
    KeyError
        If required columns are missing in either input DataFrame, or if
        hab_selection columns don't exist in habitat_reference_df.

    Warns
    -----
    UserWarning
        - If rows are dropped due to missing division assignment
        - If all_pixels_gdf has no CRS and epsg=None

    Notes
    -----
    - **Filtering**: Rows where {hab_selection}_division is NaN (not assigned in the
      habitat reference) are automatically removed from the output. A warning is issued
      if any rows are dropped.
    - **NaN colors**: NaN values in the color column are replaced with None in the output.
    - **Year parsing**: The years column can contain:
        - Single values: "2017" → [2017]
        - Comma-separated: "2017,2018,2019" → [2017, 2018, 2019]
        - Range with ellipsis: "2018, 2019, ..., 2024" → [2018, 2019, 2020, 2021, 2022, 2023, 2024]
        - Lists/tuples/sets: [2017, 2018] → [2017, 2018]
      Numeric strings are automatically converted to integers.
    - **CRS handling**: The output CRS is set based on the epsg parameter. If the input
      has a different CRS, it will be reprojected.

    Examples
    --------
    >>> all_info_gdf = all_pixels_gpkg_processing(
    ...     all_pixels_gdf,
    ...     habitat_reference_df,
    ...     hab_selection="WD1",
    ...     epsg="EPSG:32631"
    ... )
    >>> all_info_gdf.head()
                  years    type  facecolor geometry
    index                                          
    0             2017  Forest    #2d5016  POLYGON(...)
    0             2018  Forest    #2d5016  POLYGON(...)
    """

    # Validate inputs
    if years_col not in all_pixels_gdf.columns:
        raise KeyError(f"all_pixels_gdf is missing column {years_col!r}.")
    if "habitatType1" not in all_pixels_gdf.columns:
        raise KeyError(f"all_pixels_gdf is missing column 'habitatType1'.")
    
    division_col = f"{hab_selection}_division"
    color_col = f"{hab_selection}_color"
    
    if division_col not in habitat_reference_df.columns:
        raise KeyError(f"habitat_reference_df is missing column {division_col!r}.")
    if color_col not in habitat_reference_df.columns:
        raise KeyError(f"habitat_reference_df is missing column {color_col!r}.")

    # Helper function
    def _parse_years_cell(v) -> list:
        """Parse years from various input formats, including ellipsis ranges."""
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return []
        if isinstance(v, (list, tuple, set)):
            return [normalize_year(x) for x in v]
        
        s = str(v).strip()
        
        # Check for ellipsis pattern: "2018, 2019, ..., 2024"
        if "..." in s:
            # Remove ellipsis and split by comma
            parts = [p.strip() for p in s.replace("...", "").split(",") if p.strip()]
            
            if len(parts) >= 2:
                # Get first and last year
                try:
                    start_year = normalize_year(parts[0])
                    end_year = normalize_year(parts[-1])
                    
                    # Generate range
                    return list(range(start_year, end_year + 1))
                except (ValueError, TypeError):
                    # If parsing fails, fall back to regular comma-separated parsing
                    pass
        
        # Regular comma-separated parsing
        if "," in s:
            parts = [p.strip() for p in s.split(",") if p.strip() != ""]
            return [normalize_year(p) for p in parts]
        
        return [normalize_year(s)]

    # Ensure CRS
    target_crs = epsg_to_crs(epsg) if epsg is not None else None
    base = all_pixels_gdf.copy()
    
    if target_crs is not None:
        if base.crs is None:
            base = base.set_crs(target_crs, allow_override=True)
        elif base.crs != target_crs:
            base = base.to_crs(target_crs)
    else:
        if base.crs is None:
            warnings.warn("all_pixels_gdf has no CRS and epsg=None; output CRS will be None.", UserWarning)

    # Join with habitat reference to get division and color
    # Reset index temporarily to preserve it
    base = base.reset_index()
    base = base.merge(
        habitat_reference_df[[division_col, color_col]],
        left_on="habitatType1",
        right_index=True,
        how="left",
    )
    base = base.set_index("index")

    # Replace NaN colors with None
    base[color_col] = base[color_col].replace({np.nan: None})

    # Filter out rows where division is NaN (not assigned)
    n_before = len(base)
    base = base.dropna(subset=[division_col])
    n_dropped = n_before - len(base)

    if n_dropped > 0:
        print(
            f"Dropped {n_dropped} row(s) with no assigned Typology in {division_col!r}"
        )

    # Expand years and build output rows
    rows = []
    for idx, r in base.iterrows():
        years_list = _parse_years_cell(r[years_col])
        if not years_list:
            continue

        ftype = r[division_col]
        face = r[color_col]

        for y in years_list:
            rows.append(
                {
                    "index": idx,
                    "years": y,
                    "type": ftype,
                    "facecolor": face,
                    "geometry": r.geometry,
                }
            )

    # Build output GeoDataFrame
    out = gpd.GeoDataFrame(pd.DataFrame(rows), geometry="geometry", crs=base.crs)
    out = out.set_index("index")
    out.index.name = index_name

    cols = ["years", "type", "facecolor", "geometry"]
    out = out[[c for c in cols if c in out.columns]]

    return out



################################################################################
## BUILD TRAINING/VALIDATION PIXELS DF
################################################################################
import warnings
import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Any, Mapping

from functions.gpkg_funcs import import_gpkg_func


def training_validation_pixels_gdf(
    pixels_dir: Path | str,
    hab_selection: str,
    train_split_attempt: str,
    epsg: str | int,
    type_color_df: pd.DataFrame,
    *,
    type_color_df_type_col: str = "type",
    type_color_df_color_col: str = "color",
    filename_pattern: str | None = None,
    years_col: str = "years",
    index_name: str = "index",
) -> gpd.GeoDataFrame:
    """
    Import training or validation pixels from a GeoPackage and prepare for plotting.

    Reads a pixels GeoPackage, maps types to colors using the provided type-color
    DataFrame, and adds styling columns for plotting. The output is indexed and
    column-ordered for direct use in visualization pipelines.

    Parameters
    ----------
    pixels_dir : Path | str
        Directory containing the pixels GeoPackage.
    hab_selection : str
        Habitat selection code (e.g., "WD1", "WD2") used in the filename.
    train_split_attempt : str
        Training split attempt code (e.g., "at1", "at2") used in the filename.
    epsg : str | int
        EPSG/CRS code (e.g., "UTM32631", "EPSG:32631", 32631).
        Used both for the filename and to set the CRS if missing.
    type_color_df : pd.DataFrame
        DataFrame mapping habitat types to colors with columns 'type' and 'color'.
        Typically the second output from `all_pixels_gpkg_processing()`.
    filename_pattern : str, optional
        Custom filename pattern with placeholders for formatting.
        Must include {hab_selection}, {train_split_attempt}, and {epsg} placeholders.
        
        Default pattern (if None):
        "validation_pixels__{hab_selection}_plusOW_p80_tmp2_ML_{train_split_attempt}__gelderland__{epsg}__gpkg.gpkg"
        
        Example for training pixels:
        "training_pixels__{hab_selection}_plusOW_p80_tmp2_ML_{train_split_attempt}__gelderland__{epsg}__gpkg.gpkg"
        
        **Note**: This function assumes all GeoPackage files matching the pattern are
        in the same directory (`pixels_dir`). It does not support subdirectory traversal
        or multiple file locations.
    years_col : str, default "years"
        Name of the years column (must exist in the GeoPackage).
    index_name : str, default "index"
        Name for the output index.

    Returns
    -------
    gpd.GeoDataFrame
        Plot-ready GeoDataFrame with columns ordered as:
        [years_col, 'type', 'facecolor', 'edgecolor', 'linewidth', 'linestyle', 'geometry']
        
        Indexed by the 'index' column from the source file.

    Raises
    ------
    FileNotFoundError
        If the constructed GeoPackage path does not exist.
    KeyError
        If required columns ('type', years_col) are missing from the GeoPackage.
    ValueError
        If type_color_df is missing required columns ('type', 'color').

    Warns
    -----
    UserWarning
        - If any 'type' values are not found in type_color_df
        - If the GeoPackage has no CRS

    Notes
    -----
    - **Filename pattern**: The default pattern expects validation pixels with the format:
      `validation_pixels__{hab_selection}_plusOW_p80_tmp2_ML_{train_split_attempt}__gelderland__{epsg}__gpkg.gpkg`
      
      Use `filename_pattern` to override for training pixels or custom naming schemes.
    
    - **Directory constraint**: All files must be in the same directory (`pixels_dir`).
      The function does not search subdirectories or handle files in different locations.
    
    - **Missing colors**: Types not in type_color_df get facecolor=None with a warning.
    
    - **Index handling**: If the GeoPackage lacks an 'index' column, the current
      index is reset and used.
    
    - **EPSG in filename**: The EPSG code is automatically formatted as "UTM{code}"
      for the filename (e.g., "EPSG:32631" becomes "UTM32631").

    Examples
    --------
    >>> # Using default pattern for validation pixels
    >>> _, type_colors = all_pixels_gpkg_processing(...)
    >>> val_gdf = training_validation_pixels_gdf(
    ...     pixels_dir="data/validation",
    ...     hab_selection="WD1",
    ...     train_split_attempt="at1",
    ...     epsg="EPSG:32631",
    ...     type_color_df=type_colors
    ... )
    
    >>> # Using custom pattern for training pixels
    >>> train_gdf = training_validation_pixels_gdf(
    ...     pixels_dir="data/training",
    ...     hab_selection="WD1",
    ...     train_split_attempt="at1",
    ...     epsg=32631,
    ...     type_color_df=type_colors,
    ...     filename_pattern="training_pixels__{hab_selection}_plusOW_p80_tmp2_ML_{train_split_attempt}__gelderland__{epsg}__gpkg.gpkg"
    ... )
    
    >>> val_gdf.head()
              years      type facecolor edgecolor  linewidth linestyle geometry
    index                                                                       
    0          2017    Forest   #2d5016     black        0.5     solid  POLYGON(...)
    1          2018  Grassland   #7cb342     black        0.5     solid  POLYGON(...)
    """
    pixels_dir = Path(pixels_dir)

    # Validate type_color_df structure
    required_color_cols = [type_color_df_type_col, type_color_df_color_col]
    missing_color_cols = [col for col in required_color_cols if col not in type_color_df.columns]
    if missing_color_cols:
        raise ValueError(
            f"type_color_df must contain columns: {required_color_cols}. "
            f"Missing: {missing_color_cols}"
        )

    # Normalize EPSG for filename construction
    # Extract numeric part for filename (e.g., "EPSG:32631" -> "UTM32631")
    epsg_normalized = epsg_to_crs(epsg)  # "EPSG:32631"
    epsg_code = epsg_normalized.replace("EPSG:", "") if epsg_normalized else str(epsg)
    epsg_filename = f"UTM{epsg_code}"

    # Use default pattern if not provided
    if filename_pattern is None:
        filename_pattern = (
            "validation_pixels__{hab_selection}_plusOW_p80_tmp2_ML_{train_split_attempt}"
            "__gelderland__{epsg}__gpkg.gpkg"
        )

    # Construct GeoPackage path using the pattern
    filename = filename_pattern.format(
        hab_selection=hab_selection,
        train_split_attempt=train_split_attempt,
        epsg=epsg_filename
    )
    pixels_path = pixels_dir / filename

    # Check if file exists
    if not pixels_path.exists():
        raise FileNotFoundError(
            f"Pixels GeoPackage not found: {pixels_path}\n"
            f"Directory: {pixels_dir}\n"
            f"Pattern: {filename_pattern}"
        )

    # Import GeoPackage with CRS handling
    gdf = import_gpkg_func(pixels_path, crs=epsg_normalized)

    # Validate required columns
    required_cols = ["type", years_col]
    missing_cols = [col for col in required_cols if col not in gdf.columns]
    if missing_cols:
        raise KeyError(
            f"Missing required column(s) {missing_cols} in {pixels_path.name}"
        )

    # Ensure 'index' column exists
    if "index" not in gdf.columns:
        gdf = gdf.reset_index()
        if "index" not in gdf.columns:
            # If reset_index didn't create 'index', create it manually
            gdf["index"] = range(len(gdf))

    # Map types to colors using type_color_df
    # Create a dictionary mapping type -> color
    color_mapping = dict(zip(type_color_df[type_color_df_type_col], type_color_df[type_color_df_color_col]))
    gdf["facecolor"] = gdf["type"].map(color_mapping)
    
    # Warn about missing type mappings
    missing_types = gdf[gdf["facecolor"].isna()]["type"].unique()
    if len(missing_types) > 0:
        warnings.warn(
            f"Types not found in type_color_df (will use None): {sorted(missing_types)}",
            UserWarning
        )

    # Set index
    gdf = gdf.set_index("index", drop=True)
    gdf.index.name = index_name

    # Order columns for consistency
    column_order = [years_col, "type", "facecolor", "geometry"]
    available_cols = [c for c in column_order if c in gdf.columns]
    gdf = gdf[available_cols]

    return gpd.GeoDataFrame(gdf, geometry="geometry", crs=gdf.crs)



################################################################################
## CLIPPING TO WINDOW EXTENT
################################################################################
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.windows import from_bounds
import numpy as np

def clip_gdf_to_window(
    gdf: gpd.GeoDataFrame,
    window_gdf: gpd.GeoDataFrame,
    year: int | None = None,
) -> gpd.GeoDataFrame:
    """
    Filter a GeoDataFrame to polygons within a plot window, optionally filtering by year.
    
    Returns complete polygons that intersect the window boundary - does NOT clip/cut geometries.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame (can be much larger than the window)
    window_gdf : gpd.GeoDataFrame
        Window defining the spatial extent to filter to
    year : int, optional
        If provided, filter gdf by 'years' column before spatial filtering
    
    Returns
    -------
    gpd.GeoDataFrame
        Filtered GeoDataFrame with complete geometries in the same CRS as window_gdf
    """
    if gdf.empty:
        return gdf.copy()
    
    # 1. Filter by year if provided
    if year is not None and 'years' in gdf.columns:
        gdf = gdf[gdf['years'] == year].copy()
    
    if gdf.empty:
        return gdf
    
    # 2. Reproject to window CRS if needed
    if gdf.crs != window_gdf.crs:
        gdf = gdf.to_crs(window_gdf.crs)
    
    # 3. Create bounding box from window
    window_union = window_gdf.geometry.union_all()
    
    # 4. Filter to polygons that intersect the window (keeps complete geometries)
    mask = gdf.geometry.intersects(window_union)
    filtered = gdf[mask].copy()
    
    return filtered



def clip_raster_to_window(
    raster_path: str,
    window_gdf: gpd.GeoDataFrame,
) -> tuple[np.ndarray, dict, tuple]:
    """
    Clip a raster to a plot window and return image, metadata, and bounds.
    """
    with rasterio.open(raster_path) as src:
        # 1. Reproject window to raster CRS
        if window_gdf.crs != src.crs:
            window_reproj = window_gdf.to_crs(src.crs)
        else:
            window_reproj = window_gdf
        
        # 2. Clip raster using mask
        clipped_array, clipped_transform = mask(
            src, 
            window_reproj.geometry, 
            crop=True,
            filled=True,
            all_touched=True,
        )
        
        # 3. Calculate bounds from the clipped array shape and transform
        height, width = clipped_array.shape[1], clipped_array.shape[2]
        bounds_raster_crs = rasterio.transform.array_bounds(
            height,
            width,
            clipped_transform
        )
        
        # 4. Transform bounds back to window CRS for plotting
        if window_gdf.crs != src.crs:
            from rasterio.warp import transform_bounds
            bounds_window_crs = transform_bounds(
                src.crs,
                window_gdf.crs,
                *bounds_raster_crs
            )
        else:
            bounds_window_crs = bounds_raster_crs
        
        # 5. Prepare metadata
        metadata = {
            'transform': clipped_transform,
            'crs': src.crs,
            'height': height,
            'width': width,
            'count': clipped_array.shape[0],
        }
        
        return clipped_array, metadata, bounds_window_crs



################################################################################
## ADD NORTHARROW
################################################################################
import io
import cairosvg
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patches as patches
from pathlib import Path
    
def add_svg_northarrow(
    ax: plt.Axes,
    svg_path: str | Path,
    location: tuple[float, float] = (0.92, 0.89),
    zoom: float = 0.04,
    zorder: int = 50,
    output_width: int = 800,
    background_color: str = "transparent",
) -> AnnotationBbox:
    """
    Add an SVG north arrow to a matplotlib axes.
    
    Uses axes fraction coordinates for consistent positioning across plots.
    
    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to add north arrow to.
    svg_path : str | Path
        Path to the SVG file.
    location : tuple[float, float], default (0.93, 0.91)
        (x, y) position in axes fraction coordinates (0-1).
    zoom : float, default 0.04
        Zoom factor for the north arrow image.
    zorder : int, default 50
        Z-order for layering (higher = on top).
    output_width : int, default 800
        PNG render width in pixels (higher = better quality).
    background_color : str, default "transparent"
        Background color for SVG rendering.
    
    Returns
    -------
    AnnotationBbox
        The added annotation box artist.
    """
    from pathlib import Path
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    import matplotlib.image as mpimg
    import io
    import cairosvg
    
    # Convert to Path object and verify
    svg_path = Path(svg_path)
    
    if not svg_path.exists():
        raise FileNotFoundError(f"SVG file not found: {svg_path}")
    
    # Convert SVG to PNG
    with open(svg_path, 'rb') as f:
        png_bytes = cairosvg.svg2png(
            file_obj=f,
            background_color=background_color,
            output_width=output_width
        )

    # Load PNG as image
    img = mpimg.imread(io.BytesIO(png_bytes), format="png")

    # Create offset image and annotation box
    oi = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(
        oi,
        location,  # Use location tuple directly
        xycoords=ax.transAxes,  # Axes fraction coordinates
        frameon=False,
        box_alignment=(0.5, 0.5),  # Center the image on the location point
        zorder=zorder
    )
    ax.add_artist(ab)
    
    return ab


def get_default_north_arrow_kwargs(figsize: tuple[float, float]) -> dict:
    """
    Generate default north arrow kwargs scaled to figure size.
    
    Uses consistent scaling relative to a 4x4 inch base size.
    
    Parameters
    ----------
    figsize : tuple[float, float]
        (width, height) of the axis in inches
    
    Returns
    -------
    dict
        Default kwargs for add_svg_northarrow()
    """
    base_size = 4.0
    scale_factor = min(figsize) / base_size
    
    # Base values for 4x4 inch figure
    base_zoom = 0.04
    base_output_width = 800
    
    return {
        'location': (0.92, 0.90),
        'zoom': base_zoom * scale_factor,
        'zorder': 50,
        'output_width': int(base_output_width * scale_factor),
        'background_color': 'transparent',
    }



################################################################################
## ADD SCALEBAR
################################################################################
def add_bw_scalebar(ax, length_m=150, segment_m=50,
                    location=(0.05, 0.05), height=0.015,
                    text_offset=0.015,
                    edgecolor="black", lw=0.5, fontsize=8,
                    bar_width_fraction=0.20):  
    """
    Black-white-black style scalebar in DATA coordinates.
    
    Parameters
    ----------
    length_m : float
        Actual length in meters (map units)
    segment_m : float
        Segment length in meters
    location : tuple
        (x_frac, y_frac) in axes fraction (0-1), converted to data coords
    height : float
        Height as fraction of y-axis range
    text_offset : float
        Text offset as fraction of y-axis range
    bar_width_fraction : float
        Not used (width determined by length_m in data coords)
    """
    from matplotlib import patches
    
    # Get axis limits in data coordinates
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    
    # Convert axes fractions to data coordinates
    x_frac, y_frac = location
    x0 = xlim[0] + x_frac * x_range
    y0 = ylim[0] + y_frac * y_range
    
    # Convert height and text_offset from fractions to data units
    height_data = height * y_range
    text_offset_data = text_offset * y_range
    
    nseg = int(round(length_m / segment_m))
    seg_w = length_m / nseg

    # Draw segments in DATA coordinates
    for i in range(nseg):
        fc = "black" if i % 2 == 0 else "white"
        r = patches.Rectangle((x0 + i*seg_w, y0), seg_w, height_data,
                              facecolor=fc,
                              edgecolor='none',
                              linewidth=0, 
                              alpha=1.0,
                              zorder=10)
        ax.add_patch(r)

    # Outline
    outline = patches.Rectangle((x0, y0), length_m, height_data,
                                facecolor="none",
                                edgecolor=edgecolor, 
                                linewidth=lw, 
                                alpha=1.0,
                                zorder=11)
    ax.add_patch(outline)

    # Text label
    ax.text(x0 + length_m/2, y0 - text_offset_data, f"{length_m:g} m",
            ha="center", va="top",
            fontsize=fontsize, color="black", zorder=12)


def get_default_scalebar_kwargs(figsize: tuple[float, float], 
                                 ax=None,
                                 target_fraction=0.20) -> dict:
    """
    Generate default scalebar kwargs scaled to figure size.
    Optionally auto-sizes based on axis extent.
    """
    base_size = 4.0
    scale_factor = min(figsize) / base_size
    
    # Auto-determine appropriate length if axis provided
    length_m = 150  # default
    if ax is not None:
        xlim = ax.get_xlim()
        x_range = xlim[1] - xlim[0]
        target_length = x_range * target_fraction
        
        # Round to nice number (50, 100, 150, 200, 500, 1000, etc.)
        import math
        magnitude = 10 ** math.floor(math.log10(target_length))
        nice_numbers = [1, 1.5, 2, 5]
        length_m = min([n * magnitude for n in nice_numbers], 
                       key=lambda x: abs(x - target_length))
    
    segment_m = length_m / 3  # Always 3 segments
    
    # Positioning in lower right
    y_location = 0.06 + 0.01 * (scale_factor - 1)
    x_location = 0.92 - scale_factor * 0.17
    
    return {
        'length_m': length_m,
        'segment_m': segment_m,
        'location': (x_location, y_location),
        'height': 0.015 * scale_factor,
        'text_offset': 0.015 * scale_factor,
        'edgecolor': 'black',
        'lw': 0.5 * scale_factor,
        'fontsize': 8 * scale_factor,
    }



################################################################################
## ADD LEGEND
################################################################################
def add_color_legend(
    ax: plt.Axes,
    color_df: pd.DataFrame,
    location: tuple[float, float] = (0, 0),
    anchor: str = 'lower left',
    frameon: bool = True,
    framealpha: float = 0.7,
    edgecolor: str = 'black',
    facecolor: str = 'white',
    fontsize: int = 9,
    title: str | None = None,
    title_fontsize: int | None = None,
    ncol: int = 1,
    linewidth: float = 0.5,
    handlelength: float = 1.5,
    handleheight: float = 1.0,
    columnspacing: float = 1.0,
    labelspacing: float = 0.5,
    label_col: str = 'type',
) -> None:
    """
    Add a color legend to a matplotlib axes based on a color DataFrame.
    
    Uses axes fraction coordinates for consistent positioning across plots.
    
    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to add legend to.
    color_df : pd.DataFrame
        DataFrame with columns for labels and 'color' column with hex colors.
        Must contain at minimum: label_col (default 'class_name') and 'color'.
    location : tuple[float, float], default (0.01, 0.01)
        (x, y) position in axes fraction coordinates (0-1).
    anchor : str, default 'lower left'
        Which corner of the legend box to anchor at location.
    frameon : bool, default True
        Whether to draw a frame around the legend.
    framealpha : float, default 0.7
        Transparency of the legend frame.
    edgecolor : str, default 'black'
        Color of the legend frame edge.
    facecolor : str, default 'white'
        Background color of the legend.
    fontsize : int, default 9
        Font size for legend labels.
    title : str | None, default None
        Title for the legend.
    title_fontsize : int | None, default None
        Font size for legend title.
    ncol : int, default 1
        Number of columns in the legend.
    linewidth : float, default 0.5
        Width of patch borders.
    handlelength : float, default 1.5
        Length of legend handles in font-size units.
    handleheight : float, default 1.0
        Height of legend handles in font-size units.
    columnspacing : float, default 1.0
        Spacing between columns in font-size units.
    labelspacing : float, default 0.5
        Vertical spacing between entries in font-size units.
    label_col : str, default 'class_name'
        Name of the column containing the labels for the legend.
    
    Returns
    -------
    matplotlib.legend.Legend
        The legend object that was added to the axes.
    
    Raises
    ------
    ValueError
        If color_df is missing required columns ('color' or label_col).
    
    Examples
    --------
    >>> color_df = pd.DataFrame({
    ...     'type': ['Forest', 'Grassland', 'Urban'],
    ...     'color': ['#2d5016', '#7cb342', '#d32f2f']
    ... })
    >>> fig, ax = plt.subplots()
    >>> add_color_legend(ax, color_df, title='Land Cover')
    
    >>> # Using a different label column
    >>> type_df = pd.DataFrame({
    ...     'type': ['Water', 'Vegetation', 'Built-up'],
    ...     'color': ['#0000ff', '#00ff00', '#ff0000']
    ... })
    >>> add_color_legend(ax, type_df, label_col='type')
    """
    from matplotlib.patches import Patch
    
    # Validate required columns
    required_cols = ['color', label_col]
    missing_cols = [col for col in required_cols if col not in color_df.columns]
    if missing_cols:
        raise ValueError(
            f"color_df must contain columns: {required_cols}. "
            f"Missing: {missing_cols}"
        )
    
    handles = []
    labels = []
    
    for _, row in color_df.iterrows():
        label = row[label_col]
        color = row['color']
        patch = Patch(facecolor=color, edgecolor='black', linewidth=linewidth)
        handles.append(patch)
        labels.append(str(label))
    
    if title_fontsize is None:
        title_fontsize = fontsize
    
    # Map anchor string to loc parameter
    loc_map = {
        'lower right': 'lower right',
        'lower left': 'lower left',
        'upper right': 'upper right',
        'upper left': 'upper left',
        'center': 'center',
    }
    loc = loc_map.get(anchor, 'lower right')
    
    legend = ax.legend(
        handles=handles,
        labels=labels,
        loc=loc,
        bbox_to_anchor=location,
        bbox_transform=ax.transAxes,  # Explicitly use axes coordinates
        frameon=frameon,
        framealpha=framealpha,
        edgecolor=edgecolor,
        facecolor=facecolor,
        fontsize=fontsize,
        title=title,
        title_fontsize=title_fontsize,
        ncol=ncol,
        handlelength=handlelength,
        handleheight=handleheight,
        columnspacing=columnspacing,
        labelspacing=labelspacing,
    )
    
    return legend


def get_default_legend_kwargs(cell_figsize: tuple[float, float]) -> dict:
    """
    Get default kwargs for color legend scaled to cell size.
    
    Uses consistent scaling relative to a 4x4 inch base size.
    
    Parameters
    ----------
    cell_figsize : tuple[float, float]
        (width, height) of the cell in inches.
    
    Returns
    -------
    dict
        Default kwargs for add_color_legend.
    """
    base_size = 4.0
    scale_factor = min(cell_figsize) / base_size
    
    # Scale all size-dependent parameters
    base_fontsize = 9
    base_linewidth = 0.5
    base_handlelength = 1.5
    base_handleheight = 1.0
    base_columnspacing = 1.0
    base_labelspacing = 0.5
    
    return {
        'location': (0.0, 0.0),
        'anchor': 'lower left',
        'frameon': True,
        'framealpha': 0.7,
        'edgecolor': 'black',
        'facecolor': 'white',
        'fontsize': int(base_fontsize * scale_factor),
        'title': None,
        'title_fontsize': None,
        'ncol': 1,
        'linewidth': base_linewidth * scale_factor,
        'handlelength': base_handlelength * scale_factor,
        'handleheight': base_handleheight * scale_factor,
        'columnspacing': base_columnspacing * scale_factor,
        'labelspacing': base_labelspacing * scale_factor,
    }



################################################################################
## ADD LINE LEGEND
################################################################################
def add_line_legend(ax, legend_items, location=(0.03, 0.03), 
                     spacing=0.035, line_length=0.08,
                     fontsize=8, edgecolor='black', lw=0.5):
    """
    Add a custom legend with line samples in DATA coordinates.
    
    Parameters
    ----------
    ax : matplotlib axis
        The axis to add legend to
    legend_items : list of dict
        Each dict should contain: 'label', 'edgecolor', 'linestyle', 'linewidth'
    location : tuple
        (x_frac, y_frac) in axes fraction (0-1), converted to data coords
        y_frac represents the BOTTOM of the legend box
    spacing : float
        Vertical spacing between items as fraction of y-axis range
    line_length : float
        Length of line sample as fraction of x-axis range
    fontsize : float
        Font size for labels
    edgecolor : str
        Color of legend box edge
    lw : float
        Line width of legend box
    """
    from matplotlib import patches
    import matplotlib.lines as mlines
    
    # Get axis limits in data coordinates
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    
    # Convert location to data coordinates (location is BOTTOM left)
    x_frac, y_frac = location
    x0 = xlim[0] + x_frac * x_range
    y_bottom = ylim[0] + y_frac * y_range
    
    # Convert spacing and line_length to data units
    spacing_data = spacing * y_range
    line_length_data = line_length * x_range
    text_offset_data = 0.015 * x_range  # Gap between line and text
    
    # Calculate legend box dimensions
    n_items = len(legend_items)
    vertical_padding = 0.025 * y_range  # ✅ Increased padding
    horizontal_padding = 0.015 * x_range  # ✅ Left margin inside box
    box_height = (n_items - 1) * spacing_data + 2 * vertical_padding
    box_width = line_length_data + 0.25 * x_range  # ✅ Larger width
    
    # Draw background box (starting from bottom)
    bg_box = patches.Rectangle(
        (x0, y_bottom), box_width, box_height,
        facecolor='white', edgecolor=edgecolor, linewidth=lw,
        alpha=0.9, zorder=20
    )
    ax.add_patch(bg_box)
    
    # Draw each legend item (from bottom to top)
    for i, item in enumerate(reversed(legend_items)):
        y_pos = y_bottom + vertical_padding + i * spacing_data
        
        # Draw line sample
        line = mlines.Line2D(
            [x0 + horizontal_padding, x0 + horizontal_padding + line_length_data],
            [y_pos, y_pos],
            color=item.get('edgecolor', 'black'),
            linestyle=item.get('linestyle', 'solid'),
            linewidth=item.get('linewidth', 1.0),
            zorder=21
        )
        ax.add_line(line)
        
        # Draw text label
        ax.text(
            x0 + horizontal_padding + line_length_data + text_offset_data,
            y_pos,
            item.get('label', ''),
            ha='left', va='center',
            fontsize=fontsize,
            color='black',
            zorder=21
        )


def get_default_line_legend_kwargs(figsize: tuple[float, float]) -> dict:
    """
    Generate default legend kwargs scaled to figure size.
    Positioned in lower left corner (mirroring scalebar in lower right).
    """
    base_size = 4.0
    scale_factor = min(figsize) / base_size
    
    # Position in lower left corner
    y_location = 0.015 + 0.005 * (scale_factor - 1)
    x_location = 0.015 + 0.005 * (scale_factor - 1)
    
    return {
        'location': (x_location, y_location),
        'spacing': 0.035 * scale_factor,
        'line_length': 0.08 * scale_factor,
        'fontsize': 8 * scale_factor,
        'edgecolor': 'black',
        'lw': 0.5 * scale_factor,
    }



################################################################################
## ADD CONSISTENCY LEGEND
################################################################################
def get_default_consistency_legend_kwargs(cell_figsize: tuple[float, float]) -> dict:
    """Get default kwargs for consistency legend based on cell size."""
    width, height = cell_figsize
    base_size = min(width, height)
    
    return {
        'loc': 'lower right',
        'fontsize': max(6, base_size * 2),
        'framealpha': 0.8,
        'title': 'Agreement',
        'title_fontsize': max(7, base_size * 2.2),
    }


def add_consistency_legend(
    ax: plt.Axes,
    loc: str = 'lower right',
    fontsize: float = 8,
    framealpha: float = 0.8,
    title: str = 'Model Agreement',
    title_fontsize: float = 9,
    tie_color: str = '#e9830e',
    chaos_color: str = '#ff0000',
) -> None:
    """
    Add consistency legend explaining saturation-based agreement visualization.
    
    Saturation indicates agreement strength:
    - Vivid colors = high agreement
    - Pale colors = low agreement
    - Special colors for ties and complete chaos
    """
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    legend_elements = [
        # Text-only header for saturation explanation
        Line2D([0], [0], marker='', color='none', 
               label='Saturation = classification consistency',
               markersize=0, linewidth=0),
        # Special cases
        Patch(facecolor=tie_color, edgecolor='black', linewidth=0.5, 
              label=f'Tied classifications'),
        Patch(facecolor=chaos_color, edgecolor='black', linewidth=0.5, 
              label=f'Unstable classifications'),
    ]
    
    ax.legend(
        handles=legend_elements,
        loc=loc,
        fontsize=fontsize,
        framealpha=framealpha,
        title=title,
        title_fontsize=title_fontsize,
    )



################################################################################
## BUILD DF FROM RASTER LEGEND
################################################################################
import rasterio
import pandas as pd
from pathlib import Path


def build_legend_df_from_raster(
    raster_path: str | Path,
    label_col: str = 'classes'
) -> pd.DataFrame:
    """
    Extract legend from raster metadata and build a DataFrame for legend plotting.
    
    Reads the LEGEND tag from raster metadata and converts it to a DataFrame
    with 'pixel_value', label column, and 'color' columns.
    
    Parameters
    ----------
    raster_path : str | Path
        Path to the raster file with LEGEND metadata.
    label_col : str, default 'classes'
        Name for the label column in output DataFrame.
        Common values: 'classes', 'description', 'type'
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'pixel_value': str, the keys from the LEGEND dict
        - label_col: str, the class/category labels
        - 'color': str, hex color codes
        - 'rgb': list, RGB values (optional, for reference)
    
    Raises
    ------
    ValueError
        If raster has no LEGEND metadata or LEGEND is malformed.
    FileNotFoundError
        If raster_path does not exist.
    
    Examples
    --------
    >>> # From raster with LEGEND metadata
    >>> legend_df = build_legend_df_from_raster("stability.tif")
    >>> legend_df
      pixel_value                        classes   color              rgb
    0         1_2  Remaining / Semi-Wet Nature  #749c9a  [116, 156, 154]
    1         1_3       Remaining / Wet Nature  #6da36d  [109, 163, 109]
    2         2_3   Semi-Wet Nature / Wet Nature  #38965e    [56, 150, 94]
    3  highly_unstable            Highly Unstable  #FF0000    [255, 0, 0]
    
    >>> # Use with add_color_legend
    >>> add_color_legend(ax, legend_df, label_col='classes', title='Stability')
    """
    import json
    
    raster_path = Path(raster_path)
    
    # Check if file exists
    if not raster_path.exists():
        raise FileNotFoundError(f"Raster file not found: {raster_path}")
    
    # Open raster and read LEGEND metadata
    with rasterio.open(raster_path) as src:
        legend_str = src.tags().get('LEGEND')
        
        if legend_str is None:
            raise ValueError(f"No LEGEND metadata found in {raster_path.name}")
    
    # Parse LEGEND string to dict
    try:
        legend_dict = json.loads(legend_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LEGEND metadata: {e}")
    
    # Build DataFrame
    records = []
    for pixel_value, attributes in legend_dict.items():
        record = {
            'pixel_value': pixel_value,
            label_col: attributes.get('classes', ''),
            'color': attributes.get('color', '#000000'),
            'rgb': attributes.get('rgb', [0, 0, 0])
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    
    # Validate required columns
    if df.empty:
        raise ValueError(f"LEGEND metadata is empty in {raster_path.name}")
    
    if df['color'].isna().any():
        print(f"Warning: Some legend entries missing 'color' in {raster_path.name}")
    
    return df

def build_legend_df_from_dict(
    legend_dict: dict,
    label_col: str = 'classes'
) -> pd.DataFrame:
    """
    Build legend DataFrame directly from a LEGEND dictionary.
    
    Useful when you already have the legend dict in memory and don't need
    to read it from a raster file.
    
    Parameters
    ----------
    legend_dict : dict
        Dictionary with pixel values as keys and attributes as values.
        Each value dict should contain 'classes', 'color', and optionally 'rgb'.
    label_col : str, default 'classes'
        Name for the label column in output DataFrame.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: 'pixel_value', label_col, 'color', 'rgb'
    
    Examples
    --------
    >>> legend = {
    ...     "1_2": {"classes": "Remaining / Semi-Wet", "color": "#749c9a"},
    ...     "1_3": {"classes": "Remaining / Wet", "color": "#6da36d"}
    ... }
    >>> df = build_legend_df_from_dict(legend)
    """
    records = []
    for pixel_value, attributes in legend_dict.items():
        record = {
            'pixel_value': pixel_value,
            label_col: attributes.get('classes', ''),
            'color': attributes.get('color', '#000000'),
            'rgb': attributes.get('rgb', [0, 0, 0])
        }
        records.append(record)
    
    return pd.DataFrame(records)



################################################################################
## ADD CLASS PERFORMANCE 
################################################################################
def add_class_performance_badges(
    ax: plt.Axes,
    metrics_df: pd.DataFrame,
    year: int,
    color_df: pd.DataFrame,
    cell_figsize: tuple[float, float] = (4, 4),
    box_height: float = 0.12,
    padding: float = 0.01,
    font_size: int = 8,
    text_color: str = 'black',
    tile_size: float = 0.025,
    tile_spacing: float = 0.01,
    max_classes: int = 8,
    show_class_names: bool = True,
    class_name_font_size: int = 8,
    row_spacing_pts: float = 12.0, 
    class_name_overrides: dict[str, str] = None,
    column_spacing_pts: float = 30.0,  
) -> None:
    """
    Add a box below the plot showing class performance metrics.
    
    Layout (4 rows):
    - Row 1: [■ Class Name]  [■ Class Name]  [■ Class Name]
    - Row 2:    n=150           n=200           n=175
    - Row 3:   acc=0.92        acc=0.88        acc=0.91
    - Row 4:   f1=0.88         f1=0.85         f1=0.87
    
    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to add the badges to
    metrics_df : pd.DataFrame
        DataFrame containing performance metrics per class
    year : int
        Year to filter metrics for
    color_df : pd.DataFrame
        DataFrame with class colors (columns: type, color)
    cell_figsize : tuple[float, float]
        Size of the cell (width, height) in inches
    box_height : float
        Height of the box in axes coordinates
    padding : float
        Padding around elements
    font_size : int
        Font size for metric text
    text_color : str
        Color for the text
    tile_size : float
        Size of colored tiles
    tile_spacing : float
        Horizontal space between tile and class name
    max_classes : int
        Maximum number of classes to display
    show_class_names : bool
        If True, show class names next to tiles
    class_name_font_size : int
        Font size for class name labels
    row_spacing_pts : float
        Absolute spacing between rows in points (default: 12 pts)
    class_name_overrides : dict[str, str], optional
        Dictionary mapping original class names to display names
        Example: {"Remaining Terrestrial": "Remaining", "Wet Nature": "Wet"}
    column_spacing_pts : float
        Absolute spacing between columns in points (default: 30 pts)
    """
    from matplotlib.patches import Rectangle
    import numpy as np
    
    # Filter metrics for the specific year
    year_metrics = metrics_df[metrics_df['clas_year'] == year]
    
    if year_metrics.empty:
        print(f"    Warning: No metrics found for year {year}")
        print(f"    Available years: {metrics_df['clas_year'].unique()}")
        return
    
    # Extract classes from color_df
    if 'type' in color_df.columns:
        classes = color_df['type'].tolist()
        colors = color_df['color'].tolist()
    elif color_df.index.name == 'type' or (color_df.index.name and 'type' in str(color_df.index.name).lower()):
        classes = color_df.index.tolist()
        colors = color_df['color'].tolist()
    else:
        classes = color_df.index.tolist()
        colors = color_df['color'].tolist()
    
    # Limit to max_classes
    if len(classes) > max_classes:
        print(f"    Warning: {len(classes)} classes found, showing only first {max_classes}")
        classes = classes[:max_classes]
        colors = colors[:max_classes]
    
    n_classes = len(classes)
    
    if n_classes == 0:
        print(f"    Warning: No classes found in color_df")
        return
    
    top_margin = 0.002
    box_height_ax = box_height / cell_figsize[1]
    box_y = -top_margin - box_height_ax
    
    n_rows = 4
    
    dpi = 72.0
    row_spacing_inches = row_spacing_pts / dpi
    row_spacing_ax = row_spacing_inches / cell_figsize[1]
    
    row_centers = []
    current_y = box_y + box_height_ax - row_spacing_ax
    
    for i in range(n_rows):
        row_centers.append(current_y)
        current_y -= row_spacing_ax
    
    column_spacing_inches = column_spacing_pts / dpi
    column_spacing_ax = column_spacing_inches / cell_figsize[0]
    
    total_content_width = (n_classes - 1) * column_spacing_ax if n_classes > 1 else 0
    start_x = (1.0 - total_content_width) / 2
    
    column_positions = []
    for i in range(n_classes):
        column_positions.append(start_x + i * column_spacing_ax)
    
    effective_tile_size = min(tile_size, row_spacing_ax * 0.6)
    
    def find_metric_column(prefix: str, class_name: str, df_columns: list) -> str:
        """Find a column matching pattern: {prefix}_{class_name}"""
        variations = [
            class_name,
            class_name.replace(' ', '_'),
            class_name.replace(' ', '-'),
            class_name.replace('-', '_'),
            class_name.replace('_', '-'),
        ]
        
        col_map = {col.lower(): col for col in df_columns}
        
        for variation in variations:
            target = f"{prefix}_{variation}".lower()
            if target in col_map:
                return col_map[target]
        
        return None
    
    def get_display_name(class_name: str) -> str:
        """Get display name for class (with optional override)."""
        if class_name_overrides and class_name in class_name_overrides:
            return class_name_overrides[class_name]
        return class_name
    
    # Helper to estimate text width (rough approximation)
    def estimate_text_width(text: str, fontsize: int) -> float:
        """Estimate text width in axes coordinates."""
        char_width_inches = fontsize / 72.0 * 0.6  # Rough estimate
        text_width_inches = len(text) * char_width_inches
        return text_width_inches / cell_figsize[0]
    
    df_columns = year_metrics.columns.tolist()
    
    for idx, (class_name, color) in enumerate(zip(classes, colors)):
        n_col = find_metric_column('n', class_name, df_columns)
        acc_col = find_metric_column('acc', class_name, df_columns)
        f1_col = find_metric_column('f1', class_name, df_columns)
        
        if not n_col:
            print(f"    Warning: Could not find 'n' column for class '{class_name}'")
        if not acc_col:
            print(f"    Warning: Could not find 'acc' column for class '{class_name}'")
        if not f1_col:
            print(f"    Warning: Could not find 'f1' column for class '{class_name}'")
        
        n_val = year_metrics[n_col].values[0] if n_col else np.nan
        acc_val = year_metrics[acc_col].values[0] if acc_col else np.nan
        f1_val = year_metrics[f1_col].values[0] if f1_col else np.nan
        
        column_center_x = column_positions[idx]
        
        # Calculate total width of tile+text combination for Row 1
        display_name = get_display_name(class_name) if show_class_names else ""
        text_width = estimate_text_width(display_name, class_name_font_size) if show_class_names else 0
        total_width = effective_tile_size + (tile_spacing + text_width if show_class_names else 0)
        
        # Start tile+text group at column center minus half of total width
        group_start_x = column_center_x - total_width / 2
        
        # Tile position (left side of group)
        tile_x = group_start_x
        tile_y = row_centers[0] - effective_tile_size / 2
        
        tile_rect = Rectangle(
            (tile_x, tile_y),
            effective_tile_size,
            effective_tile_size,
            transform=ax.transAxes,
            facecolor=color,
            edgecolor='black',
            linewidth=0.5,
            zorder=11,
            clip_on=False,
        )
        ax.add_patch(tile_rect)
        
        if show_class_names:
            name_x = tile_x + effective_tile_size + tile_spacing
            ax.text(
                name_x,
                row_centers[0],
                display_name,
                transform=ax.transAxes,
                ha='left',
                va='center',
                fontsize=class_name_font_size,
                color=text_color,
                style='italic',
                zorder=11,
                clip_on=False,
            )
        
        n_str = f"n={int(n_val)}" if not np.isnan(n_val) and n_val > 0 else "n=-"
        ax.text(
            column_center_x,
            row_centers[1],
            n_str,
            transform=ax.transAxes,
            ha='center',
            va='center',
            fontsize=font_size,
            color=text_color,
            zorder=11,
            clip_on=False,
        )
        
        acc_str = f"acc={acc_val:.2f}" if not np.isnan(acc_val) else "acc=-"
        ax.text(
            column_center_x,
            row_centers[2],
            acc_str,
            transform=ax.transAxes,
            ha='center',
            va='center',
            fontsize=font_size,
            color=text_color,
            zorder=11,
            clip_on=False,
        )
        
        f1_str = f"f1={f1_val:.2f}" if not np.isnan(f1_val) else "f1=-"
        ax.text(
            column_center_x,
            row_centers[3],
            f1_str,
            transform=ax.transAxes,
            ha='center',
            va='center',
            fontsize=font_size,
            color=text_color,
            zorder=11,
            clip_on=False,
        )


def get_default_class_performance_kwargs(cell_figsize: tuple[float, float]) -> dict:
    """Get default kwargs for class performance badges based on cell size."""
    return {
        'cell_figsize': cell_figsize,
        'box_height': 0.65,
        'padding': 0.01,
        'font_size': 8,
        'text_color': 'black',
        'tile_size': 0.025,
        'tile_spacing': 0.01,
        'max_classes': 8,
        'show_class_names': True,
        'class_name_font_size': 8,
        'row_spacing_pts': 12.0,
        'class_name_overrides': None,
        'column_spacing_pts': 30.0,
    }



################################################################################
## ADD PIXEL COUNTS
################################################################################
def add_class_pixel_counts(
    ax: plt.Axes,
    gdf: gpd.GeoDataFrame,
    color_df: pd.DataFrame,
    year: int = None,  
    year_column: str = 'years',  
    class_column: str = 'type',
    cell_figsize: tuple[float, float] = (4, 4),
    box_height: float = 0.12,
    padding: float = 0.01,
    font_size: int = 8,
    text_color: str = 'black',
    show_percentage: bool = True,
    tile_size: float = 0.025,
    tile_spacing: float = 0.01,
    max_classes: int = 6,
    show_class_names: bool = True,
    class_name_font_size: int = 8,
    row_spacing_pts: float = 12.0,
    column_spacing_pts: float = 30.0,
    class_name_overrides: dict[str, str] = None,
    horizontal_offset_pts: float = 0.0,
) -> None:
    """
    Add a box below the plot showing pixel counts per class for a specific year.
    
    Layout (3 rows):
    - Row 1: [■ Class Name]  [■ Class Name]  [■ Class Name]
    - Row 2:    n=1,250         n=2,340         n=1,875
    - Row 3:     (25.3%)        (32.1%)         (42.6%)
    
    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to add the tiles to
    gdf : gpd.GeoDataFrame
        GeoDataFrame with pixel data
    color_df : pd.DataFrame
        DataFrame with class colors (columns: type, color)
    year : int, optional
        Year to filter data for
    year_column : str
        Column name containing year information
    class_column : str
        Column name containing class labels
    cell_figsize : tuple[float, float]
        Size of the cell (width, height) in inches
    box_height : float
        Height of the box in axes coordinates
    padding : float
        Padding around elements
    font_size : int
        Font size for count/percentage text
    text_color : str
        Color for the text
    show_percentage : bool
        If True, show percentages in row 3
    tile_size : float
        Size of colored tiles
    tile_spacing : float
        Horizontal space between tile and class name
    max_classes : int
        Maximum number of classes to display
    show_class_names : bool
        If True, show class names next to tiles
    class_name_font_size : int
        Font size for class name labels
    row_spacing_pts : float
        Absolute spacing between rows in points
    column_spacing_pts : float
        Absolute spacing between columns in points
    class_name_overrides : dict[str, str], optional
        Dictionary mapping original class names to display names
    horizontal_offset_pts : float
        Absolute horizontal offset in points (negative = left, positive = right)
        Default 0.0 centers the legend
    """
    from matplotlib.patches import Rectangle
    import numpy as np
    import pandas as pd
    
    if gdf.empty:
        print(f"    Warning: GeoDataFrame is empty")
        return
    
    if class_column not in gdf.columns:
        print(f"    Warning: Column '{class_column}' not found in GeoDataFrame. Available: {list(gdf.columns)}")
        return
    
    if year is not None:
        if year_column not in gdf.columns:
            print(f"    Warning: Year column '{year_column}' not found in GeoDataFrame. Using all data.")
            gdf_filtered = gdf
        else:
            try:
                gdf_year_int = gdf[year_column].astype(int)
                gdf_filtered = gdf[gdf_year_int == int(year)]
            except (ValueError, TypeError):
                gdf_filtered = gdf[gdf[year_column] == str(year)]
            
            if gdf_filtered.empty:
                print(f"    Warning: No data found for year {year}")
                unique_years = gdf[year_column].unique()
                print(f"    Available years in GDF: {sorted(unique_years)}")
                gdf_filtered = gpd.GeoDataFrame()
    else:
        gdf_filtered = gdf
    
    pixel_counts = gdf_filtered[class_column].value_counts() if not gdf_filtered.empty else pd.Series(dtype=int)
    total_pixels = len(gdf_filtered)
    
    if 'type' in color_df.columns:
        all_classes = color_df['type'].tolist()
        color_map = dict(zip(color_df['type'], color_df['color']))
    elif color_df.index.name == 'type' or 'type' in str(color_df.index.name).lower():
        all_classes = color_df.index.tolist()
        color_map = dict(zip(color_df.index, color_df['color']))
    else:
        all_classes = color_df.index.tolist()
        color_map = dict(zip(color_df.index, color_df['color']))
    
    if len(all_classes) > max_classes:
        print(f"    Warning: color_df contains {len(all_classes)} classes, but only the first {max_classes} will be displayed.")
        print(f"    Displayed classes: {all_classes[:max_classes]}")
        print(f"    Omitted classes: {all_classes[max_classes:]}")
        classes = all_classes[:max_classes]
    else:
        classes = all_classes
    
    colors = [color_map[cls] for cls in classes]
    n_classes = len(classes)
    
    if n_classes == 0:
        print(f"    Warning: No classes found in color_df")
        return
    
    top_margin = 0.002
    box_height_ax = box_height / cell_figsize[1]
    box_y = -top_margin - box_height_ax
    
    n_rows = 3 if show_percentage else 2
    
    dpi = 72.0
    row_spacing_inches = row_spacing_pts / dpi
    row_spacing_ax = row_spacing_inches / cell_figsize[1]
    
    row_centers = []
    current_y = box_y + box_height_ax - row_spacing_ax
    
    for i in range(n_rows):
        row_centers.append(current_y)
        current_y -= row_spacing_ax
    
    column_spacing_inches = column_spacing_pts / dpi
    column_spacing_ax = column_spacing_inches / cell_figsize[0]
    
    # Convert horizontal offset from points to axes coordinates
    horizontal_offset_inches = horizontal_offset_pts / dpi
    horizontal_offset_ax = horizontal_offset_inches / cell_figsize[0]
    
    total_content_width = (n_classes - 1) * column_spacing_ax if n_classes > 1 else 0
    start_x = (1.0 - total_content_width) / 2 + horizontal_offset_ax
    
    column_positions = []
    for i in range(n_classes):
        column_positions.append(start_x + i * column_spacing_ax)
    
    effective_tile_size = min(tile_size, row_spacing_ax * 0.6)
    
    def get_display_name(class_name: str) -> str:
        if class_name_overrides and class_name in class_name_overrides:
            return class_name_overrides[class_name]
        return class_name
    
    # Helper to estimate text width (rough approximation)
    def estimate_text_width(text: str, fontsize: int) -> float:
        """Estimate text width in axes coordinates."""
        char_width_inches = fontsize / 72.0 * 0.6  # Rough estimate
        text_width_inches = len(text) * char_width_inches
        return text_width_inches / cell_figsize[0]
    
    for idx, (class_name, color) in enumerate(zip(classes, colors)):
        n_pixels = pixel_counts.get(class_name, 0)
        percentage = (n_pixels / total_pixels * 100) if total_pixels > 0 else 0
        
        column_center_x = column_positions[idx]
        
        # Calculate total width of tile+text combination for Row 1
        display_name = get_display_name(class_name) if show_class_names else ""
        text_width = estimate_text_width(display_name, class_name_font_size) if show_class_names else 0
        total_width = effective_tile_size + (tile_spacing + text_width if show_class_names else 0)
        
        # Start tile+text group at column center minus half of total width
        group_start_x = column_center_x - total_width / 2
        
        # Tile position (left side of group)
        tile_x = group_start_x
        tile_y = row_centers[0] - effective_tile_size / 2
        
        tile_rect = Rectangle(
            (tile_x, tile_y),
            effective_tile_size,
            effective_tile_size,
            transform=ax.transAxes,
            facecolor=color,
            edgecolor='black',
            linewidth=0.5,
            zorder=11,
            clip_on=False,
        )
        ax.add_patch(tile_rect)
        
        if show_class_names:
            name_x = tile_x + effective_tile_size + tile_spacing
            ax.text(
                name_x,
                row_centers[0],
                display_name,
                transform=ax.transAxes,
                ha='left',
                va='center',
                fontsize=class_name_font_size,
                color=text_color,
                style='italic',
                zorder=11,
                clip_on=False,
            )
        
        count_text = f"n={n_pixels:,}" if n_pixels > 0 else "n=-"
        ax.text(
            column_center_x,
            row_centers[1],
            count_text,
            transform=ax.transAxes,
            ha='center',
            va='center',
            fontsize=font_size,
            color=text_color,
            zorder=11,
            clip_on=False,
        )
        
        if show_percentage:
            percentage_text = f"({percentage:.1f}%)" if n_pixels > 0 else "(-)"
            ax.text(
                column_center_x,
                row_centers[2],
                percentage_text,
                transform=ax.transAxes,
                ha='center',
                va='center',
                fontsize=font_size,
                color=text_color,
                zorder=11,
                clip_on=False,
            )


def get_default_class_pixel_counts_kwargs(cell_figsize: tuple[float, float]) -> dict:
    """Get default kwargs for class pixel count badges based on cell size."""
    return {
        'cell_figsize': cell_figsize,
        'box_height': 0.45,  
        'padding': 0.01,
        'font_size': 8,
        'text_color': 'black',
        'show_percentage': True,
        'class_column': 'type',
        'year_column': 'year',
        'tile_size': 0.025,
        'tile_spacing': 0.01,
        'max_classes': 6,
        'show_class_names': True,
        'class_name_font_size': 8,
        'row_spacing_pts': 12.0,
        'column_spacing_pts': 30.0,
        'class_name_overrides': None,
        'horizontal_offset_pts': 0.0,
    }



################################################################################
## ADD STABILITY PERCENTAGES
################################################################################
def add_stability_percentages(
    ax: plt.Axes,
    raster_path: Path,
    color_df: pd.DataFrame,
    cell_figsize: tuple[float, float] = (4, 4),
    box_height: float = 0.12,
    padding: float = 0.01,
    font_size: int = 8,
    text_color: str = 'black',
    tile_size: float = 0.025,
    tile_spacing: float = 0.01,
    show_description: bool = True,
    description_font_size: int = 8,
    row_spacing_pts: float = 12.0,
    column_spacing_pts: float = 30.0,
    horizontal_offset_pts: float = 0.0,
    percentage_horizontal_offset_pts: float = 0.0,
) -> None:
    """
    Add a box below the plot showing stability percentages from raster metadata.
    
    Layout (2 rows):
    - Row 1: [■ Description] [■ Description] [■ Description]
    - Row 2:    [25.3%]          [32.1%]          [42.6%]
    
    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to add the tiles to
    raster_path : Path
        Path to the decision_category.tif raster file
    color_df : pd.DataFrame
        DataFrame with columns: value, description, color
    cell_figsize : tuple[float, float]
        Size of the cell (width, height) in inches
    box_height : float
        Height of the box in axes coordinates
    padding : float
        Padding around elements
    font_size : int
        Font size for percentage text
    text_color : str
        Color for the text
    tile_size : float
        Size of colored tiles
    tile_spacing : float
        Horizontal space between tile and description
    show_description : bool
        Whether to show description labels
    description_font_size : int
        Font size for description text
    row_spacing_pts : float
        Absolute spacing between rows in points
    column_spacing_pts : float
        Absolute spacing between columns in points
    horizontal_offset_pts : float
        Absolute horizontal offset in points for row 1 (negative = left, positive = right)
        Default 0.0 centers the legend
    percentage_horizontal_offset_pts : float
        Absolute horizontal offset in points for row 2 percentages (negative = left, positive = right)
        Default 0.0 uses same centering as row 1
    """
    
    from matplotlib.patches import Rectangle
    import rasterio
    import ast
    
    if not raster_path.exists():
        print(f"    Warning: Raster file not found: {raster_path}")
        return
    
    try:
        with rasterio.open(raster_path) as src:
            tags = src.tags()
            percentages_str = tags.get('VALUE-PERCENTAGES', None)
            
            if percentages_str is None:
                print(f"    Warning: VALUE-PERCENTAGES tag not found in raster")
                return
            
            percentages_dict = ast.literal_eval(percentages_str)
            percentages_dict = {int(k): v for k, v in percentages_dict.items()}
            
    except Exception as e:
        print(f"    Warning: Failed to read raster percentages: {e}")
        import traceback
        traceback.print_exc()
        return
    
    color_map = dict(zip(color_df['value'], color_df['color']))
    description_map = dict(zip(color_df['value'], color_df['description']))
    
    present_values = sorted(percentages_dict.keys())
    
    colors = [color_map.get(val, '#808080') for val in present_values]
    percentages = [percentages_dict[val] for val in present_values]
    descriptions = [description_map.get(val, f'Value {val}') for val in present_values]
    
    n_classes = len(present_values)
    
    if n_classes == 0:
        print(f"    Warning: No valid stability values found")
        return
    
    top_margin = 0.002
    box_height_ax = box_height / cell_figsize[1]
    box_y = -top_margin - box_height_ax

    n_rows = 2
    
    dpi = 72.0
    row_spacing_inches = row_spacing_pts / dpi
    row_spacing_ax = row_spacing_inches / cell_figsize[1]
    
    row_centers = []
    current_y = box_y + box_height_ax - row_spacing_ax
    
    for i in range(n_rows):
        row_centers.append(current_y)
        current_y -= row_spacing_ax
    
    column_spacing_inches = column_spacing_pts / dpi
    column_spacing_ax = column_spacing_inches / cell_figsize[0]
    
    # Convert horizontal offset from points to axes coordinates
    horizontal_offset_inches = horizontal_offset_pts / dpi
    horizontal_offset_ax = horizontal_offset_inches / cell_figsize[0]
    
    # Convert percentage horizontal offset from points to axes coordinates
    percentage_horizontal_offset_inches = percentage_horizontal_offset_pts / dpi
    percentage_horizontal_offset_ax = percentage_horizontal_offset_inches / cell_figsize[0]
    
    total_content_width = (n_classes - 1) * column_spacing_ax if n_classes > 1 else 0
    start_x = (1.0 - total_content_width) / 2 + horizontal_offset_ax
    
    column_positions = []
    for i in range(n_classes):
        column_positions.append(start_x + i * column_spacing_ax)
    
    effective_tile_size = min(tile_size, row_spacing_ax * 0.6)
    
    # Helper to estimate text width (rough approximation)
    def estimate_text_width(text: str, fontsize: int) -> float:
        """Estimate text width in axes coordinates."""
        char_width_inches = fontsize / 72.0 * 0.6  # Rough estimate
        text_width_inches = len(text) * char_width_inches
        return text_width_inches / cell_figsize[0]
    
    for idx, (pixel_value, color, percentage, description) in enumerate(
        zip(present_values, colors, percentages, descriptions)
    ):
        column_center_x = column_positions[idx]
        
        # Calculate total width of tile+text combination for Row 1
        text_width = estimate_text_width(description, description_font_size) if show_description else 0
        total_width = effective_tile_size + (tile_spacing + text_width if show_description else 0)
        
        # Start tile+text group at column center minus half of total width
        group_start_x = column_center_x - total_width / 2
        
        # Tile position (left side of group)
        tile_x = group_start_x
        tile_y = row_centers[0] - effective_tile_size / 2
        
        tile_rect = Rectangle(
            (tile_x, tile_y),
            effective_tile_size,
            effective_tile_size,
            transform=ax.transAxes,
            facecolor=color,
            edgecolor='black',
            linewidth=0.5,
            zorder=11,
            clip_on=False,
        )
        ax.add_patch(tile_rect)
        
        if show_description:
            desc_x = tile_x + effective_tile_size + tile_spacing
            ax.text(
                desc_x,
                row_centers[0],
                description,
                transform=ax.transAxes,
                ha='left',
                va='center',
                fontsize=description_font_size,
                color=text_color,
                style='italic',
                zorder=11,
                clip_on=False,
            )
        
        percentage_text = f"[{percentage:.1f}%]"
        ax.text(
            column_center_x + percentage_horizontal_offset_ax,
            row_centers[1],
            percentage_text,
            transform=ax.transAxes,
            ha='center',
            va='center',
            fontsize=font_size,
            color=text_color,
            zorder=11,
            clip_on=False,
        )


def get_default_stability_percentages_kwargs(cell_figsize: tuple[float, float]) -> dict:
    """Get default kwargs for stability percentage tiles based on cell size."""
    return {
        'cell_figsize': cell_figsize,
        'box_height': 0.5,
        'padding': 0.01,
        'font_size': 8,
        'text_color': 'black',
        'tile_size': 0.025,
        'tile_spacing': 0.01,
        'show_description': True,
        'description_font_size': 8,
        'row_spacing_pts': 12.0,
        'column_spacing_pts': 30.0,
        'horizontal_offset_pts': 0.0,
        'percentage_horizontal_offset_pts': 0.0,
    }



################################################################################
## ADD THE STABLE PIXELS TILES
################################################################################
def add_alpha_legend(
    ax: plt.Axes,
    color_df: pd.DataFrame,
    alpha_map: dict[int, float],
    cell_figsize: tuple[float, float] = (4, 4),
    box_height: float = 0.12,
    font_size: int = 8,
    text_color: str = 'black',
    tile_size: float = 0.025,
    tile_spacing: float = 0.01,
    show_class_names: bool = True,
    class_name_font_size: int = 8,
    row_spacing_pts: float = 12.0,
    column_spacing: float = 0.33,
    class_name_overrides: dict[str, str] = None,
    stability_labels: dict[int, str] = None,
    horizontal_offset_pts: float = 0.0,
) -> None:
    """
    Add a legend showing how alpha values are applied to each class.
    
    Centered in the axes coordinate system (0-1), where 0.5 is the middle of the figure.
    """
    from matplotlib.patches import Rectangle
    import numpy as np
    
    if 'type' in color_df.columns:
        classes = color_df['type'].tolist()
        colors = color_df['color'].tolist()
    elif color_df.index.name == 'type' or (color_df.index.name and 'type' in str(color_df.index.name).lower()):
        classes = color_df.index.tolist()
        colors = color_df['color'].tolist()
    else:
        classes = color_df.index.tolist()
        colors = color_df['color'].tolist()
    
    n_classes = len(classes)
    
    if n_classes == 0:
        print(f"    Warning: No classes found in color_df")
        return
    
    sorted_alpha_items = sorted(alpha_map.items())
    n_alpha_rows = len(sorted_alpha_items)
    
    top_margin = 0.002
    box_height_ax = box_height / cell_figsize[1]
    box_y = -top_margin - box_height_ax
    
    n_rows = 1 + n_alpha_rows
    
    dpi = 72.0
    row_spacing_inches = row_spacing_pts / dpi
    row_spacing_ax = row_spacing_inches / cell_figsize[1]
    
    row_centers = []
    current_y = box_y + box_height_ax - row_spacing_ax
    
    for i in range(n_rows):
        row_centers.append(current_y)
        current_y -= row_spacing_ax
    
    # Convert horizontal offset from points to axes coordinates
    horizontal_offset_inches = horizontal_offset_pts / dpi
    horizontal_offset_ax = horizontal_offset_inches / cell_figsize[0]

    # ADJUSTED: For non-square figures, we need to account for aspect ratio
    # The visual center is still at 0.5 in axes coords, but we need to 
    # think in terms of visual balance
    aspect_ratio = cell_figsize[0] / cell_figsize[1]  # width / height

    # Calculate column positions based on number of classes
    if n_classes % 2 == 0:
        # Even number: center around 0.5
        total_width = (n_classes - 1) * column_spacing
        base_start = 0.5 - total_width / 2
    else:
        # Odd number: center column at 0.5
        middle_idx = n_classes // 2
        base_start = 0.5 - middle_idx * column_spacing

    # Apply horizontal offset to all columns
    column_positions = [base_start + i * column_spacing + horizontal_offset_ax for i in range(n_classes)]
    
    effective_tile_size = min(tile_size, row_spacing_ax * 0.6)
    
    def get_display_name(class_name: str) -> str:
        if class_name_overrides and class_name in class_name_overrides:
            return class_name_overrides[class_name]
        return class_name
    
    for idx, (class_name, color) in enumerate(zip(classes, colors)):
        tile_x = column_positions[idx]
        
        # Row 1: Class name (aligned with tiles below, no invisible tile)
        if show_class_names:
            display_name = get_display_name(class_name)
            ax.text(
                tile_x,
                row_centers[0],
                display_name,
                transform=ax.transAxes,
                ha='left',
                va='center',
                fontsize=class_name_font_size,
                color=text_color,
                style='italic',
                zorder=11,
                clip_on=False,
            )
        
        # Alpha stability rows (2+)
        for alpha_idx, (stability_value, alpha_value) in enumerate(sorted_alpha_items):
            row_idx = alpha_idx + 1
            
            if stability_labels and stability_value in stability_labels:
                label = stability_labels[stability_value]
            else:
                label = f"α={alpha_value:.1f}"
            
            tile_y_alpha = row_centers[row_idx] - effective_tile_size / 2
            
            tile_rect_alpha = Rectangle(
                (tile_x, tile_y_alpha),
                effective_tile_size,
                effective_tile_size,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor='black',
                linewidth=0.5,
                alpha=alpha_value,
                zorder=11,
                clip_on=False,
            )
            ax.add_patch(tile_rect_alpha)
            
            label_x = tile_x + effective_tile_size + tile_spacing
            ax.text(
                label_x,
                row_centers[row_idx],
                label,
                transform=ax.transAxes,
                ha='left',
                va='center',
                fontsize=font_size,
                color=text_color,
                zorder=11,
                clip_on=False,
            )


def get_default_alpha_legend_kwargs(cell_figsize: tuple[float, float]) -> dict:
    """Get default kwargs for alpha legend based on cell size."""
    return {
        'cell_figsize': cell_figsize,
        'box_height': 0.5,
        'font_size': 8,
        'text_color': 'black',
        'tile_size': 0.025,
        'tile_spacing': 0.01,
        'show_class_names': True,
        'class_name_font_size': 8,
        'row_spacing_pts': 12.0,
        'column_spacing': 0.33,
        'class_name_overrides': None,
        'stability_labels': None,
        'horizontal_offset_pts': 0.0,
    }



################################################################################
## ADD UNSTABLE PIXELS LEGEND
################################################################################
def add_unstable_pixels_legend(
    ax: plt.Axes,
    color_df: pd.DataFrame,
    cell_figsize: tuple[float, float] = (4, 4),
    box_height: float = 0.12,
    padding: float = 0.01,
    font_size: int = 8,
    text_color: str = 'black',
    tile_size: float = 0.025,
    tile_spacing: float = 0.01,
    row_spacing_pts: float = 12.0,
    left_margin: float = 0.05,
    label_col: str = 'description',
    label_overrides: dict[str, str] = None,
    tiles_per_row: int = 2,
    col1_horizontal_offset_pts: float = 0.0,
    col2_horizontal_offset_pts: float = 0.0,
) -> None:
    """
    Add a legend showing unstable pixel categories below the plot.
    
    Layout (2 tiles per row, fixed positions):
    - Row 1: [■ Description 1]                    [■ Description 2]
    - Row 2: [■ Description 3]                    [■ Description 4]
    
    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to add the legend to
    color_df : pd.DataFrame
        DataFrame with unstable pixel types and colors
        Must contain columns: label_col (default 'description') and 'color'
    cell_figsize : tuple[float, float]
        Size of the cell (width, height) in inches
    box_height : float
        Height of the box in axes coordinates
    padding : float
        Padding around elements
    font_size : int
        Font size for description text
    text_color : str
        Color for the text
    tile_size : float
        Size of colored tiles
    tile_spacing : float
        Horizontal space between tile and description
    row_spacing_pts : float
        Absolute spacing between rows in points
    left_margin : float
        Left margin for first column in axes coordinates (0-1)
    label_col : str
        Column name containing the labels/descriptions
    label_overrides : dict[str, str], optional
        Dictionary mapping original labels to display labels
    tiles_per_row : int
        Number of tiles to show per row (default 2)
    col1_horizontal_offset_pts : float
        Absolute horizontal offset for column 1 in points (negative = left, positive = right)
        Default 0.0 uses default positioning
    col2_horizontal_offset_pts : float
        Absolute horizontal offset for column 2 in points (negative = left, positive = right)
        Default 0.0 uses default positioning
    """
    from matplotlib.patches import Rectangle
    import numpy as np
    import math
    
    if label_col not in color_df.columns:
        raise ValueError(f"color_df must contain column '{label_col}'")
    if 'color' not in color_df.columns:
        raise ValueError("color_df must contain column 'color'")
    
    labels = color_df[label_col].tolist()
    colors = color_df['color'].tolist()
    n_items = len(labels)
    
    if n_items == 0:
        print(f"    Warning: No items found in color_df")
        return
    
    n_rows = math.ceil(n_items / tiles_per_row)
    
    top_margin = 0.002
    box_height_ax = box_height / cell_figsize[1]
    box_y = -top_margin - box_height_ax
    
    dpi = 72.0
    row_spacing_inches = row_spacing_pts / dpi
    row_spacing_ax = row_spacing_inches / cell_figsize[1]
    
    row_centers = []
    current_y = box_y + box_height_ax - row_spacing_ax
    
    for i in range(n_rows):
        row_centers.append(current_y)
        current_y -= row_spacing_ax
    
    # Convert horizontal offsets from points to axes coordinates
    col_offsets_pts = [col1_horizontal_offset_pts, col2_horizontal_offset_pts]
    col_offsets_ax = []
    for offset_pts in col_offsets_pts:
        offset_inches = offset_pts / dpi
        offset_ax = offset_inches / cell_figsize[0]
        col_offsets_ax.append(offset_ax)
    
    # Fixed column positions with column-specific offsets
    column_positions = [
        left_margin + col_offsets_ax[0],        
        0.5 + col_offsets_ax[1],
    ]
    
    effective_tile_size = min(tile_size, row_spacing_ax * 0.6)
    
    def get_display_label(label: str) -> str:
        if label_overrides and label in label_overrides:
            return label_overrides[label]
        return label
    
    for idx, (label, color) in enumerate(zip(labels, colors)):
        row_idx = idx // tiles_per_row
        col_idx = idx % tiles_per_row
        
        tile_x = column_positions[col_idx]
        tile_y = row_centers[row_idx] - effective_tile_size / 2
        
        tile_rect = Rectangle(
            (tile_x, tile_y),
            effective_tile_size,
            effective_tile_size,
            transform=ax.transAxes,
            facecolor=color,
            edgecolor='black',
            linewidth=0.5,
            zorder=11,
            clip_on=False,
        )
        ax.add_patch(tile_rect)
        
        display_label = get_display_label(label)
        text_x = tile_x + effective_tile_size + tile_spacing
        ax.text(
            text_x,
            row_centers[row_idx],
            display_label,
            transform=ax.transAxes,
            ha='left',
            va='center',
            fontsize=font_size,
            color=text_color,
            zorder=11,
            clip_on=False,
        )


def get_default_unstable_pixels_legend_kwargs(cell_figsize: tuple[float, float]) -> dict:
    """Get default kwargs for unstable pixels legend based on cell size."""
    return {
        'cell_figsize': cell_figsize,
        'box_height': 0.5,
        'padding': 0.01,
        'font_size': 8,
        'text_color': 'black',
        'tile_size': 0.025,
        'tile_spacing': 0.01,
        'row_spacing_pts': 12.0,
        'left_margin': 0.05,
        'label_col': 'description',
        'label_overrides': None,
        'tiles_per_row': 2,
        'col1_horizontal_offset_pts': 0.0,
        'col2_horizontal_offset_pts': 0.0,
    }



################################################################################
## WORK WITH {FORMATS}
################################################################################
def resolve_background_path(
    path_template: str,
    location: str,
    **kwargs
) -> Path:
    """
    Resolve a path template with location and other variables.
    
    Parameters
    ----------
    path_template : str
        Path template with placeholders like {location}, {hab_selection}, etc.
    location : str
        Location name to substitute
    **kwargs
        Additional variables for substitution
    
    Returns
    -------
    Path
        Resolved path
    
    Raises
    ------
    KeyError
        If a required placeholder variable is missing
    ValueError
        If path formatting fails
    """
    from pathlib import Path
    
    path_template = path_template.strip("'\"")
    
    format_vars = {'location': location}
    format_vars.update(kwargs)
    
    try:
        resolved = path_template.format(**format_vars)
        return Path(resolved)
    
    except KeyError as e:
        raise KeyError(f"Missing placeholder {e} in path: {path_template}")
    
    except Exception as e:
        raise ValueError(f"Failed to resolve path '{path_template}': {e}")


def resolve_stability_overlay_path(
    path_template: str,
    location: str,
    hab_selection: str,
    train_split_attempt: str,
    band_selection: str,
    vis_years: str,
    year: int,
    **kwargs
) -> Path:
    """
    Resolve a stability overlay path template for a specific year.
    
    Parameters
    ----------
    path_template : str
        Path template with placeholders
    location : str
        Location name
    hab_selection : str
        Habitat selection identifier
    train_split_attempt : str
        Training split identifier
    band_selection : str
        Band selection identifier (e.g., "Q1234")
    vis_years : str
        Visualization years description (e.g., "2020-2022")
    year : int
        Specific year for this overlay raster
    **kwargs
        Additional variables for substitution
    
    Returns
    -------
    Path
        Resolved path for the specific year
    """
    from pathlib import Path
    
    path_template = path_template.strip("'\"")
      
    # Build format variables
    format_vars = {
        'location': location,
        'hab_selection': hab_selection,
        'train_split_attempt': train_split_attempt,
        'band_selection': band_selection,
        'year': year,
        'vis_years': vis_years,
        'vis_years_description': vis_years,
        'raster_stack': f"{band_selection}",
    }
    
    # Override with kwargs
    format_vars.update(kwargs)
    
    try:
        resolved = path_template.format(**format_vars)
        return Path(resolved)
    except KeyError as e:
        available_keys = ', '.join(format_vars.keys())
        raise KeyError(
            f"Missing placeholder {e} in path template.\n"
            f"Template: {path_template}\n"
            f"Available keys: {available_keys}"
        )
    except Exception as e:
        raise ValueError(f"Failed to resolve stability overlay path '{path_template}': {e}")



################################################################################
## CREATE RGBA RASTER FROM ARRAY MAPPING USING A DF
################################################################################
def create_rgba_from_color_df(array, color_df, alpha=0.7):
    """
    Convert a classified raster array to RGBA using a color lookup dataframe.
    
    Parameters:
    - array: 2D numpy array with integer class values
    - color_df: DataFrame with columns 'pixel_value', 'description', 'color' (hex)
    - alpha: transparency value (0-1)
    
    Returns:
    - 3D numpy array (height, width, 4) with RGBA values (0-255)
    """
    import numpy as np
    
    # Initialize RGBA array (transparent by default)
    rgba = np.zeros((*array.shape, 4), dtype=np.uint8)
    
    # Create lookup for each pixel value
    for _, row in color_df.iterrows():
        pixel_val = row['pixel_value']
        hex_color = row['color']
        
        # Convert hex to RGB
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        # Apply color where array matches pixel value
        mask = (array == pixel_val)
        rgba[mask] = [r, g, b, int(alpha * 255)]
    
    return rgba



################################################################################
## CREATE RGBA RASTER FOR PIXEL STABILITY MAP
################################################################################
def create_stable_pixels_rgba(
    stability_array: np.ndarray,
    modal_class_array: np.ndarray,
    class_map: dict,
    color_df: pd.DataFrame,
    stabilities_to_include: list[int] | None = None,
    alpha_map: dict[int, float] | None = None,
    alpha: float = 1.0,
) -> np.ndarray:
    """
    Create RGBA overlay showing modal class colors only for specified stability values,
    with alpha varying based on stability category.
    
    Parameters
    ----------
    stability_array : np.ndarray
        Array with stability/decision category values (e.g., 1-5)
    modal_class_array : np.ndarray
        Array with modal class IDs
    class_map : dict
        Mapping of class IDs to class names
    color_df : pd.DataFrame
        DataFrame with 'class_name' and 'color' columns
    stabilities_to_include : list[int] | None
        List of stability values to visualize (e.g., [1, 2, 3] for stable pixels)
        If None, all non-zero values are included
    alpha_map : dict[int, float] | None
        Mapping of stability values to alpha values (0.0-1.0)
        e.g., {1: 1.0, 2: 0.8, 3: 0.6}
        Takes precedence over `alpha` parameter
    alpha : float
        Uniform alpha value for all included pixels (default: 1.0)
        Only used if alpha_map is None
    
    Returns
    -------
    np.ndarray
        RGBA array (height, width, 4) with uint8 values
    """
    import numpy as np
    
    # First, convert modal_class_array to RGBA using existing function
    # This respects which classes are in color_df
    rgba_full = process_CLASS_MAP_background(
        raster_path=None,
        array=modal_class_array,
        color_df=color_df,
        class_map=class_map,
        background_alpha_override=1.0,  # Start with full opacity for classes that exist
    )
    
    # Create mask for pixels to include based on stability values
    if stabilities_to_include is not None:
        # Only include specified stability values
        include_mask = np.isin(stability_array, stabilities_to_include)
    else:
        # Include all non-zero stability values
        include_mask = (stability_array > 0)
    
    # Set all pixels NOT in the include list to transparent
    rgba_full[~include_mask, 3] = 0
    
    # Apply alpha values
    if alpha_map is not None:
        # Use alpha_map to set different alpha values per stability category
        for stability_value, alpha_value in alpha_map.items():
            # Only apply alpha to pixels that:
            # 1. Have this stability value
            # 2. Are in the include list
            # 3. Already have color (alpha > 0)
            mask = (
                (stability_array == stability_value) & 
                include_mask & 
                (rgba_full[:, :, 3] > 0)
            )
            rgba_full[mask, 3] = int(alpha_value * 255)
    else:
        # Use constant alpha for all included pixels that have color
        mask = include_mask & (rgba_full[:, :, 3] > 0)
        rgba_full[mask, 3] = int(alpha * 255)
    
    return rgba_full



################################################################################
## PROCESS THE RF ROW STABLE PIXELS OVERLAY
################################################################################
def process_rf_stable_pixels_overlays(
    RF_row_stable_pixels_config: dict | None,
    year_range: list[int],
    window_gdf: gpd.GeoDataFrame,
    location: str,
    hab_selection: str,
    train_split_attempt: str,
    band_selection: str,
    vis_years: str,
) -> dict:
    """
    Process RF stable pixels overlays for multiple years.
    
    Parameters
    ----------
    RF_row_stable_pixels_config : dict | None
        Configuration dict with overlay specifications
        Each overlay_config can include:
        - 'stability_path': path template to stability raster
        - 'modal_class_path': path template to modal class raster (must have CLASS-MAP metadata)
        - 'color_df': DataFrame with 'class_name' and 'color' columns
        - 'stabilities_to_include': list[int], optional - list of stability values to show (e.g., [1, 2, 3])
        - 'alpha_map': dict[int, float], optional - per-stability-value alpha mapping (e.g., {1: 1.0, 2: 0.8, 3: 0.6})
        - 'alpha': float, optional - uniform alpha value for all included pixels (default: 1.0)
        - 'zorder': int, optional - z-order for plotting (default: 2)
    year_range : list[int]
        List of years to process
    window_gdf : gpd.GeoDataFrame
        GeoDataFrame for clipping rasters
    location : str
        Location identifier
    hab_selection : str
        Habitat selection identifier
    train_split_attempt : str
        Training split attempt identifier
    band_selection : str
        Band selection identifier
    vis_years : str
        Visualization years identifier
    
    Returns
    -------
    dict
        Dictionary mapping column names to year-specific overlay configs
        Format: {col_name: {year: [overlay_config_list]}}
        Each overlay_config_list contains dicts with 'rgba_array' and 'zorder'
    """
    import rasterio
    from rasterio.mask import mask
    
    result = {}
    
    if RF_row_stable_pixels_config is None:
        return result
    
    for col_name, overlay_config_list in RF_row_stable_pixels_config.items():
        col_raster_overlays = {}
        
        for year in year_range:
            year_raster_overlay_configs = []
            
            for overlay_config in overlay_config_list:
                # Resolve paths
                try:
                    stability_path = resolve_stability_overlay_path(
                        path_template=overlay_config['stability_path'],
                        location=location,
                        hab_selection=hab_selection,
                        train_split_attempt=train_split_attempt,
                        band_selection=band_selection,
                        vis_years=vis_years,
                        year=year,
                    )
                    
                    modal_class_path = resolve_stability_overlay_path(
                        path_template=overlay_config['modal_class_path'],
                        location=location,
                        hab_selection=hab_selection,
                        train_split_attempt=train_split_attempt,
                        band_selection=band_selection,
                        vis_years=vis_years,
                        year=year,
                    )

                except (KeyError, ValueError) as e:
                    print(f"    Warning: Could not resolve paths for {col_name} [{year}]: {e}")
                    continue
                
                # Check if rasters exist
                if not stability_path.exists():
                    print(f"    Warning: Stability raster not found for {col_name} [{year}]: {stability_path}")
                    continue
                
                if not modal_class_path.exists():
                    print(f"    Warning: Modal class raster not found for {col_name} [{year}]: {modal_class_path}")
                    continue
                
                # Extract class_typology_dict from modal_class raster metadata
                class_typology_dict = extract_class_map(modal_class_path)
                
                if class_typology_dict is None:
                    print(f"    Warning: No CLASS-MAP found in modal_class raster for {col_name} [{year}]")
                    continue
                
                # Clip the two rasters
                try:
                    # 1. Stability raster (decision categories)
                    with rasterio.open(stability_path) as src:
                        stability_array, _ = mask(
                            src,
                            window_gdf.geometry,
                            crop=True,
                            all_touched=True,
                        )
                    stability_array = stability_array[0]
                    
                    # 2. Modal class raster (class IDs)
                    with rasterio.open(modal_class_path) as src:
                        modal_class_array, _ = mask(
                            src,
                            window_gdf.geometry,
                            crop=True,
                            all_touched=True,
                        )
                    modal_class_array = modal_class_array[0]
                    
                    # Create RGBA overlay: show modal class colors only for specified stability pixels
                    # Alpha is already baked into the RGBA array
                    rgba_array = create_stable_pixels_rgba(
                        stability_array=stability_array,
                        modal_class_array=modal_class_array,
                        class_map=class_typology_dict,
                        color_df=overlay_config['color_df'],
                        stabilities_to_include=overlay_config.get('stabilities_to_include', None),
                        alpha_map=overlay_config.get('alpha_map', None),
                        alpha=overlay_config.get('alpha', 1.0),
                    )
                    
                    # Add to overlay configs
                    # Note: No 'alpha' parameter here since alpha is already in the RGBA array
                    year_raster_overlay_configs.append({
                        "rgba_array": rgba_array,
                        "zorder": overlay_config.get("zorder", 2),
                    })
                    
                except Exception as e:
                    print(f"    Warning: Could not process stable pixels overlay for {col_name} [{year}]: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            col_raster_overlays[year] = year_raster_overlay_configs
            
            if year_raster_overlay_configs:
                print(f"    RF row stable pixels overlay ({len(year_raster_overlay_configs)}) added: \"{col_name}\" [{year}]")
        
        result[col_name] = col_raster_overlays
    
    return result



################################################################################
## PROCESS STABILITY OVERLAY
################################################################################
def process_rf_stability_overlays(
    RF_row_stability_config: dict | None,
    year_range: list[int],
    window_gdf: gpd.GeoDataFrame,
    location: str,
    hab_selection: str,
    train_split_attempt: str,
    band_selection: str,
    vis_years: str,
) -> dict:
    """
    Process RF stability overlays for multiple years.
    
    Parameters
    ----------
    RF_row_stability_config : dict | None
        Configuration dict with overlay specifications
        Each overlay_config can include:
        - 'raster_path': path template
        - 'color_df': DataFrame with 'value', 'label', 'color' columns
        - 'classes_to_remove': list[int], optional - class values to exclude from visualization
        - 'alpha_override': float, optional - uniform alpha value for the entire overlay
        - 'alpha_map': dict[int, float], optional - per-pixel-value alpha mapping {pixel_value: alpha}
        - 'zorder': int, optional - z-order for plotting
        
        Note: 'alpha_map' takes precedence over 'alpha_override' if both are specified.
    year_range : list[int]
        List of years to process
    window_gdf : gpd.GeoDataFrame
        GeoDataFrame for clipping rasters
    location : str
        Location identifier
    hab_selection : str
        Habitat selection identifier
    train_split_attempt : str
        Training split attempt identifier
    band_selection : str
        Band selection identifier
    vis_years : str
        Visualization years identifier
    
    Returns
    -------
    dict
        Dictionary mapping column names to year-specific overlay configs
        Format: {col_name: {year: [overlay_config_list]}}
    """
    import rasterio
    from rasterio.mask import mask
    
    result = {}
    
    if RF_row_stability_config is None:
        return result
    
    for col_name, overlay_config_list in RF_row_stability_config.items():
        col_raster_overlays = {}
        
        for year in year_range:
            year_raster_overlay_configs = []
            
            for overlay_config in overlay_config_list:
                # Resolve path
                try:
                    raster_path = resolve_stability_overlay_path(
                        path_template=overlay_config['raster_path'],
                        location=location,
                        hab_selection=hab_selection,
                        train_split_attempt=train_split_attempt,
                        band_selection=band_selection,
                        vis_years=vis_years,
                        year=year,
                    )

                except (KeyError, ValueError) as e:
                    print(f"    Warning: Could not resolve raster overlay path for {col_name} [{year}]: {e}")
                    continue
                
                if not raster_path.exists():
                    print(f"    Warning: Raster overlay not found for {col_name} [{year}]: {raster_path}")
                    continue
                
                # Clip and convert to RGBA
                try:
                    with rasterio.open(raster_path) as src:
                        clipped_array, clipped_transform = mask(
                            src,
                            window_gdf.geometry,
                            crop=True,
                            all_touched=True,
                        )
                    
                    # Filter color_df if classes_to_remove is specified
                    color_df = overlay_config['color_df'].copy()
                    classes_to_remove = overlay_config.get('classes_to_remove', None)
                    
                    if classes_to_remove is not None:
                        color_df = color_df[~color_df['pixel_value'].isin(classes_to_remove)].copy()
                    
                    # Determine alpha strategy
                    alpha_map = overlay_config.get('alpha_map', None)
                    alpha_override = overlay_config.get('alpha_override', 1)
                    
                    if alpha_map is not None:
                        # Use per-pixel-value alpha mapping
                        rgba_array = create_rgba_from_color_df_with_alpha_map(
                            array=clipped_array[0],
                            color_df=color_df,
                            alpha_map=alpha_map,
                        )
                    else:
                        # Use uniform alpha
                        rgba_array = create_rgba_from_color_df(
                            array=clipped_array[0],
                            color_df=color_df,
                            alpha=alpha_override,
                        )
                    
                    year_raster_overlay_configs.append({
                        "rgba_array": rgba_array,
                        "alpha": overlay_config.get("alpha_override", 1),
                        "zorder": overlay_config.get("zorder", 3),
                    })
                    
                except Exception as e:
                    print(f"    Warning: Could not process raster overlay for {col_name} [{year}]: {e}")
                    continue
            
            col_raster_overlays[year] = year_raster_overlay_configs
            
            if year_raster_overlay_configs:
                print(f"    RF row stability overlay ({len(year_raster_overlay_configs)}) added: \"{col_name}\" [{year}]")
        
        result[col_name] = col_raster_overlays
    
    return result


def create_rgba_from_color_df_with_alpha_map(
    array: np.ndarray,
    color_df: pd.DataFrame,
    alpha_map: dict[int, float],
) -> np.ndarray:
    """
    Convert a classified raster array to RGBA using color_df with per-value alpha mapping.
    
    Parameters
    ----------
    array : np.ndarray
        2D array of pixel values
    color_df : pd.DataFrame
        DataFrame with columns: 'pixel_value', 'color'
        'color' should be hex color strings (e.g., '#FF0000')
    alpha_map : dict[int, float]
        Mapping from pixel_value to alpha value (0.0 to 1.0)
        Example: {1: 1.0, 2: 0.7, 3: 0.4}
        Pixel values not in the map will use alpha=1.0 by default
    
    Returns
    -------
    np.ndarray
        RGBA array with shape (height, width, 4) with dtype uint8
    """
    import matplotlib.colors as mcolors
    
    # Standardize column names
    df = color_df.copy()
    
    # Handle both 'value' and 'pixel_value' column names
    if 'pixel_value' in df.columns and 'value' not in df.columns:
        df = df.rename(columns={'pixel_value': 'value'})
    elif 'value' not in df.columns:
        raise ValueError("color_df must have either 'value' or 'pixel_value' column")
    
    # Handle both 'label' and 'description' column names
    if 'description' in df.columns and 'label' not in df.columns:
        df = df.rename(columns={'description': 'label'})

    # Initialize RGBA array
    height, width = array.shape
    rgba_array = np.zeros((height, width, 4), dtype=np.uint8)
    
    # Process each pixel value in color_df
    for _, row in color_df.iterrows():
        pixel_value = row['value']
        color_hex = row['color']
        
        # Get alpha for this pixel value (default to 1.0 if not in map)
        alpha = alpha_map.get(pixel_value, 1.0)
        
        # Convert hex color to RGB
        rgb = mcolors.to_rgb(color_hex)
        
        # Create mask for this pixel value
        mask = (array == pixel_value)
        
        # Assign RGBA values
        rgba_array[mask, 0] = int(rgb[0] * 255)  # R
        rgba_array[mask, 1] = int(rgb[1] * 255)  # G
        rgba_array[mask, 2] = int(rgb[2] * 255)  # B
        rgba_array[mask, 3] = int(alpha * 255)   # A
    
    return rgba_array



################################################################################
## CREATE UNSTABLE PIXELS OVERLAY
################################################################################
def process_rf_unstable_pixels_overlays(
    RF_row_unstable_pixels_config: dict | None,
    year_range: list[int],
    window_gdf: gpd.GeoDataFrame,
    location: str,
    hab_selection: str,
    train_split_attempt: str,
    band_selection: str,
    vis_years: str,
) -> dict:
    """
    Process RF unstable pixels overlays for multiple years.
    
    Parameters
    ----------
    RF_row_unstable_pixels_config : dict | None
        Configuration dict with overlay specifications
        Each overlay_config can include:
        - 'unstable_pixels_path': path template to unstable pixels raster (RGB raster)
        - 'alpha': float, optional - uniform alpha value (default: 1.0)
        - 'zorder': int, optional - z-order for plotting (default: 2)
        - 'include_legend_keys': list[str], optional - legend keys to include (default: all)
        - 'exclude_legend_keys': list[str], optional - legend keys to exclude (default: none)
    year_range : list[int]
        List of years to process
    window_gdf : gpd.GeoDataFrame
        GeoDataFrame for clipping rasters
    location : str
        Location identifier
    hab_selection : str
        Habitat selection identifier
    train_split_attempt : str
        Training split attempt identifier
    band_selection : str
        Band selection identifier
    vis_years : str
        Visualization years identifier
    
    Returns
    -------
    dict
        Dictionary mapping column names to year-specific overlay configs
        Format: {col_name: {year: [overlay_config_list]}}
        Each overlay_config_list contains dicts with 'rgba_array' and 'zorder'
    """
    import rasterio
    from rasterio.mask import mask
    import json
    
    result = {}
    
    if RF_row_unstable_pixels_config is None:
        return result
    
    for col_name, overlay_config_list in RF_row_unstable_pixels_config.items():
        col_raster_overlays = {}
        
        for year in year_range:
            year_raster_overlay_configs = []
            
            for overlay_config in overlay_config_list:
                # Resolve path
                try:
                    unstable_pixels_path = resolve_stability_overlay_path(
                        path_template=overlay_config['unstable_pixels_path'],
                        location=location,
                        hab_selection=hab_selection,
                        train_split_attempt=train_split_attempt,
                        band_selection=band_selection,
                        vis_years=vis_years,
                        year=year,
                    )
                    
                except (KeyError, ValueError) as e:
                    print(f"    Warning: Could not resolve path for {col_name} [{year}]: {e}")
                    continue
                
                # Check if raster exists
                if not unstable_pixels_path.exists():
                    print(f"    Warning: Unstable pixels raster not found for {col_name} [{year}]: {unstable_pixels_path}")
                    continue
                
                # Extract LEGEND from raster metadata and clip RGB bands
                try:
                    with rasterio.open(unstable_pixels_path) as src:
                        tags = src.tags()
                        
                        if 'LEGEND' not in tags:
                            print(f"    Warning: No LEGEND found in unstable_pixels raster for {col_name} [{year}]")
                            continue
                        
                        # Parse LEGEND JSON
                        legend_dict = json.loads(tags['LEGEND'])
                        
                        # Clip the RGB raster (3 bands)
                        rgb_array, _ = mask(
                            src,
                            window_gdf.geometry,
                            crop=True,
                            all_touched=True,
                        )
                        # rgb_array shape: (3, height, width) - bands are [R, G, B]
                    
                    # Create RGBA overlay from RGB raster
                    rgba_array = create_unstable_pixels_rgba_from_rgb(
                        rgb_array=rgb_array,
                        legend_dict=legend_dict,
                        alpha=overlay_config.get('alpha', 1.0),
                        include_legend_keys=overlay_config.get('include_legend_keys', None),
                        exclude_legend_keys=overlay_config.get('exclude_legend_keys', None),
                    )
                    
                    # Add to overlay configs
                    year_raster_overlay_configs.append({
                        "rgba_array": rgba_array,
                        "zorder": overlay_config.get("zorder", 2),
                    })
                    
                except json.JSONDecodeError as e:
                    print(f"    Warning: Could not parse LEGEND for {col_name} [{year}]: {e}")
                    continue
                except Exception as e:
                    print(f"    Warning: Could not process unstable pixels overlay for {col_name} [{year}]: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            col_raster_overlays[year] = year_raster_overlay_configs
            
            if year_raster_overlay_configs:
                print(f"    RF row unstable pixels overlay ({len(year_raster_overlay_configs)}) added: \"{col_name}\" [{year}]")
        
        result[col_name] = col_raster_overlays
    
    return result


def create_unstable_pixels_rgba_from_rgb(
    rgb_array: np.ndarray,
    legend_dict: dict,
    alpha: float = 1.0,
    include_legend_keys: list[str] | None = None,
    exclude_legend_keys: list[str] | None = None,
) -> np.ndarray:
    """
    Create RGBA overlay from RGB unstable pixels raster.
    
    The raster is already RGB-encoded (3 bands). We just need to add alpha channel
    and set transparency for black pixels (background) and optionally filter by legend keys.
    
    Parameters
    ----------
    rgb_array : np.ndarray
        RGB array with shape (3, height, width) - bands are [R, G, B]
    legend_dict : dict
        LEGEND dictionary from raster metadata
        Format: {"1_2": {"classes": "...", "color": "#...", "rgb": [R, G, B]}, ...}
    alpha : float
        Uniform alpha value for all non-black pixels (0.0-1.0)
    include_legend_keys : list[str], optional
        If provided, only pixels matching these legend keys will be shown.
        Example: ["1_2", "2_3"] to show only those two categories
    exclude_legend_keys : list[str], optional
        If provided, pixels matching these legend keys will be made transparent.
        Ignored if include_legend_keys is specified.
    
    Returns
    -------
    np.ndarray
        RGBA array (height, width, 4) with uint8 values
    """
    import numpy as np
    
    # Transpose from (3, height, width) to (height, width, 3)
    rgb_transposed = np.transpose(rgb_array, (1, 2, 0))
    
    height, width, _ = rgb_transposed.shape
    
    # Create RGBA array
    rgba_array = np.zeros((height, width, 4), dtype=np.uint8)
    
    # Copy RGB channels
    rgba_array[:, :, 0:3] = rgb_transposed
    
    # Determine which pixels should be visible
    if include_legend_keys is not None:
        # Only show pixels matching the included legend keys
        visible_mask = np.zeros((height, width), dtype=bool)
        
        for key in include_legend_keys:
            if key in legend_dict:
                rgb_value = legend_dict[key]['rgb']
                # Find pixels matching this RGB value
                pixel_match = (
                    (rgb_transposed[:, :, 0] == rgb_value[0]) &
                    (rgb_transposed[:, :, 1] == rgb_value[1]) &
                    (rgb_transposed[:, :, 2] == rgb_value[2])
                )
                visible_mask |= pixel_match
        
        # Set alpha for visible pixels
        rgba_array[visible_mask, 3] = int(alpha * 255)
        
    elif exclude_legend_keys is not None:
        # Show all non-black pixels except those in exclude list
        non_black_mask = (
            (rgb_transposed[:, :, 0] > 0) | 
            (rgb_transposed[:, :, 1] > 0) | 
            (rgb_transposed[:, :, 2] > 0)
        )
        
        # Create mask for excluded pixels
        excluded_mask = np.zeros((height, width), dtype=bool)
        
        for key in exclude_legend_keys:
            if key in legend_dict:
                rgb_value = legend_dict[key]['rgb']
                # Find pixels matching this RGB value
                pixel_match = (
                    (rgb_transposed[:, :, 0] == rgb_value[0]) &
                    (rgb_transposed[:, :, 1] == rgb_value[1]) &
                    (rgb_transposed[:, :, 2] == rgb_value[2])
                )
                excluded_mask |= pixel_match
        
        # Set alpha for visible (non-black and non-excluded) pixels
        visible_mask = non_black_mask & ~excluded_mask
        rgba_array[visible_mask, 3] = int(alpha * 255)
        
    else:
        # Default behavior: show all non-black pixels
        non_black_mask = (
            (rgb_transposed[:, :, 0] > 0) | 
            (rgb_transposed[:, :, 1] > 0) | 
            (rgb_transposed[:, :, 2] > 0)
        )
        rgba_array[non_black_mask, 3] = int(alpha * 255)
    
    return rgba_array




################################################################################
## GET THE CLASS BANDS
################################################################################
import json

def extract_class_map(raster_path: str | Path) -> dict | None:
    """
    Extract CLASS-MAP from raster metadata.
    
    Parameters
    ----------
    raster_path : str | Path
        Path to the raster file
    
    Returns
    -------
    dict | None
        Dictionary mapping integer values to class names,
        or None if CLASS-MAP not found.
        Example: {0: "Open water", 1: "Remaining", 2: "Semi-Wet Nature"}
    """
    import rasterio
    
    with rasterio.open(raster_path) as src:
        # Get tags/metadata
        tags = src.tags()
        
        if 'CLASS-MAP' in tags:
            try:
                # Parse JSON string
                class_map_str = tags['CLASS-MAP']
                class_map_dict = json.loads(class_map_str)
                
                # Convert string keys to integers
                class_map = {int(k): v for k, v in class_map_dict.items()}
                
                return class_map
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Warning: Could not parse CLASS-MAP: {e}")
                return None
        
        return None



################################################################################
## RGB ARRAY PROCESSING
################################################################################
def process_rgb_array(
    array: np.ndarray,
    background_alpha_override: float = 1.0,
) -> np.ndarray:
    """
    Process RGB array to RGBA with configurable transparency.
    Memory-efficient version using float32.
    
    Parameters
    ----------
    array : np.ndarray
        Either (3, H, W) or (H, W, 3) with values typically 0-255 or 0-1
    background_alpha : float, default 1.0
        Alpha (transparency) value for the background, range [0, 1].
        0 = fully transparent, 1 = fully opaque.
    
    Returns
    -------
    np.ndarray
        RGBA array (H, W, 4) with values in [0, 1] as float32
    """
    # Ensure shape is (H, W, 3)
    if array.ndim == 3 and array.shape[0] == 3:
        # Convert from (3, H, W) to (H, W, 3)
        array = np.transpose(array, (1, 2, 0))
    
    # Normalize to [0, 1] using float32 (saves 50% memory vs float64)
    if array.max() > 1.0:
        array = array.astype(np.float32) / 255.0
    else:
        # Already in [0, 1], but ensure float32
        array = array.astype(np.float32)
    
    # Clip to valid range
    array = np.clip(array, 0, 1)
    
    # Add alpha channel
    height, width = array.shape[:2]
    rgba = np.dstack([
        array,
        np.full((height, width), background_alpha_override, dtype=np.float32)
    ])
      
    return rgba


################################################################################
## CLASSIFICATION BACKGROUND PROCESSING
################################################################################
def process_CLASS_MAP_background(
    raster_path: str | Path,
    array: np.ndarray,
    color_df: pd.DataFrame,
    class_map: dict | None = None,
    background_alpha_override: float | None = None,
) -> np.ndarray:
    """
    Process classification map to RGBA.
    
    Parameters
    ----------
    raster_path : str | Path
        Path to the raster file (used to extract class_map if needed)
    array : np.ndarray
        2D array of class indices, shape (H, W)
    color_df : pd.DataFrame
        DataFrame with columns: 'type', 'color' (hex), optionally 'alpha'
    class_map : dict | None, optional
        Dictionary mapping integer values to class names.
        If None, will attempt to extract from raster metadata.
        Example: {0: "Open water", 1: "Forest", 2: "Urban"}
    background_alpha_override : float | None, optional
        Override alpha value for all pixels (0-1)
    
    Returns
    -------
    np.ndarray
        RGBA array, shape (H, W, 4), dtype uint8
    """
    
    # Extract class_map from raster if not provided
    if class_map is None:
        class_map = extract_class_map(raster_path)
        
        if class_map is None:
            raise ValueError(
                f"CLASS-MAP colorscheme requires class_map, but none found in "
                f"config or raster metadata for {raster_path}"
            )
    
    H, W = array.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    
    # Create lookup from class_name to color info
    color_lookup = {}
    for _, row in color_df.iterrows():
        class_name = row['type']
        hex_color = row['color']
        alpha = row.get('alpha', 1.0) if background_alpha_override is None else background_alpha_override
        
        # Convert hex to RGB
        rgb = hex_to_rgb(hex_color)
        rgba_val = (*[int(c * 255) for c in rgb], int(alpha * 255))
        color_lookup[class_name] = rgba_val
    
    # Map pixel values to colors
    for pixel_value, class_name in class_map.items():
        if class_name in color_lookup:
            mask = (array == pixel_value)
            rgba[mask] = color_lookup[class_name]
    
    return rgba



################################################################################
## OVERRIDING THE ALPHA CHANNEL WHEN SPECIFIED
################################################################################
def override_alpha_channel(
    rgba: np.ndarray,
    background_alpha_override: float,
) -> np.ndarray:
    """
    Replace the alpha channel of an RGBA array with a fixed value.
    
    Parameters
    ----------
    rgba : np.ndarray
        RGBA array of shape (H, W, 4)
    background_alpha_override : float
        New alpha value in range [0, 1]
    
    Returns
    -------
    np.ndarray
        RGBA array with updated alpha channel
    """
    if rgba.shape[-1] != 4:
        raise ValueError(f"Expected RGBA array (H, W, 4), got shape {rgba.shape}")
    
    # Create a copy to avoid modifying the original
    rgba_copy = rgba.copy()
    rgba_copy[..., 3] = background_alpha_override
    
    print(f"    Alpha channel overridden: {background_alpha_override}")
    
    return rgba_copy



################################################################################
## FORMATTING OF THE BACKGROUND
################################################################################
def process_background(
    raster_path: Path,
    array: np.ndarray,
    colorscheme: str,
    background_alpha_override: float | None = None,
    color_df: pd.DataFrame | None = None,
    class_map: dict | None = None,
) -> np.ndarray:
    """
    Process a background array and return RGBA in range [0, 1].
    
    Parameters
    ----------
    raster_path : Path
        Path to the raster file (used for error messages)
    array : np.ndarray
        Raw array data from the raster
    colorscheme : str
        One of: 'rgb', 'rgba', 'CLASS-MAP'
    background_alpha_override : float | None, default None
        If provided, override the alpha channel with this value.
        If None and colorscheme='rgba', preserve original alpha values.
    color_df : pd.DataFrame | None
        Required for CLASS-MAP colorscheme
    class_map : dict | None
        Required for CLASS-MAP colorscheme
    
    Returns
    -------
    np.ndarray
        RGBA array (H, W, 4) with values in [0, 1]
    """
    
    if colorscheme == 'CLASS-MAP':
        if color_df is None:
            raise ValueError("color_df required for CLASS-MAP colorscheme")
        
        alpha = background_alpha_override if background_alpha_override is not None else 1.0
        
        return process_CLASS_MAP_background(
            raster_path=raster_path,
            array=array,
            color_df=color_df,
            class_map=class_map,
            background_alpha_override=alpha,
        )
    
    elif colorscheme == 'rgb':
        alpha = background_alpha_override if background_alpha_override is not None else 1.0
        
        return process_rgb_array(
            array=array,
            background_alpha_override=alpha,
        )
    
    elif colorscheme == 'rgba':
        if array.ndim == 3 and array.shape[0] == 4:
            array = np.transpose(array, (1, 2, 0))
        
        if array.shape[-1] != 4:
            raise ValueError(
                f"colorscheme='rgba' expects 4 bands, got shape {array.shape}"
            )
        
        if array.max() > 1.0:
            rgba = array.astype(np.float32) / 255.0
        else:
            rgba = array.astype(np.float32)
        
        rgba = np.clip(rgba, 0, 1)
        
        # Override alpha if requested, otherwise keep original
        if background_alpha_override is not None:
            rgba = override_alpha_channel(rgba, background_alpha_override)
        
        return rgba
    
    else:
        raise ValueError(
            f"Unknown colorscheme: '{colorscheme}'. "
            f"Must be one of: 'rgb', 'rgba', 'CLASS-MAP'"
        )



################################################################################
## GET SPECIFIC YEAR FROM RASTER STACK
################################################################################
def get_band_mapping_for_raster(
    raster_path: str | Path,
    colorscheme: str,
    band_mapping: dict[int, int | tuple] | str = "auto",
    warn_missing_years: bool = True,
) -> dict[int, int | tuple]:
    """
    Get year-to-band mapping for a multi-band raster.
    
    Parameters
    ----------
    raster_path : str | Path
        Path to the multi-band raster.
    colorscheme : str
        Color scheme defining expected bands per year:
        - "rgb": 3 bands per year (Red, Green, Blue)
        - "rgba": 4 bands per year (Red, Green, Blue, Alpha)
        - "CLASS-MAP": 1 band per year (classification map)
    band_mapping : dict[int, int | tuple] | str, default "auto"
        Either:
        - "auto": Automatically extract from band descriptions (recommended)
        - dict: Manual mapping of year -> band index or tuple of indices
          Must comply with colorscheme band requirements.
    warn_missing_years : bool, default True
        If True, warn when years are missing in the raster stack.
    
    Returns
    -------
    dict[int, int | tuple]
        Mapping of year -> band index (1-indexed).
        - If colorscheme="CLASS-MAP": {2017: 1, 2018: 2, ...}
        - If colorscheme="rgb": {2017: (1, 2, 3), 2018: (4, 5, 6), ...}
        - If colorscheme="rgba": {2017: (1, 2, 3, 4), 2018: (5, 6, 7, 8), ...}
    
    Raises
    ------
    ValueError
        If colorscheme is invalid or band mapping doesn't comply with colorscheme.
    FileNotFoundError
        If raster_path does not exist.
    """
    import re
    import rasterio
    from pathlib import Path
    
    raster_path = Path(raster_path)
    
    if not raster_path.exists():
        raise FileNotFoundError(f"Raster not found: {raster_path}")
    
    # Validate and determine expected bands per year
    VALID_COLORSCHEMES = {
        "CLASS-MAP": 1,
        "rgb": 3,
        "rgba": 4,
    }
    
    if colorscheme not in VALID_COLORSCHEMES:
        raise ValueError(
            f"Invalid colorscheme: '{colorscheme}'. "
            f"Must be one of: {list(VALID_COLORSCHEMES.keys())}"
        )
    
    bands_per_year = VALID_COLORSCHEMES[colorscheme]
    
    # Manual mapping provided - validate compliance
    if isinstance(band_mapping, dict):
        _validate_band_mapping_compliance(band_mapping, bands_per_year, colorscheme, raster_path)
        
        if warn_missing_years:
            _validate_year_sequence(band_mapping.keys(), raster_path)
        
        return band_mapping
    
    # Automatic extraction
    if band_mapping == "auto":
        with rasterio.open(raster_path) as src:
            # Single band per year (CLASS-MAP)
            if bands_per_year == 1:
                year_to_band = {}
                
                for band_idx in range(1, src.count + 1):
                    desc = src.descriptions[band_idx - 1]
                    
                    if desc is None:
                        continue
                    
                    match = re.search(r'(20\d{2})', desc)
                    if match:
                        year = int(match.group(1))
                        year_to_band[year] = band_idx
                
                if not year_to_band:
                    raise ValueError(
                        f"No years could be extracted from band descriptions in {raster_path}.\n"
                        f"Expected format: 'Band N: YYYY' (e.g., 'Band 1: 2017').\n"
                        f"Use band_mapping=dict to provide manual mapping."
                    )
                
                sorted_mapping = dict(sorted(year_to_band.items()))
                
                if warn_missing_years:
                    _validate_year_sequence(sorted_mapping.keys(), raster_path)
                
                return sorted_mapping
            
            # Multiple bands per year (rgb, rgba)
            else:
                from collections import defaultdict
                
                year_groups = defaultdict(list)
                
                # Extract years from each band
                for band_idx in range(1, src.count + 1):
                    desc = src.descriptions[band_idx - 1]
                    
                    if desc is None:
                        continue
                    
                    match = re.search(r'(20\d{2})', desc)
                    if match:
                        year = int(match.group(1))
                        year_groups[year].append(band_idx)
                
                # Validate and create mapping
                if year_groups:
                    year_to_bands = {}
                    incomplete_years = []
                    
                    for year, bands in sorted(year_groups.items()):
                        if len(bands) == bands_per_year:
                            year_to_bands[year] = tuple(sorted(bands))
                        else:
                            incomplete_years.append(
                                f"{year}: {len(bands)}/{bands_per_year} bands"
                            )
                    
                    if incomplete_years:
                        print(
                            f"  Warning: [{raster_path.name}] Incomplete years skipped (colorscheme='{colorscheme}'):\n"
                            + "    " + "\n    ".join(incomplete_years)
                        )
                    
                    if year_to_bands:
                        if warn_missing_years:
                            _validate_year_sequence(year_to_bands.keys(), raster_path)
                        return year_to_bands
                
                # Try extracting from filename as fallback
                filename = raster_path.stem
                year_range_match = re.search(r'(20\d{2})[-_](20\d{2})', filename)
                
                if year_range_match:
                    start_year = int(year_range_match.group(1))
                    end_year = int(year_range_match.group(2))
                    years = list(range(start_year, end_year + 1))
                    
                    expected_years = src.count // bands_per_year
                    if len(years) != expected_years:
                        raise ValueError(
                            f"Inferred {len(years)} years from filename, "
                            f"but expected {expected_years} based on band count "
                            f"(colorscheme='{colorscheme}' requires {bands_per_year} bands/year)."
                        )
                    
                    year_to_bands = {}
                    for year_idx, year in enumerate(years):
                        start_band = year_idx * bands_per_year + 1
                        band_tuple = tuple(range(start_band, start_band + bands_per_year))
                        year_to_bands[year] = band_tuple
                    
                    return year_to_bands
                
                raise ValueError(
                    f"Could not extract years from band descriptions or filename in {raster_path}.\n"
                    f"Colorscheme '{colorscheme}' expects {bands_per_year} bands per year.\n"
                    f"Use band_mapping=dict to provide manual mapping."
                )
    
    raise ValueError(
        f"band_mapping must be 'auto' or dict, got {type(band_mapping).__name__}"
    )


def _validate_band_mapping_compliance(
    band_mapping: dict,
    expected_bands_per_year: int,
    colorscheme: str,
    raster_path: Path,
) -> None:
    """Validate that manual band mapping complies with colorscheme requirements."""
    for year, band_indices in band_mapping.items():
        # Single band expected
        if expected_bands_per_year == 1:
            if not isinstance(band_indices, int):
                raise ValueError(
                    f"Colorscheme '{colorscheme}' expects single band per year, "
                    f"but year {year} has: {band_indices} (type: {type(band_indices).__name__})"
                )
        
        # Multiple bands expected
        else:
            if not isinstance(band_indices, (tuple, list)):
                raise ValueError(
                    f"Colorscheme '{colorscheme}' expects {expected_bands_per_year} bands per year, "
                    f"but year {year} has single band: {band_indices}"
                )
            
            if len(band_indices) != expected_bands_per_year:
                raise ValueError(
                    f"Colorscheme '{colorscheme}' expects {expected_bands_per_year} bands per year, "
                    f"but year {year} has {len(band_indices)} bands: {band_indices}"
                )


def _validate_year_sequence(years: list[int], raster_path: Path) -> None:
    """Warn if years are not consecutive in the raster stack."""
    sorted_years = sorted(years)
    
    if len(sorted_years) < 2:
        return
    
    missing_years = []
    for i in range(len(sorted_years) - 1):
        current = sorted_years[i]
        next_year = sorted_years[i + 1]
        
        if next_year - current > 1:
            missing = list(range(current + 1, next_year))
            missing_years.extend(missing)
    
    if missing_years:
        print(
            f"Warning: [{raster_path.name}] Missing years in stack: {missing_years}\n"
            f"    Found years: {sorted_years}"
        )



################################################################################
## MERGE ILLUSTRATION SETTINGS
################################################################################
def merge_illustration_settings(
    global_settings: dict,
    column_settings: dict,
) -> dict:
    """
    Merge global and column-specific illustration settings.
    Column settings override global settings.
    
    Parameters
    ----------
    global_settings : dict
        Global settings for all illustrations
    column_settings : dict
        Column-specific settings that override global
    
    Returns
    -------
    dict
        Merged settings
    """
    merged = {}
    
    # Get all illustration types from both dicts
    all_types = set(global_settings.keys()) | set(column_settings.keys())
    
    for illustration_type in all_types:
        global_ill_settings = global_settings.get(illustration_type, {})
        column_ill_settings = column_settings.get(illustration_type, {})
        
        # Merge: column settings override global
        merged[illustration_type] = {
            **global_ill_settings,
            **column_ill_settings,
        }
    
    return merged



################################################################################
## PRECLIPPING OF ALL DATA PER WINDOW
################################################################################
def prepare_location_data(
    location: str | None,
    year_range: list[int],
    window_gdf: gpd.GeoDataFrame,
    background_config: dict[str, dict],
    gdf_overlay_config: dict[str, list[dict]],
    RF_row_stability_config: dict[str, list[dict]] | None = None,
    RF_row_stable_pixels_config: dict[str, list[dict]] | None = None,
    RF_row_unstable_pixels_config: dict[str, list[dict]] | None = None,
    hab_selection: str | None = None,
    train_split_attempt: str | None = None,  
    band_selection: str | None = None,  
    vis_years: list[int] | None = None,
    aggregate_years: bool = False,
    background_year: int | None = None,
) -> dict:
    """
    Prepare location data by clipping backgrounds and overlays to window.
    
    This function loads raster backgrounds, raster overlays, and vector overlays 
    for a specific location window, clipping all data to the window bounds.
    
    Parameters
    ----------
    location : str
        Location name (used to resolve raster paths)
    year_range : list[int]
        Years to process
    window_gdf : gpd.GeoDataFrame
        Window geometry to clip data to
    background_config : dict[str, dict]
        Background configuration for each column:
        {
            "column_name": {
                "raster_path": Path or str,
                "colorscheme": "rgb" | "CLASS-MAP",
                "band_mapping": dict or "auto",
                "background_alpha_override": float | None,
                "color_df": pd.DataFrame (if CLASS-MAP),
                "class_map": dict (if CLASS-MAP),
            }
        }
    gdf_overlay_config : dict[str, list[dict]]
        Vector overlays for each column with visualization settings:
        {
            "column_name": [
                {
                    "gdf": gpd.GeoDataFrame,
                    "alpha": float (optional, default 0.7),
                    "zorder": int (optional, default 2),
                    "edgecolor": str (optional),
                    "facecolor": str (optional),
                    "linewidth": float (optional, default 1.0),
                },
                ...
            ]
        }
    RF_row_stability_config : dict[str, list[dict]] | None
        Raster overlays for each column (list of configs like gdf_overlay_config):
        {
            "column_name": [
                {
                    "raster_path": Path or str,
                    "colorscheme": "rgb" | "CLASS-MAP",
                    "band_mapping": dict or "auto",
                    "alpha": float (optional, default 0.7),
                    "zorder": int (optional, default 3),
                    "color_df": pd.DataFrame (optional, if CLASS-MAP),
                    "class_map": dict (optional, if CLASS-MAP),
                },
                ...
            ]
        }
        Default is None (no raster overlays).
    aggregate_years : bool
        If True, combine all years into a single overlay (useful for header rows).
        Vector overlays will be stored under year_range[0] with all years merged.
    background_year : int | None
        If provided, use ONLY this year for backgrounds (ignores year_range for backgrounds).
        If None, uses year_range (or first year if aggregate_years=True).


    
    Returns
    -------
    dict
        {
            'backgrounds': {col_name: {year: rgba_array}},
            'raster_overlays': {col_name: {year: [rgba_array_config_dict, ...]}},
            'vector_overlays': {col_name: {year: [overlay_config_dict, ...]}},
            'transform': affine.Affine,
        }
    
    Notes
    -----
    The returned 'vector_overlays' structure preserves the overlay settings
    from the input config, with clipped GeoDataFrames replacing the originals.
    
    Raster overlays are processed similarly to backgrounds but:
    - Stored as a list per column/year (like vector overlays)
    - Can have multiple raster overlays per column
    - Each overlay includes alpha and zorder for layering
    
    Examples
    --------
    >>> window_gdf = locations_gdf[locations_gdf['location'] == 'Amsterdam']
    >>> prepared = prepare_location_data(
    ...     location='Amsterdam',
    ...     year_range=[2020, 2021, 2022],
    ...     window_gdf=window_gdf,
    ...     background_config=background_config,
    ...     gdf_overlay_config=gdf_overlay_config,
    ...     raster_overlay_config=raster_overlay_config,
    ... )
    >>> prepared['backgrounds']['RF_Q1'][2020]  # RGBA array
    >>> prepared['raster_overlays']['RF_Q1'][2020]  # List of raster overlay configs
    >>> prepared['vector_overlays']['Training Data'][2020]  # List of vector overlay configs
    """
    import rasterio
    from rasterio.mask import mask
    from pathlib import Path
    
    result = {
        'backgrounds': {},
        'raster_overlays': {},
        'vector_overlays': {},
        'transform': None,
    }
    
    # ========================================================================
    # 1. PREPARE BACKGROUNDS
    # ========================================================================
    
    for col_name, bg_config in background_config.items():
        # Find the background-raster path
        raster_path = resolve_background_path(
            bg_config['raster_path'],
            location=location,
        )
        
        if not raster_path.exists():
            print(f"    Warning: Raster not found for {col_name}: {raster_path}")
            result['backgrounds'][col_name] = {}
            continue
        
        if 'colorscheme' not in bg_config:
            print(f"    Warning: Missing 'colorscheme' for {col_name}. Skipping.")
            result['backgrounds'][col_name] = {}
            continue
        
        colorscheme = bg_config['colorscheme']
        
        # Get the raster bands for the background
        try:
            band_mapping = get_band_mapping_for_raster(
                raster_path,
                colorscheme=colorscheme,
                band_mapping=bg_config.get('band_mapping', 'auto'),
            )
        except Exception as e:
            print(f"    Warning: Could not get band mapping for {col_name}: {e}")
            result['backgrounds'][col_name] = {}
            continue
        
        if background_year is not None:
            # Use explicitly provided background year
            if background_year not in band_mapping:
                print(
                    f"    Warning: background_year {background_year} not available "
                    f"in band_mapping {list(band_mapping.keys())} for {col_name}. Skipping."
                )
                result['backgrounds'][col_name] = {}
                continue
            years_to_process = [background_year]
        else:
            # Process all available years
            requested_years = [y for y in year_range if y in band_mapping]
            
            if not requested_years:
                print(
                    f"    Warning: No overlap between requested years {year_range} "
                    f"and available years {list(band_mapping.keys())} for {col_name}. Skipping."
                )
                result['backgrounds'][col_name] = {}
                continue
            
            years_to_process = requested_years
        
        try:
            with rasterio.open(raster_path) as src:
                clipped_array, clipped_transform = mask(
                    src,
                    window_gdf.geometry,
                    crop=True,
                    all_touched=True,
                )
                
                if result['transform'] is None:
                    result['transform'] = clipped_transform
        
        except Exception as e:
            print(f"    Warning: Could not clip raster for {col_name}: {e}")
            result['backgrounds'][col_name] = {}
            continue
        
        color_df = bg_config.get('color_df', None)
        class_map = bg_config.get('class_map', None)
        background_alpha_override = bg_config.get('background_alpha_override', None)
        
        col_backgrounds = {}
        
        for year in years_to_process:
            band_indices = band_mapping[year]
            
            # Extract band(s) for this year
            if isinstance(band_indices, int):
                raw_array = clipped_array[band_indices - 1]
            else:
                band_arrays = [clipped_array[b - 1] for b in band_indices]
                raw_array = np.stack(band_arrays, axis=0)
            
            try:
                processed_rgba = process_background(
                    raster_path=raster_path,
                    array=raw_array,
                    colorscheme=colorscheme,
                    color_df=color_df,
                    class_map=class_map,
                    background_alpha_override=background_alpha_override,
                )
                col_backgrounds[year] = processed_rgba
                print(f"    Background added: \"{col_name}\" [{year}]")
            
            except Exception as e:
                print(f"    Warning: Could not process {col_name} for year {year}: {e}")
                continue
        
        result['backgrounds'][col_name] = col_backgrounds
      
    # ========================================================================
    # 2. RF_row_stability overlay
    # ========================================================================

    rf_stability_overlays = process_rf_stability_overlays(
        RF_row_stability_config=RF_row_stability_config,
        year_range=year_range,
        window_gdf=window_gdf,
        location=location,
        hab_selection=hab_selection,
        train_split_attempt=train_split_attempt,
        band_selection=band_selection,
        vis_years=vis_years,
    )

    # Merge into result
    result['raster_overlays'].update(rf_stability_overlays)

    # ========================================================================
    # 3. RF_stable_pixels overlay (stability + modal class)
    # ========================================================================

    rf_stable_pixels_overlays = process_rf_stable_pixels_overlays(
        RF_row_stable_pixels_config=RF_row_stable_pixels_config,
        year_range=year_range,
        window_gdf=window_gdf,
        location=location,
        hab_selection=hab_selection,
        train_split_attempt=train_split_attempt,
        band_selection=band_selection,
        vis_years=vis_years,
    )

    # Merge into result
    result['raster_overlays'].update(rf_stable_pixels_overlays)


    # ========================================================================
    # 4. RF_unstable_pixels overlay (stability + modal class)
    # ========================================================================

    unstable_overlays = process_rf_unstable_pixels_overlays(
        RF_row_unstable_pixels_config=RF_row_unstable_pixels_config,
        year_range=year_range,
        window_gdf=window_gdf,
        location=location,
        hab_selection=hab_selection,
        train_split_attempt=train_split_attempt,
        band_selection=band_selection,
        vis_years=vis_years,
    )

    # Merge into result
    result['raster_overlays'].update(unstable_overlays)


    # ========================================================================
    # 5. PREPARE VECTOR OVERLAYS
    # ========================================================================

    for col_name, overlay_config_list in gdf_overlay_config.items():
        col_overlays = {}
        
        # Combine all years into one overlay if aggregate_years=True (for header rows)
        if aggregate_years:
            aggregated_overlay_configs = []
            
            for overlay_config in overlay_config_list:
                gdf = overlay_config.get("gdf")
                
                if gdf is None or gdf.empty:
                    continue
                
                # Reproject if needed
                if gdf.crs != window_gdf.crs:
                    gdf = gdf.to_crs(window_gdf.crs)
                
                # Filter by year range if 'years' column exists
                if 'years' in gdf.columns:
                    gdf_filtered = gdf[gdf['years'].isin(year_range)].copy()
                else:
                    gdf_filtered = gdf.copy()
                
                if gdf_filtered.empty:
                    continue
                
                # Clip to window
                clipped_gdf = gdf_filtered.clip(window_gdf)
                
                if clipped_gdf.empty:
                    continue
                
                # Create overlay config with clipped GDF
                clipped_overlay_config = {
                    "gdf": clipped_gdf,
                    "alpha": overlay_config.get("alpha", 0.7),
                    "zorder": overlay_config.get("zorder", 2),
                }
                
                # Copy styling parameters (config overrides GDF columns)
                for param in ["edgecolor", "facecolor", "linewidth", "linestyle"]:
                    if param in overlay_config:
                        clipped_overlay_config[param] = overlay_config[param]
                    elif param in clipped_gdf.columns and not clipped_gdf.empty:
                        unique_values = clipped_gdf[param].unique()
                        if len(unique_values) == 1:
                            clipped_overlay_config[param] = unique_values[0]

                if "label" in overlay_config:
                    clipped_overlay_config["label"] = overlay_config["label"]
                
                aggregated_overlay_configs.append(clipped_overlay_config)
            
            # Store aggregated overlays under first year
            col_overlays[year_range[0]] = aggregated_overlay_configs
            
            if aggregated_overlay_configs:
                print(
                    f"    Vector overlays ({len(aggregated_overlay_configs)}) aggregated: "
                    f"\"{col_name}\" [all years: {year_range[0]}-{year_range[-1]}]"
                )
        
        else:
            for year in year_range:
                year_overlay_configs = []
                
                for overlay_config in overlay_config_list:
                    gdf = overlay_config.get("gdf")
                    
                    if gdf is None or gdf.empty:
                        continue
                    
                    # Reproject if needed
                    if gdf.crs != window_gdf.crs:
                        gdf = gdf.to_crs(window_gdf.crs)
                    
                    # Filter by year if 'years' column exists
                    if 'years' in gdf.columns:
                        gdf_year = gdf[gdf['years'] == year].copy()
                    else:
                        gdf_year = gdf.copy()
                    
                    if gdf_year.empty:
                        continue
                    
                    # Clip to window
                    clipped_gdf = gdf_year.clip(window_gdf)
                    
                    if clipped_gdf.empty:
                        continue
                    
                    # Create overlay config
                    clipped_overlay_config = {
                        "gdf": clipped_gdf,
                        "alpha": overlay_config.get("alpha", 0.7),
                        "zorder": overlay_config.get("zorder", 2),
                    }
                    
                    # Copy optional parameters
                    for param in ["edgecolor", "facecolor", "linewidth", "linestyle"]:
                        if param in overlay_config:
                            clipped_overlay_config[param] = overlay_config[param]

                    if "label" in overlay_config:
                        clipped_overlay_config["label"] = overlay_config["label"]


                    year_overlay_configs.append(clipped_overlay_config)
                
                col_overlays[year] = year_overlay_configs
                
                if year_overlay_configs:
                    print(
                        f"    Vector overlays ({len(year_overlay_configs)}) added: "
                        f"\"{col_name}\" [{year}]"
                    )
        
        result['vector_overlays'][col_name] = col_overlays

    return result



################################################################################
## HANDLE UNDEFINED PLOT YEARS
################################################################################
def auto_detect_year_range(
    locations_gdf: gpd.GeoDataFrame,
    background_config: dict[str, dict],
    strategy: str = "intersection",
) -> list[int]:
    """
    Automatically detect which years to plot based on available rasters.
    
    Parameters
    ----------
    locations_gdf : gpd.GeoDataFrame
        GeoDataFrame with location windows (to resolve {location} placeholders).
    background_config : dict[str, dict]
        Per-column background raster configuration.
    strategy : str, default "intersection"
        Strategy for combining years from multiple rasters:
        - "intersection": Years common to ALL rasters (safest)
        - "union": All years from ANY raster (may have gaps)
        - "first": Years from first raster
        - "max": Years from raster with most years
    
    Returns
    -------
    list[int]
        Sorted list of years to plot.
    
    Raises
    ------
    ValueError
        If no years can be detected or strategy is invalid.
    
    Examples
    --------
    >>> years = auto_detect_year_range(
    ...     locations_gdf,
    ...     background_config,
    ...     strategy="intersection"
    ... )
    >>> # Returns: [2017, 2018, 2019, 2020]  (common to all rasters)
    """
    from pathlib import Path
    
    # Get first location for path resolution
    if locations_gdf.empty:
        raise ValueError("locations_gdf is empty, cannot auto-detect years")
    
    first_location = locations_gdf.iloc[0]['location']
    
    # Collect years from all rasters
    all_raster_years = []
    
    for col_name, bg_config in background_config.items():
        # Resolve path for first location
        raster_path = resolve_background_path(
            bg_config['raster_path'],
            location=first_location,
        )
        
        if not raster_path.exists():
            print(f"  Warning: Raster not found for auto-detection: {raster_path}")
            continue
        
        # Get band mapping
        try:
            bands_per_year = bg_config.get('bands_per_year', 1)
            band_mapping = get_band_mapping_for_raster(
                raster_path,
                band_mapping=bg_config.get('band_mapping', 'auto'),
                bands_per_year=bands_per_year,
            )
            
            years = sorted(band_mapping.keys())
            all_raster_years.append(years)
            print(f"  {col_name}: {len(years)} years → {years}")
        
        except Exception as e:
            print(f"  Warning: Could not extract years from {col_name}: {e}")
            continue
    
    if not all_raster_years:
        raise ValueError(
            "Could not extract years from any raster. "
            "Please specify year_range explicitly."
        )
    
    # ========================================================================
    # Apply strategy
    # ===================================================================F=====
    
    if strategy == "intersection":
        # Years common to ALL rasters
        common_years = set(all_raster_years[0])
        for years in all_raster_years[1:]:
            common_years &= set(years)
        
        if not common_years:
            # Fallback: use years from first raster
            print("  Warning: No common years across all rasters! "
                  "Falling back to 'first' strategy.")
            result_years = all_raster_years[0]
        else:
            result_years = sorted(common_years)
        
        print(f"\n  Strategy 'intersection': Using {len(result_years)} common years → {result_years}")
    
    elif strategy == "union":
        # All years from ANY raster
        all_years = set()
        for years in all_raster_years:
            all_years |= set(years)
        
        result_years = sorted(all_years)
        print(f"\n  Strategy 'union': Using {len(result_years)} years from all rasters → {result_years}")
        print("  Warning: Some cells may be empty if rasters don't have all years!")
    
    elif strategy == "first":
        # Years from first raster
        result_years = all_raster_years[0]
        print(f"\n  Strategy 'first': Using {len(result_years)} years from first raster → {result_years}")
    
    elif strategy == "max":
        # Years from raster with most years
        max_years = max(all_raster_years, key=len)
        result_years = max_years
        print(f"\n  Strategy 'max': Using {len(result_years)} years from longest raster → {result_years}")
    
    else:
        raise ValueError(
            f"Invalid year_range_strategy: '{strategy}'. "
            f"Must be one of: 'intersection', 'union', 'first', 'max'"
        )
    
    if not result_years:
        raise ValueError(
            "No years detected after applying strategy. "
            "Please check rasters or specify year_range explicitly."
        )
    
    return result_years



################################################################################
## SINGLE PLOT
################################################################################
import numpy as np
from typing import Any
import geopandas as gpd
from affine import Affine

# Constants
VALID_ILLUSTRATIONS = {
    "north_arrow", 
    "scalebar", 
    "legend", 
    "line_legend",
    "consistency_legend",
    "unstable_pixels_legend",
    "class_performance", 
    "class_pixel_counts",
    "stability_percentages",
    "stability_tiles_legend",
}
YEAR_REQUIRED_ILLUSTRATIONS = {
    "class_performance", 
    "class_pixel_counts",
    "stability_percentages",
}


def plot_prepared_cell(
    ax: plt.Axes,
    background: np.ndarray | None,
    bounds: tuple[float, float, float, float],
    vector_overlay_list: list[dict],
    illustrations: list[str],
    illustration_settings: dict,
    cell_figsize: tuple[float, float],
    raster_overlay_list: list[dict] | None = None,  
    year: int | None = None,
    background_interpolation: str = 'bilinear',
    location: str | None = None,  
    hab_selection: str | None = None, 
    train_split_attempt: str | None = None,
    band_selection: str | None = None,
    vis_years: str | None = None, 
) -> None:
    """
    Plot a single cell with pre-processed data.
    
    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axis to plot on
    background : np.ndarray | None
        Pre-clipped RGBA background image (H, W, 4)
    bounds : tuple
        Spatial bounds (minx, miny, maxx, maxy) for both background and axis limits
    vector_overlay_list : list[dict]
        List of overlay configs, each containing:
        - "gdf": GeoDataFrame to plot (must have style columns)
        - "alpha": transparency (optional, default 0.7)
        - "zorder": drawing order (optional, default 2)
    raster_overlay_list : list[dict] | None
        List of raster overlay configs, each containing:
        - "rgba_array": Pre-clipped RGBA array (H, W, 4)
        - "alpha": transparency (optional, default 0.7)
        - "zorder": drawing order (optional, default 3)
    illustrations : list[str]
        List of illustration types to add
    illustration_settings : dict
        Settings for each illustration type
    cell_figsize : tuple
        Cell size (width, height) for scaling illustrations
    year : int | None
        Year for time-dependent illustrations
    background_interpolation : str
        Interpolation method for background ('bilinear', 'nearest', etc.)
    """
    
    # Set axis limits
    minx, miny, maxx, maxy = bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect('equal')

    # Hide axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.xticklabels = []
    ax.yticklabels = []

    # Plot background
    _plot_background(ax, background, bounds, background_interpolation)

    # Plot raster overlays (BEFORE vector overlays, respecting zorder)  # ADD THIS
    if raster_overlay_list:
        _plot_raster_overlay_list(ax, raster_overlay_list, bounds, background_interpolation)

    # Plot vector overlays (in order, respecting zorder and alpha from configs)
    _plot_vector_overlay_list(ax, vector_overlay_list)

    # Validate & add illustrations
    illustrations = _validate_illustrations(illustrations, year) 

    # Add illustrations
    for illustration_type in illustrations:
        _add_illustration(
            ax, 
            illustration_type, 
            illustration_settings, 
            cell_figsize, 
            year,
            location=location, 
            hab_selection=hab_selection,
            train_split_attempt=train_split_attempt, 
            band_selection=band_selection,
            vis_years=vis_years,
            vector_overlay_list=vector_overlay_list,
        )


# ========================================================================
# BACKGROUND HANDLER
# ========================================================================
def _plot_background(
    ax: plt.Axes,
    background: np.ndarray | None,
    bounds: tuple[float, float, float, float],
    interpolation: str,
) -> None:
    """
    Plot pre-clipped RGBA background raster.
    
    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axis to plot on
    background : np.ndarray | None
        Pre-clipped RGBA background image (H, W, 4)
    bounds : tuple
        Spatial bounds (minx, miny, maxx, maxy) matching the clipped raster
    interpolation : str
        Interpolation method ('bilinear', 'nearest', etc.)
    """
    if background is None:
        print(f"        _plot_background: background is None, skipping")
        return
    

    try:
        if background.ndim != 3 or background.shape[2] != 4:
            print(
                f"    Warning: Expected RGBA background (H, W, 4), "
                f"got shape {background.shape}"
            )
            return
        
        # Unpack bounds
        minx, miny, maxx, maxy = bounds
        extent = [minx, maxx, miny, maxy]
        
        # Plot
        ax.imshow(
            background,
            extent=extent,
            interpolation=interpolation,
            origin='upper',
            zorder=1,
        )
    
    except Exception as e:
        print(f"    Warning: Could not plot background: {e}")


# ========================================================================
# VECTOR OVERLAY HANDLERS
# ========================================================================
def _plot_vector_overlay_list(
    ax: plt.Axes,
    overlay_list: list[dict],
) -> None:
    """
    Plot list of overlay configs in order.
    
    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axis to plot on
    overlay_list : list[dict]
        List of overlay configs, each containing:
        - "gdf": GeoDataFrame to plot
        - "edgecolor", "facecolor", "linewidth", "linestyle": styling params
        - "alpha": transparency (optional, default 0.7)
        - "zorder": drawing order (optional, default 2)
    """
    if not overlay_list:
        return
    
    try:
        # Sort by zorder (lower zorder = drawn first)
        sorted_overlays = sorted(
            overlay_list,
            key=lambda x: x.get("zorder", 2)
        )
        
        # Plot each overlay
        for overlay_config in sorted_overlays:
            gdf = overlay_config.get("gdf")
            
            if gdf is None or gdf.empty:
                continue
            
            # Pass entire config to preserve all styling parameters
            _plot_single_overlay(ax, overlay_config)
    
    except Exception as e:
        print(f"    Warning: Could not plot overlay list: {e}")


def _plot_single_overlay(ax: plt.Axes, overlay_config: dict) -> None:
    gdf = overlay_config.get("gdf")
    
    if gdf is None or gdf.empty:
        print(f"      -> Skipping overlay: GDF is None or empty")
        return
    
    alpha = overlay_config.get("alpha", 0.7)
    zorder = overlay_config.get("zorder", 2)
    
    plot_kwargs = {
        "ax": ax,
        "alpha": alpha,
        "zorder": zorder,
        "rasterized": False,
    }
    
    # For each style parameter
    for param in ["edgecolor", "facecolor", "linewidth", "linestyle"]:
        if param in overlay_config:
            plot_kwargs[param] = overlay_config[param]
        elif param in gdf.columns:
            plot_kwargs[param] = gdf[param]
    
    try:
        gdf_plot = gdf.copy()
        
        # Count vertices (handle different geometry types)
        def count_vertices(geom):
            if geom.geom_type == 'Polygon':
                return len(geom.exterior.coords)
            elif geom.geom_type == 'MultiPolygon':
                return sum(len(p.exterior.coords) for p in geom.geoms)
            elif geom.geom_type == 'LineString':
                return len(geom.coords)
            elif geom.geom_type == 'MultiLineString':
                return sum(len(line.coords) for line in geom.geoms)
            return 0
        
        n_vertices = gdf_plot.geometry.apply(count_vertices).sum()
        
        # Calculate tolerance based on axis extent
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        extent_width = xlim[1] - xlim[0]
        extent_height = ylim[1] - ylim[0]
        avg_extent = (extent_width + extent_height) / 2
        
        # MORE AGGRESSIVE simplification for large geometries
        if n_vertices > 50000:
            tolerance = avg_extent * 0.01  # 1% for very complex geometries
        elif n_vertices > 10000:
            tolerance = avg_extent * 0.005  # 0.5% for moderately complex
        else:
            tolerance = avg_extent * 0.001  # 0.1% for simple geometries
        
        
        gdf_plot['geometry'] = gdf_plot['geometry'].simplify(
            tolerance=tolerance,
            preserve_topology=True
        )
        
        n_vertices_after = gdf_plot.geometry.apply(count_vertices).sum()
        reduction = 100 * (1 - n_vertices_after / n_vertices) if n_vertices > 0 else 0
        
        gdf_plot.plot(**plot_kwargs)
        
    except Exception as e:
        print(f"      Warning: Could not plot overlay: {e}")

# ========================================================================
# RASTER OVERLAY HANDLER
# ========================================================================
def _plot_raster_overlay_list(
    ax: plt.Axes,
    overlay_list: list[dict],
    bounds: tuple[float, float, float, float],
    interpolation: str = 'bilinear',
) -> None:
    """
    Plot list of raster overlay configs in order.
    
    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axis to plot on
    overlay_list : list[dict]
        List of raster overlay configs, each containing:
        - "rgba_array": Pre-clipped RGBA array (H, W, 4)
        - "alpha": transparency (optional, default 0.7) - NOTE: already baked into RGBA
        - "zorder": drawing order (optional, default 3)
    bounds : tuple
        Spatial bounds (minx, miny, maxx, maxy) matching the clipped rasters
    interpolation : str
        Interpolation method ('bilinear', 'nearest', etc.)
    """
    if not overlay_list:
        return
    
    try:
        # Sort by zorder (lower zorder = drawn first)
        sorted_overlays = sorted(
            overlay_list,
            key=lambda x: x.get("zorder", 3)
        )
        
        # Unpack bounds
        minx, miny, maxx, maxy = bounds
        extent = [minx, maxx, miny, maxy]
        
        # Plot each raster overlay
        for overlay_config in sorted_overlays:
            rgba_array = overlay_config.get("rgba_array")
            zorder = overlay_config.get("zorder", 3)
            
            if rgba_array is None:
                continue
            
            if rgba_array.ndim != 3 or rgba_array.shape[2] != 4:
                print(
                    f"    Warning: Expected RGBA overlay (H, W, 4), "
                    f"got shape {rgba_array.shape}"
                )
                continue
            
            # Plot (alpha is already baked into the RGBA array)
            ax.imshow(
                rgba_array,
                extent=extent,
                interpolation=interpolation,
                origin='upper',
                zorder=zorder,
            )
    
    except Exception as e:
        print(f"    Warning: Could not plot raster overlay list: {e}")


# ========================================================================
# ILLUSTRATION HANDLERS
# ========================================================================
def _validate_illustrations(illustrations: list[str], year: int | None) -> list[str]:
    """Validate and filter illustration types."""
    # Check for unknown types
    invalid = [ill for ill in illustrations if ill not in VALID_ILLUSTRATIONS]
    if invalid:
        print(f"    Warning: Called unknown illustration types: {invalid}")
    
    # Filter to valid types
    valid_illustrations = [ill for ill in illustrations if ill in VALID_ILLUSTRATIONS]
    
    # Check year requirement
    if any(ill in YEAR_REQUIRED_ILLUSTRATIONS for ill in valid_illustrations) and year is None:
        print(f"    Warning: year required for {YEAR_REQUIRED_ILLUSTRATIONS}, skipping those")
        valid_illustrations = [
            ill for ill in valid_illustrations 
            if ill not in YEAR_REQUIRED_ILLUSTRATIONS
        ]
    
    return valid_illustrations


def _add_illustration(
    ax: plt.Axes,
    illustration_type: str,
    illustration_settings: dict,
    cell_figsize: tuple[float, float],
    year: int | None,
    location: str | None = None, 
    hab_selection: str | None = None,  
    train_split_attempt: str | None = None, 
    band_selection: str | None = None, 
    vis_years: str | None = None, 
    vector_overlay_list: list[dict] | None = None,
) -> None:
    """Add a single illustration to the axis."""
    
    handlers = {
        "north_arrow": _handle_north_arrow,
        "scalebar": _handle_scalebar,
        "legend": _handle_legend,
        "line_legend": _handle_line_legend,
        "consistency_legend": _handle_consistency_legend,
        "unstable_pixels_legend": _handle_unstable_pixels_legend,
        "class_performance": _handle_class_performance,
        "class_pixel_counts": _handle_class_pixel_counts,
        "stability_percentages": _handle_stability_percentages,
        "stability_tiles_legend": _handle_stability_tiles,
    }
    
    handler = handlers.get(illustration_type)
    if handler is None:
        return
    
    try:
        if illustration_type == "stability_percentages":
            handler(
                ax, 
                illustration_settings, 
                cell_figsize, 
                year,
                location=location,  
                hab_selection=hab_selection,  
                train_split_attempt=train_split_attempt,  
                band_selection=band_selection,  
                vis_years=vis_years  
            )
        elif illustration_type == "line_legend":
            handler(ax, illustration_settings, cell_figsize, vector_overlay_list)
        else:
            handler(ax, illustration_settings, cell_figsize, year)
    except Exception as e:
        print(f"    Warning: Failed to add {illustration_type}: {e}")


# ========================================================================
# ILLUSTRATION SETUP
# ========================================================================
def _handle_north_arrow(ax, settings_dict, cell_figsize, year):
    settings = settings_dict.get("north_arrow", {})
    svg_path = settings.get("svg_path")
    if svg_path is None:
        raise ValueError("north_arrow requires svg_path")
    
    default_kwargs = get_default_north_arrow_kwargs(cell_figsize)
    final_kwargs = {**default_kwargs, **settings}
    final_kwargs.pop('svg_path', None)
    add_svg_northarrow(ax, svg_path, **final_kwargs)


def _handle_scalebar(ax, settings_dict, cell_figsize, year):
    settings = settings_dict.get("scalebar", {})
    default_kwargs = get_default_scalebar_kwargs(cell_figsize, ax=ax)
    final_kwargs = {**default_kwargs, **settings}
    add_bw_scalebar(ax, **final_kwargs)


def _handle_legend(ax, settings_dict, cell_figsize, year):
    settings = settings_dict.get("legend", {})
    color_df = settings.get("color_df")
    if color_df is None:
        raise ValueError("legend requires color_df")
    
    default_kwargs = get_default_legend_kwargs(cell_figsize)
    final_kwargs = {**default_kwargs, **settings}
    final_kwargs.pop('color_df', None)
    add_color_legend(ax, color_df, **final_kwargs)


def _handle_line_legend(ax, settings_dict, cell_figsize, overlays):
   
    # Handle None case
    if overlays is None:
        return
    
    # Extract legend items from overlays
    legend_items = []
    for overlay in overlays:
        if 'label' in overlay:
            legend_items.append({
                'label': overlay['label'],
                'edgecolor': overlay.get('edgecolor', 'black'),
                'linestyle': overlay.get('linestyle', 'solid'),
                'linewidth': overlay.get('linewidth', 1.0),
            })
    
    if not legend_items:
        return
    
    settings = settings_dict.get("line_legend", {})
    default_kwargs = get_default_line_legend_kwargs(cell_figsize)
    final_kwargs = {**default_kwargs, **settings}
    
    add_line_legend(ax, legend_items, **final_kwargs)


def _handle_consistency_legend(ax, settings_dict, cell_figsize, year):
    """Handle consistency legend illustration."""
    settings = settings_dict.get("consistency_legend", {})
    
    # Get consistency color dataframe
    consistency_df = settings.get("color_df")
    if consistency_df is None:
        raise ValueError("consistency_legend requires color_df")
    
    # Get custom kwargs or use defaults
    default_kwargs = get_default_legend_kwargs(cell_figsize)
    legend_kwargs = settings.get("legend_kwargs", {})
    
    # Merge defaults with custom settings
    final_kwargs = {**default_kwargs, **legend_kwargs}
    
    # Use 'description' as label column for consistency data
    final_kwargs['label_col'] = 'description'
    
    # Add the legend
    add_color_legend(
        ax=ax,
        color_df=consistency_df,
        **final_kwargs
    )

def _handle_unstable_pixels_legend(ax, settings_dict, cell_figsize, year):
    """Handle unstable pixels legend illustration."""
    settings = settings_dict.get("unstable_pixels_legend", {})
    color_df = settings.get("color_df")
    
    if color_df is None:
        raise ValueError("unstable_pixels_legend requires color_df")
    
    default_kwargs = get_default_unstable_pixels_legend_kwargs(cell_figsize)
    final_kwargs = {**default_kwargs, **settings}
    
    # Remove only the required parameter
    final_kwargs.pop('color_df', None)
    
    add_unstable_pixels_legend(
        ax, 
        color_df=color_df,
        **final_kwargs
    )

def _handle_class_performance(ax, settings_dict, cell_figsize, year):
    settings = settings_dict.get("class_performance", {})
    metrics_df = settings.get("metrics_df")
    color_df = settings.get("color_df")
    
    if metrics_df is None:
        raise ValueError("class_performance requires metrics_df")
    if color_df is None:
        raise ValueError("class_performance requires color_df")
    
    default_kwargs = get_default_class_performance_kwargs(cell_figsize)
    final_kwargs = {**default_kwargs, **settings}

    final_kwargs.pop('metrics_df', None)
    final_kwargs.pop('color_df', None)
    
    add_class_performance_badges(
        ax, 
        metrics_df=metrics_df,
        year=year,
        color_df=color_df,
        **final_kwargs
    )


def _handle_class_pixel_counts(ax, settings_dict, cell_figsize, year):
    settings = settings_dict.get("class_pixel_counts", {})
    color_df = settings.get("color_df")
    gdf = settings.get("gdf")
    
    if color_df is None:
        raise ValueError("class_pixel_counts requires color_df")
    if gdf is None or gdf.empty:
        raise ValueError("class_pixel_counts requires non-empty gdf")
    
    default_kwargs = get_default_class_pixel_counts_kwargs(cell_figsize)
    final_kwargs = {**default_kwargs, **settings}
    final_kwargs.pop('color_df', None)
    final_kwargs.pop('gdf', None)
    
    add_class_pixel_counts(
        ax, 
        gdf=gdf,
        color_df=color_df,
        year=year,
        **final_kwargs
    )


def _handle_stability_percentages(ax, settings_dict, cell_figsize, year, location, hab_selection, train_split_attempt, band_selection, vis_years):
    """Handle stability_percentages illustration."""
    
    settings = settings_dict.get("stability_percentages", {})
    color_df = settings.get("color_df")
    raster_path_template = settings.get("raster_path_template")
    
    if color_df is None:
        raise ValueError("stability_percentages requires color_df")
    if raster_path_template is None:
        raise ValueError("stability_percentages requires raster_path_template")
    
    # Use the resolve function instead of simple format
    try:
        raster_path = resolve_stability_overlay_path(
            path_template=raster_path_template,
            location=location,
            hab_selection=hab_selection,
            train_split_attempt=train_split_attempt,
            band_selection=band_selection,
            vis_years=vis_years,
            year=year
        )
    except (KeyError, ValueError) as e:
        print(f"    Warning: Failed to resolve raster path: {e}")
        return
    
    if not raster_path.exists():
        print(f"    Warning: Raster not found: {raster_path}")
        return
    
    default_kwargs = get_default_stability_percentages_kwargs(cell_figsize)
    final_kwargs = {**default_kwargs, **settings}
    for key in ['color_df', 'raster_path_template']:
        final_kwargs.pop(key, None)
    
    add_stability_percentages(
        ax, 
        raster_path=raster_path,
        color_df=color_df,
        **final_kwargs
    )

def _handle_stability_tiles(ax, settings_dict, cell_figsize, year):
    """Handle stability_tiles_legend illustration."""
    
    settings = settings_dict.get("stability_tiles_legend", {})
    color_df = settings.get("color_df")
    alpha_map = settings.get("alpha_map")
    
    if color_df is None:
        raise ValueError("stability_tiles_legend requires color_df")
    if alpha_map is None:
        raise ValueError("stability_tiles_legend requires alpha_map")
    
    default_kwargs = get_default_alpha_legend_kwargs(cell_figsize)
    final_kwargs = {**default_kwargs, **settings}
    
    # Remove only required parameters (keep all optional ones)
    final_kwargs.pop('color_df', None)
    final_kwargs.pop('alpha_map', None)
    
    add_alpha_legend(
        ax, 
        color_df=color_df,
        alpha_map=alpha_map,
        **final_kwargs
    )



################################################################################
## FINAL PLOTTING STATEMENT
################################################################################
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.gridspec as gridspec
import gc
from IPython.display import SVG, display

def plot_all_locations(
    locations_gdf: gpd.GeoDataFrame,
    background_config: dict[str, dict],
    gdf_overlay_config: dict[str, list[dict]],
    illustrations_config: dict[str, dict],
    RF_row_stability_config: dict[str, dict] | None = None,
    RF_row_stable_pixels_config: dict[str, list[dict]] | None = None,
    RF_row_unstable_pixels_config: dict[str, list[dict]] | None = None,
    meta_rows_config: dict[str, dict] | None = None,
    header_row_config: dict | None = None,
    locations: list[str] | None = None,
    vis_years: str | None = None,
    years: list[int] | str = "intersection",
    columns_to_vis: list[str] | None = None,
    figsize: tuple[int, int] | None = None,
    dpi: int = 300,
    background_interpolation: str = 'bilinear',
    output_dir: str | Path | None = None,
    hab_selection: str | None = None,
    train_split_attempt: str | None = None,
    band_selection: str | None = None,
    overwrite: bool = False,
) -> dict[str, str]:
    """
    Create separate grid plots for each location.
    
    Parameters
    ----------
    locations_gdf : gpd.GeoDataFrame
        GeoDataFrame with location geometries and 'location' column
    background_config : dict[str, dict]
        Configuration for background rasters per column    
    gdf_overlay_config : dict[str, list[dict]]
        Vector overlays per column. Each overlay is a dict with:
        - "gdf": gpd.GeoDataFrame (required)
        - "alpha": float (optional, default 0.7)
        - "zorder": int (optional, default 2)
        - "edgecolor": str (optional)
        - "facecolor": str (optional)
        - "linewidth": float (optional, default 1.0)
        Example:
            {
                "Column1": [
                    {"gdf": gdf1, "alpha": 0.5, "zorder": 1},
                    {"gdf": gdf2, "alpha": 0.8, "zorder": 2},
                ],
            }
    illustrations_config : dict[str, dict]
        Combined illustration configuration with:
        - "_global": Global settings for all columns
        - "<column_name>": Column-specific config with "illustrations" list and "settings" dict
    locations : list[str] | None
        Specific locations to plot. If None, plot all locations.
    years : list[int] | str
        Years to plot. Can be:
        - list[int]: Explicit years like [2020, 2021, 2022]
        - "intersection": Auto-detect years present in ALL columns
        - "union": Auto-detect years present in ANY column
        Default is "intersection".
    columns_to_vis : list[str] | None
        Columns to include. If None, use all columns from background_config.
    RF_row_stability_config : dict[str, dict]
        Configuration for RF stability visualization. Should include:
        - "RF_row_stability": Stability/confidence map
        - "RF_stable_pixels": Stable pixel map
        - "RF_unstable_pixels": Unstable pixel map
        meta_rows_config : dict[str, dict] | None
        Configuration for meta-rows that visualize cross-column information.
        Each key is a meta-row name, value is a dict with:
        - "base_columns": list[str] - which base columns this meta-row represents
        - "background_config": dict[str, dict] - per-column background config
        - "gdf_overlay_config": dict[str, list[dict]] - per-column vector overlays
        - "RF_row_stability_config": dict[str, dict] | None - stability overlays
        - "RF_row_stable_pixels_config": dict[str, dict] | None - stable pixels overlays
        - "RF_row_unstable_pixels_config": dict[str, dict] | None - unstable pixels overlays
        - "illustrations_config": dict[str, dict] | None - per-column illustrations
        
        Example:
            {
                "Stability": {
                    "base_columns": ["Carto", "RF_Q1", "RF_Q4"],
                    "background_config": {
                        "Carto": {"raster_path": carto_stability_path, ...},
                        "RF_Q1": {"raster_path": rf_q1_stability_path, ...},
                        "RF_Q4": {"raster_path": rf_q4_stability_path, ...},
                    },
                    "RF_row_stability_config": {...},
                },
            }
    header_row_config : dict | None
        Configuration for a header row showing the entire Veluwe area.
        Structure:
        {
            "window_gdf": GeoDataFrame with Veluwe boundary,
            "background_config": dict[str, dict] - per-column background config,
            "gdf_overlay_config": dict[str, list[dict]] - per-column vector overlays,
            "illustrations_config": dict[str, dict] - per-column illustrations,
            "row_label": str - label for the row (default: "Veluwe Overview"),
        }
        
        Example:
            {
                "window_gdf": veluwe_boundary_gdf,
                "row_label": "Veluwe Overview",
                "background_config": {
                    "Carto": {"raster_path": carto_output_path, "color_df": colors_df},
                    "RF_Q1": {"raster_path": RF_Q1_path, "color_df": colors_df},
                },
                "gdf_overlay_config": {
                    "Carto": [{"gdf": all_locations_gdf, "edgecolor": "red", ...}],
                },
            }
    figsize : tuple[int, int] | None
        Size of each cell (width, height). Default is (3, 3).
    dpi : int
        Resolution for output images
    background_interpolation : str
        Interpolation method for background rasters
    output_dir : str | Path | None
        Directory to save output files. If None, only display.
    hab_selection : str | None
        Habitat selection identifier for filename/subdirectory
    train_split_attempt : str | None
        Training split identifier for filename
    band_selection : str | None
        Band selection identifier for filename
    overwrite : bool
        If False, skip locations where output file already exists.
        If True, always regenerate the plot.
    
    Returns
    -------
    dict[str, str]
        Dictionary mapping location names to their output file paths
    """
       
    # ========================================================================
    # VALIDATION OF INPUTS
    # ========================================================================
    
    # -------------------------------------------------------------------
    # Check variables for writing an output file
    # -------------------------------------------------------------------  
    if output_dir is not None:
        missing_params = []
        if hab_selection is None:
            missing_params.append("hab_selection")
        if train_split_attempt is None:
            missing_params.append("train_split_attempt")
        if band_selection is None:
            missing_params.append("band_selection")
        
        if missing_params:
            raise ValueError(
                f"When output_dir is provided, the following parameters are required "
                f"to generate filenames: {', '.join(missing_params)}"
            )
    
    # -------------------------------------------------------------------
    # Locations GDF check
    # ------------------------------------------------------------------- 
    if locations_gdf.empty:
        raise ValueError("locations_gdf is empty")
    
    if 'location' not in locations_gdf.columns:
        raise ValueError("locations_gdf must have 'location' column")
    
    available_locations = locations_gdf['location'].tolist()
    
    # -------------------------------------------------------------------
    # Selecting locations to plot
    # ------------------------------------------------------------------- 
    if locations is None:
        selected_locations = available_locations
    else:
        invalid_locations = [loc for loc in locations if loc not in available_locations]
        if invalid_locations:
            raise ValueError(
                f"Invalid locations: {invalid_locations}. "
                f"Available locations: {available_locations}"
            )
        selected_locations = locations
    
    filtered_locations_gdf = locations_gdf[
        locations_gdf['location'].isin(selected_locations)
    ].copy()
    
    filtered_locations_gdf['location'] = pd.Categorical(
        filtered_locations_gdf['location'],
        categories=selected_locations,
        ordered=True
    )
    filtered_locations_gdf = filtered_locations_gdf.sort_values('location').reset_index(drop=True)
    
    # -------------------------------------------------------------------
    # Selecting columns to plot
    # -------------------------------------------------------------------  
    base_columns = list(background_config.keys())
     
    if columns_to_vis is None:
        selected_columns = base_columns
    else:
        invalid_columns = [
            col for col in columns_to_vis
            if col not in background_config
        ]
        if invalid_columns:
            raise ValueError(
                f"Invalid columns: {invalid_columns}. "
                f"Available columns: {list(background_config.keys())}"
            )
        selected_columns = columns_to_vis

    # -------------------------------------------------------------------
    # Add stability columns if specified
    # ------------------------------------------------------------------- 
    if RF_row_stability_config:
        selected_columns.append("RF row stability")
    if RF_row_stable_pixels_config:
        selected_columns.append("RF stable pixels")
    if RF_row_unstable_pixels_config:
        selected_columns.append("RF unstable pixels")

    # -------------------------------------------------------------------
    # Filter the input dicts for the selected columns
    # -------------------------------------------------------------------    

    filtered_background_config = filter_dict_by_keys(
        source_dict=background_config,
        keys=selected_columns,
        dict_name="background_config",
        allow_missing=False,
    )

    filtered_gdf_overlay_config = filter_dict_by_keys(
        source_dict=gdf_overlay_config,
        keys=selected_columns,
        dict_name="gdf_overlay_config",
        allow_missing=True,
    )

    filtered_illustrations_config = filter_dict_by_keys(
        source_dict=illustrations_config,
        keys=selected_columns,
        dict_name="illustrations_config",
        allow_missing=True,
    )

    # -------------------------------------------------------------------
    # Process year_range parameter
    # ------------------------------------------------------------------- 
    if isinstance(years, list):
        # Explicit year list provided
        year_range = years
        if not year_range:
            raise ValueError("years list is empty")
    
    elif isinstance(years, str):
        # Auto-detect years
        valid_strategies = {"intersection", "union"}
        if years not in valid_strategies:
            raise ValueError(
                f"Invalid years strategy: '{years}'. "
                f"Must be one of {valid_strategies} or a list of integers."
            )
        
        year_range = auto_detect_year_range(
            filtered_locations_gdf,
            filtered_background_config,
            strategy=years,
        )
        
        if not year_range:
            raise ValueError(
                f"No years found using strategy '{years}'. "
                f"Check that your data contains year information."
            )
    
    else:
        raise TypeError(
            f"years must be list[int] or str, got {type(years).__name__}"
        )
    
    # -------------------------------------------------------------------
    # Process meta_rows_config
    # -------------------------------------------------------------------
    meta_rows_to_plot = []
    meta_rows_data = {}

    if meta_rows_config:
        for meta_row_name, meta_row_config in meta_rows_config.items():
            # Validate structure
            if "base_columns" not in meta_row_config:
                raise ValueError(
                    f"meta_rows_config['{meta_row_name}'] must have 'base_columns' key"
                )
            
            meta_base_columns = meta_row_config["base_columns"]
            
            # Filter base columns to only include selected ones
            filtered_base_columns = [col for col in meta_base_columns if col in selected_columns]
            
            # Skip meta-row only if NO base columns are selected
            if not filtered_base_columns:
                print(f"Warning: Meta-row '{meta_row_name}' has no columns in selected_columns")
                print(f"  Skipping meta-row '{meta_row_name}'")
                continue
            
            meta_filtered_background_config = filter_dict_by_keys(
                source_dict=meta_row_config.get("background_config", {}),
                keys=selected_columns,
                dict_name=f"meta_rows_config['{meta_row_name}']['background_config']",
                allow_missing=True,
            )
            
            meta_filtered_gdf_overlay_config = filter_dict_by_keys(
                source_dict=meta_row_config.get("gdf_overlay_config", {}),
                keys=selected_columns,
                dict_name=f"meta_rows_config['{meta_row_name}']['gdf_overlay_config']",
                allow_missing=True,
            )
            
            meta_filtered_RF_row_stability_config = None
            if meta_row_config.get("RF_row_stability_config"):
                meta_filtered_RF_row_stability_config = filter_dict_by_keys(
                    source_dict=meta_row_config["RF_row_stability_config"],
                    keys=selected_columns,
                    dict_name=f"meta_rows_config['{meta_row_name}']['RF_row_stability_config']",
                    allow_missing=True,
                )
            
            meta_filtered_RF_row_stable_pixels_config = None
            if meta_row_config.get("RF_row_stable_pixels_config"):
                meta_filtered_RF_row_stable_pixels_config = filter_dict_by_keys(
                    source_dict=meta_row_config["RF_row_stable_pixels_config"],
                    keys=selected_columns,
                    dict_name=f"meta_rows_config['{meta_row_name}']['RF_row_stable_pixels_config']",
                    allow_missing=True,
                )
            
            meta_filtered_RF_row_unstable_pixels_config = None
            if meta_row_config.get("RF_row_unstable_pixels_config"):
                meta_filtered_RF_row_unstable_pixels_config = filter_dict_by_keys(
                    source_dict=meta_row_config["RF_row_unstable_pixels_config"],
                    keys=selected_columns,
                    dict_name=f"meta_rows_config['{meta_row_name}']['RF_row_unstable_pixels_config']",
                    allow_missing=True,
                )
            
            meta_filtered_illustrations_config = filter_dict_by_keys(
                source_dict=meta_row_config.get("illustrations_config", {}),
                keys=selected_columns,
                dict_name=f"meta_rows_config['{meta_row_name}']['illustrations_config']",
                allow_missing=True,
            )
            
            meta_rows_to_plot.append(meta_row_name)
            meta_rows_data[meta_row_name] = {
                "base_columns": filtered_base_columns,
                "background_config": meta_filtered_background_config,
                "gdf_overlay_config": meta_filtered_gdf_overlay_config,
                "RF_row_stability_config": meta_filtered_RF_row_stability_config,
                "RF_row_stable_pixels_config": meta_filtered_RF_row_stable_pixels_config, 
                "RF_row_unstable_pixels_config": meta_filtered_RF_row_unstable_pixels_config,
                "illustrations_config": meta_filtered_illustrations_config,
            }
    
    # -------------------------------------------------------------------
    # Process header_row_config
    # -------------------------------------------------------------------
    has_header_row = False
    header_row_data = None
    
    if header_row_config:
        if "window_gdf" not in header_row_config:
            raise ValueError("header_row_config must have 'window_gdf' key")
        
        has_header_row = True
        header_row_data = {
            "window_gdf": header_row_config["window_gdf"],
            "background_config": header_row_config.get("background_config", {}),
            "gdf_overlay_config": header_row_config.get("gdf_overlay_config", {}),
            "illustrations_config": header_row_config.get("illustrations_config", {}),
            "row_label": header_row_config.get("row_label", "Overview"),
        }
    
    # -------------------------------------------------------------------
    # Remaining checks
    # ------------------------------------------------------------------- 
    if not selected_columns:
        raise ValueError("No columns selected to plot")
    
    if not selected_locations:
        raise ValueError("No locations selected to plot")
    
    # -------------------------------------------------------------------
    # Extract global illustration settings
    # -------------------------------------------------------------------
    global_illustration_settings = illustrations_config.get("_global", {})
    
    if not global_illustration_settings:
        print("Warning: No '_global' settings found in illustrations_config")

    # ========================================================================
    # SETUP
    # ========================================================================
    
    def build_output_path(location: str) -> Path | None:
        """Helper to build output path for a location"""
        if output_dir is None:
            return None
        
        # Setting dir
        out_dir = Path(output_dir) / hab_selection
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Setting path
        safe_location = location.replace(" ", "_").replace("/", "-")
        filename = f"{safe_location}_{hab_selection}_{train_split_attempt}_{band_selection}_{vis_years}.svg"
        return out_dir / filename

    # Plotting setup
    n_years = len(year_range)
    n_meta_rows = len(meta_rows_to_plot)
    n_header_rows = 1 if has_header_row else 0
    n_total_rows = n_header_rows + n_years + n_meta_rows
    n_columns = len(selected_columns)

    if figsize is None:
        cell_width, cell_height = 3, 3
    else:
        cell_width, cell_height = figsize

    fig_width = n_columns * cell_width
    fig_height = n_total_rows * cell_height + 0.5

    # Start!
    print(f"{'='*60}")
    print(f"Locations: {selected_locations}")
    print(f"Years: {year_range}")
    print(f"Columns: {selected_columns}")
    print(f"Header row enabled: {header_row_data['row_label']}")
    print(f"Meta-rows to plot: {meta_rows_to_plot}")
    print(f"{'='*60}")
   
    # ========================================================================
    # PROCESSING LOOP -> CREATE FIGURE FOR EACH LOCATION SEPARATELY
    # ========================================================================   
    import gc

    output_paths = {}
    locations_processed = []
    locations_skipped = []

    for location in selected_locations:
        output_path = build_output_path(location)
        output_paths[location] = output_path
        
        # -------------------------------------------------------------------
        # CHECK IF FILE EXISTS (skip if overwrite=False)
        # -------------------------------------------------------------------
        
        if not overwrite and output_path and output_path.exists():
            locations_skipped.append(location)
            print(f"\n{location}: Skipping (file exists)")
            continue
        
        # -------------------------------------------------------------------
        # FILE DOESN'T EXIST OR OVERWRITE=TRUE → PROCESS
        # -------------------------------------------------------------------
        
        if output_path and output_path.exists():
            print(f"\n{location}: Overwriting existing file")
        elif output_path:
            print(f"\n{location}: Creating new file")
        else:
            print(f"\n{location}: Display only (no file output)")
        
        locations_processed.append(location)
        
        # -------------------------------------------------------------------
        # SETTING CLIPPING WINDOW
        # -------------------------------------------------------------------
        window_gdf = filtered_locations_gdf[
            filtered_locations_gdf['location'] == location
        ]
        window_bounds = window_gdf.total_bounds

        # -------------------------------------------------------------------
        # CALCULATE GRID DIMENSIONS & ROW INDEX MAPPING
        # -------------------------------------------------------------------

        # Calculate total number of rows needed
        n_total_rows = 0
        if has_header_row:
            n_total_rows += 1
        n_total_rows += len(year_range)
        if meta_rows_to_plot:
            n_total_rows += len(meta_rows_to_plot)

        print(f"  Grid dimensions: {n_total_rows} rows × {len(selected_columns)} columns")

        # Build row index mapping
        current_row = 0
        row_index_map = {}

        if has_header_row:
            row_index_map['header'] = current_row
            current_row += 1

        for year in year_range:
            row_index_map[year] = current_row
            current_row += 1

        if meta_rows_to_plot:
            for meta_row_name in meta_rows_to_plot:
                row_index_map[meta_row_name] = current_row
                current_row += 1

        # -------------------------------------------------------------------
        # CREATE FIGURE STRUCTURE (lightweight - just axes)
        # -------------------------------------------------------------------        
        print(f"  Creating figure structure...")

        fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
        fig.suptitle(location, fontsize=20, fontweight='bold', fontstyle='italic', y=0.95)

        # Build column groups from selected columns
        predefined_column_groups = [
            ["All Data", "Training Data", "Validation Data"],
            ["Carto", "RF Q1", "RF Q2", "RF Q3", "RF Q4"],
            ["RF row stability", "RF stable pixels", "RF unstable pixels"],
        ]

        column_groups = []
        for group in predefined_column_groups:
            filtered_group = [col for col in group if col in selected_columns]
            if filtered_group:
                column_groups.append(filtered_group)

        if len(column_groups) == 0:
            raise ValueError(f"No valid column groups found.")

        # Grid spacing
        group_spacing = 0.001
        intra_row_spacing = 0.17
        intra_col_spacing = 0.1

        # Create outer GridSpec
        outer_gs = gridspec.GridSpec(
            nrows=n_total_rows,
            ncols=len(selected_columns),
            figure=fig,
            hspace=intra_row_spacing,
            wspace=intra_col_spacing,
            top=0.92,
            bottom=0.05,
            left=0.08,
            right=0.98,
        )

        # Create axes with column group spacing
        axes = {}
        col_position = 0

        for group_idx, group_cols in enumerate(column_groups):
            for local_col_idx, col_name in enumerate(group_cols):
                for row_idx in range(n_total_rows):
                    # Get the subplot
                    ax = fig.add_subplot(outer_gs[row_idx, col_position])
                    
                    # Manually adjust position to add group spacing
                    if group_idx > 0:
                        pos = ax.get_position()
                        # Shift this column group to the right
                        shift = group_spacing * group_idx
                        ax.set_position([
                            pos.x0 + shift, 
                            pos.y0, 
                            pos.width, 
                            pos.height
                        ])
                    
                    axes[(row_idx, col_name)] = ax
                
                col_position += 1

        # -------------------------------------------------------------------
        # CALCULATE ACTUAL CELL SIZE (after layout is created)
        # -------------------------------------------------------------------
        if has_header_row:
            measurement_row = row_index_map[year_range[0]]
        else:
            measurement_row = 0

        sample_ax = axes[(measurement_row, selected_columns[0])]
        bbox = sample_ax.get_position()
        actual_cell_width = bbox.width * fig_width
        actual_cell_height = bbox.height * fig_height

        # -------------------------------------------------------------------
        # PROCESS COLUMN BY COLUMN (memory efficient)
        # -------------------------------------------------------------------

        for group_idx, group_cols in enumerate(column_groups):
            print(f"\n  ========================================")
            print(f"  Processing column group {group_idx + 1}/{len(column_groups)}: {group_cols}")
            print(f"  ========================================")
            
            for local_col_idx, col_name in enumerate(group_cols):
                global_col_idx = sum(len(g) for g in column_groups[:group_idx]) + local_col_idx
                
                print(f"\n    ----------------------------------------")
                print(f"    Processing column: {col_name}")
                print(f"      local_col_idx={local_col_idx}, global_col_idx={global_col_idx}")
                print(f"    ----------------------------------------")
                
                # =================================================================
                # HEADER ROW FOR THIS COLUMN
                # =================================================================               
                if has_header_row:
                    header_window_bounds = header_row_data["window_gdf"].total_bounds
                    
                    # Check if this column is configured in header row
                    if col_name in header_row_data["background_config"]:
                        print(f"      Preparing header row data for '{col_name}'...")
                        
                        # Right before calling prepare_location_data
                        gdf_config = header_row_data["gdf_overlay_config"].get(col_name, [])
                        for i, cfg in enumerate(gdf_config):
                            gdf = cfg.get('gdf')

                        # Prepare data for ONLY this column (AGGREGATE ALL YEARS)
                        prepared_header = prepare_location_data(
                            location=None,
                            year_range=year_range,
                            window_gdf=header_row_data["window_gdf"],
                            background_config={col_name: header_row_data["background_config"][col_name]},
                            gdf_overlay_config={col_name: header_row_data["gdf_overlay_config"].get(col_name, [])},
                            RF_row_stability_config=None,
                            RF_row_stable_pixels_config=None,
                            RF_row_unstable_pixels_config=None,
                            hab_selection=hab_selection,
                            train_split_attempt=train_split_attempt,
                            band_selection=band_selection,
                            vis_years=vis_years,
                            aggregate_years=True,
                            background_year=2023,
                        )

                        # Plot header cell
                        row_idx = row_index_map['header']
                        ax = axes.get((row_idx, col_name))

                        if ax is not None:
                            year = 2023
                            
                            background = prepared_header['backgrounds'].get(col_name, {}).get(year)
                            
                            # Aggregate all overlays across years
                            all_overlays = []
                            for yr, overlay_list in prepared_header['vector_overlays'].get(col_name, {}).items():
                                for i, overlay in enumerate(overlay_list):
                                    gdf = overlay.get('gdf')
                                    num_rows = len(gdf) if gdf is not None else 0
                                    edgecolor = overlay.get('edgecolor', 'none')
                                    facecolor = overlay.get('facecolor', 'none')
                                    alpha = overlay.get('alpha', 1.0)
                                all_overlays.extend(overlay_list)
                            
                            vector_overlay_list = all_overlays
                            
                            raster_overlay_list = prepared_header['raster_overlays'].get(col_name, {}).get(2023, [])
                            
                            col_illustration_config = header_row_data["illustrations_config"].get(col_name, {})
                            col_illustration = col_illustration_config.get("illustrations", [])
                            merged_illustration_settings = merge_illustration_settings(
                                global_illustration_settings,
                                col_illustration_config.get("settings", {}),
                            )
                            
                            plot_prepared_cell(
                                ax=ax,
                                background=background,
                                bounds=tuple(header_window_bounds),
                                vector_overlay_list=vector_overlay_list,
                                raster_overlay_list=raster_overlay_list,
                                illustrations=col_illustration,
                                illustration_settings=merged_illustration_settings,
                                cell_figsize=(actual_cell_width, actual_cell_height),
                                year=year,
                                background_interpolation=background_interpolation,
                            )
                            
                            if global_col_idx == 0:
                                ax.set_ylabel(header_row_data["row_label"], fontsize=10, fontstyle='italic')
                            
                            ax.set_title(col_name, fontsize=10, pad=8)
                        
                        # Free memory
                        del prepared_header
                        gc.collect()
                    else:
                        # Empty cell
                        row_idx = row_index_map['header']
                        ax = axes.get((row_idx, col_name))
                        if ax is not None:
                            ax.axis('off')
                            ax.text(0.5, 0.5, "—", ha='center', va='center',
                                fontsize=20, color='gray', transform=ax.transAxes)
                            if global_col_idx == 0: 
                                ax.set_ylabel(header_row_data["row_label"], fontsize=10, fontstyle='italic')
                            ax.set_title(col_name, fontsize=10, pad=8)
                
                # =================================================================
                # YEAR ROWS FOR THIS COLUMN
                # =================================================================
                if col_name in filtered_background_config:
                    
                    # Prepare data for ONLY this column
                    prepared = prepare_location_data(
                        location=location,
                        year_range=year_range,
                        window_gdf=window_gdf,
                        background_config={col_name: filtered_background_config[col_name]},
                        gdf_overlay_config={col_name: filtered_gdf_overlay_config.get(col_name, [])},
                        RF_row_stability_config={col_name: RF_row_stability_config[col_name]} if RF_row_stability_config and col_name in RF_row_stability_config else None,
                        RF_row_stable_pixels_config={col_name: RF_row_stable_pixels_config[col_name]} if RF_row_stable_pixels_config and col_name in RF_row_stable_pixels_config else None,
                        RF_row_unstable_pixels_config={col_name: RF_row_unstable_pixels_config[col_name]} if RF_row_unstable_pixels_config and col_name in RF_row_unstable_pixels_config else None,
                        hab_selection=hab_selection,
                        train_split_attempt=train_split_attempt,
                        band_selection=band_selection,
                        vis_years=vis_years,
                    )

                    # Plot all year rows for this column using row index map
                    for year in year_range:
                        row_idx = row_index_map[year]  
                        ax = axes.get((row_idx, col_name))
                                               
                        background = prepared['backgrounds'].get(col_name, {}).get(year)
                        vector_overlay_list = prepared['vector_overlays'].get(col_name, {}).get(year, [])
                        raster_overlay_list = prepared['raster_overlays'].get(col_name, {}).get(year, [])

                        col_illustration_config = filtered_illustrations_config.get(col_name, {})
                        col_illustration = col_illustration_config.get("illustrations", [])
                        merged_illustration_settings = merge_illustration_settings(
                            global_illustration_settings,
                            col_illustration_config.get("settings", {}),
                        )

                        plot_prepared_cell(
                            ax=ax,
                            background=background,
                            bounds=tuple(window_bounds),
                            vector_overlay_list=vector_overlay_list,
                            raster_overlay_list=raster_overlay_list,
                            illustrations=col_illustration,
                            illustration_settings=merged_illustration_settings,
                            cell_figsize=(actual_cell_width, actual_cell_height),
                            year=year,
                            background_interpolation=background_interpolation,
                            location=location, 
                            hab_selection=hab_selection, 
                            train_split_attempt=train_split_attempt, 
                            band_selection=band_selection, 
                            vis_years=vis_years,
                        )
                        
                        if global_col_idx == 0: 
                            ax.set_ylabel(str(year), fontsize=10, fontstyle='italic')
                        
                        # Add title only to first year row if no header
                        if year == year_range[0] and not has_header_row:
                            ax.set_title(col_name, fontsize=10, pad=8)
                    
                    # Free memory
                    del prepared
                    gc.collect()
                
                # =================================================================
                # META ROWS FOR THIS COLUMN
                # =================================================================
                if meta_rows_to_plot:
                    for meta_row_name in meta_rows_to_plot:
                        
                        meta_config = meta_rows_data.get(meta_row_name, {})
                        
                        # Check if this column is configured for this meta row
                        if col_name not in meta_config.get("background_config", {}):
                            # Empty cell
                            row_idx = row_index_map[meta_row_name]
                            ax = axes.get((row_idx, col_name))
                            if ax is not None:
                                ax.axis('off')
                                ax.text(0.5, 0.5, "—", ha='center', va='center',
                                    fontsize=20, color='gray', transform=ax.transAxes)
                                if global_col_idx == 0:  
                                    ax.set_ylabel(meta_row_name, fontsize=10, fontstyle='italic')
                            continue
                        
                        # Prepare data for this meta row column
                        year = 2023

                        prepared_meta = prepare_location_data(
                            location=location,
                            year_range=[year],
                            window_gdf=window_gdf,
                            background_config={col_name: meta_config["background_config"].get(col_name)} if col_name in meta_config["background_config"] else {},
                            gdf_overlay_config={col_name: meta_config["gdf_overlay_config"].get(col_name, [])},
                            RF_row_stability_config=(
                                {col_name: meta_config["RF_row_stability_config"][col_name]}
                                if meta_config.get("RF_row_stability_config") and col_name in meta_config["RF_row_stability_config"]
                                else None
                            ),
                            RF_row_stable_pixels_config=(
                                {col_name: meta_config["RF_row_stable_pixels_config"][col_name]}
                                if meta_config.get("RF_row_stable_pixels_config") and col_name in meta_config["RF_row_stable_pixels_config"]
                                else None
                            ),
                            RF_row_unstable_pixels_config=(
                                {col_name: meta_config["RF_row_unstable_pixels_config"][col_name]}
                                if meta_config.get("RF_row_unstable_pixels_config") and col_name in meta_config["RF_row_unstable_pixels_config"]
                                else None
                            ),
                            hab_selection=hab_selection,
                            train_split_attempt=train_split_attempt,
                            band_selection=band_selection,
                            vis_years=vis_years,
                        )
                        
                        # Plot meta cell using row index map
                        row_idx = row_index_map[meta_row_name]
                        ax = axes.get((row_idx, col_name))
                                               
                        if ax is not None:
                            year = 2023                           
                            background = prepared_meta['backgrounds'].get(col_name, {}).get(year)

                            vector_overlay_list = prepared_meta['vector_overlays'].get(col_name, {}).get(year, [])
                            raster_overlay_list = prepared_meta['raster_overlays'].get(col_name, {}).get(year, [])
                            
                            col_illustration_config = meta_config.get("illustrations_config", {}).get(col_name, {})
                            col_illustration = col_illustration_config.get("illustrations", [])
                            merged_illustration_settings = merge_illustration_settings(
                                global_illustration_settings,
                                col_illustration_config.get("settings", {}),
                            )
                            
                            plot_prepared_cell(
                                ax=ax,
                                background=background,
                                bounds=tuple(window_bounds),
                                vector_overlay_list=vector_overlay_list,
                                raster_overlay_list=raster_overlay_list,
                                illustrations=col_illustration,
                                illustration_settings=merged_illustration_settings,
                                cell_figsize=(actual_cell_width, actual_cell_height),
                                year=year,
                                background_interpolation=background_interpolation,
                                location=location, 
                                hab_selection=hab_selection, 
                                train_split_attempt=train_split_attempt, 
                                band_selection=band_selection, 
                                vis_years=vis_years,
                            )
                            
                            if global_col_idx == 3: 
                                ax.set_ylabel(meta_row_name, fontsize=10, fontstyle='italic')
                        
                        # Free memory
                        del prepared_meta
                        gc.collect()

        print(f"\n  ========================================")
        print(f"  Figure created successfully")
        print(f"  ========================================")
        
        # -------------------------------------------------------------------
        # DISPLAY & SAVE FIGURE
        # -------------------------------------------------------------------
        
        display(fig)

        if output_path:
            fig.savefig(output_path, format='svg', dpi=dpi, bbox_inches='tight')
            print(f"  Saved: {output_path.name}")
        
        # -------------------------------------------------------------------
        # CLEANUP MEMORY
        # -------------------------------------------------------------------
        
        plt.close(fig)
        del fig
        del axes
        del outer_gs
        del window_gdf
        del window_bounds
        gc.collect()
        
        print(f"  -> Completed {location}")

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Processed: {len(locations_processed)} location(s)")
    if locations_processed:
        print(f"    {', '.join(locations_processed)}")
    print(f"  Skipped:   {len(locations_skipped)} location(s) (already existed)")
    if locations_skipped:
        print(f"    {', '.join(locations_skipped)}")
    print(f"  Total:     {len(selected_locations)} location(s)")
    print(f"{'='*60}")

    # -------------------------------------------------------------------
    # FINAL CLEANUP AFTER ALL LOCATIONS
    # -------------------------------------------------------------------
    gc.collect()

    # Return all paths (including skipped ones for reference)
    return {
        loc: str(output_paths[loc]) 
        for loc in selected_locations 
        if output_paths[loc]
    }