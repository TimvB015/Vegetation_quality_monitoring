from __future__ import annotations

################################################################################
## FINDING DRIVE
################################################################################
from pathlib import Path

def find_drive_func(which: str) -> Path:
    key = str(which).strip().lower()
    if key == "personal":
        return Path(r"F:\Thesis\10.Thesis_Data")
    if key == "sweco":
        return Path(r"F:\Thesis\10.Thesis_Data")
    raise ValueError(f"Unknown drive '{which}'. Expected 'personal' or 'sweco'.")



################################################################################
## CHECK PROCESSING
################################################################################
from typing import List, Dict, Sequence


def check_processing(
    values: Sequence[int],
    entities: Sequence[str],
    typ: str,
    tmp: str = "tmp0"
) -> List[str]:
    """
    Check whether the given values match the expected reference values
    for the specified TMP level and habitat type.

    Parameters
    ----------
    values : sequence of int
        Values to be checked. The order must correspond exactly
        to the order of `entities`.
    entities : sequence of str
        Names of the entities to check
        (e.g. ["gelderland", "website"]).
    typ : str
        Habitat type. Must be one of: 'AO', 'FC', 'WD', 'S1'.
    tmp : str, optional
        TMP level. Must be 'tmp0' or 'tmp1'. Default is 'tmp0'.

    Returns
    -------
    List[str]
        A list of human-readable messages.
        Each element represents one output line.
    """

    # Input validation
    if len(values) != len(entities):
        raise ValueError("values and entities must have the same length.")

    if not all(isinstance(v, int) for v in values):
        raise ValueError("All elements in values must be integers.")

    typ = typ.upper()
    if typ not in ("AO", "HB", "FC", "WD", "S1"):
        raise ValueError("typ must be one of 'AO', 'HB', 'FC', 'WD', 'S1'.")

    if tmp not in ("tmp0", "tmp1"):
        raise ValueError("tmp must be 'tmp0' or 'tmp1'.")

    # Expected reference values
    tmp_expected: Dict[str, Dict[str, Dict[str, int]]] = {
        "tmp0": {
            "AO": {
                "gelderland": 49663,
                "gelderland_min_hr": 49663,
                "website": 66643,
                "website_plus_hr": 66643,
            },
            "HB": {
                "gelderland": 48895,
                "gelderland_min_hr": 48895,
                "website": 65580,
                "website_plus_hr": 65580,
            },
            "FC": {
                "gelderland": 46379,
                "gelderland_min_hr": 46379,
                "website": 62695,
                "website_plus_hr": 62695,
            },
            "WD": {
                "gelderland": 46379,
                "gelderland_min_hr": 46379,
                "website": 62695,
                "website_plus_hr": 62695,
            },
            "S1": {
                "gelderland": 473,
                "gelderland_min_hr": 473,
                "website": 580,
                "website_plus_hr": 580,
            },
        },
        "tmp1": {
            "AO": {
                "gelderland": 45508,
                "gelderland_min_hr": 45263,
                "website": 58886,
                "website_plus_hr": 59143,
            },
            "HB": {
                "gelderland": 44754,
                "gelderland_min_hr": 44511,
                "website": 57960,
                "website_plus_hr": 58215,
            },
            "FC": {
                "gelderland": 42367,
                "gelderland_min_hr": 42129,
                "website": 55286,
                "website_plus_hr": 55535,
            },
            "WD": {
                "gelderland": 42367,
                "gelderland_min_hr": 42129,
                "website": 55286,
                "website_plus_hr": 55535,
            },
            "S1": {
                "gelderland": 14305,
                "gelderland_min_hr": 14212,
                "website": 16755,
                "website_plus_hr": 16852,
            },
        },
    }

    expected_map = tmp_expected[tmp][typ]

    # Compare values
    diffs = []

    for idx, (entity, actual) in enumerate(zip(entities, values)):
        if entity not in expected_map:
            raise ValueError(f"Unknown entity '{entity}' for {tmp}/{typ}")

        expected = expected_map[entity]
        if expected != actual:
            diffs.append(
                {
                    "index": idx,
                    "name": entity,
                    "expected": expected,
                    "actual": actual,
                }
            )

    # Output
    if not diffs:
        return ["The number of habitat surveys is correct"]

    lines = ["ERROR. The values do not match. Differences:"]
    for d in diffs:
        lines.append(
            f"- {d['name']} (index {d['index']}): expected {d['expected']}, found {d['actual']}"
        )

    return lines



################################################################################
## EXTRACT COLOR PER HAB-TYPE AND BUILD COLOR DF
################################################################################
import pandas as pd
import numpy as np


def extract_selection_colors(
    df: pd.DataFrame,
    hab_selection: str = "habitat",
) -> pd.DataFrame:
    """
    Extract unique class names and colors based on habitat selection level.
    
    Parameters
    ----------
    df : pd.DataFrame
        Habitat dataframe with columns:
        - habitatType (index or column)
        - habitatnaam_disp, habitat_color (for base habitat)
        - WD_division, WD_color (for WD level)
        - WD1_division, WD1_color, WD2_division, WD2_color, etc.
    hab_selection : str
        Selection level: "habitat", "WD", "WD1", "WD2", "WD3", "WD4", "WD5"
        Default: "habitat"
    
    Returns
    -------
    color_df : pd.DataFrame
        DataFrame with columns:
        - 'type': Unique class names
        - 'color': Corresponding hex colors
        
    Examples
    --------
    >>> colors = extract_selection_colors(habitat_df, hab_selection="WD1")
    >>> print(colors)
         type    color
    0     Remaining  #aaaaaa
    
    >>> colors = extract_selection_colors(habitat_df, hab_selection="WD")
    >>> print(colors)
       type    color
    0  No-Habitat  #f54949
    1  Dry Nature  #ffff99
    """
    
    # Determine column names based on selection
    if hab_selection.lower() == "habitat":
        division_col = "habitatnaam_disp"
        color_col = "habitat_color"
    else:
        # For WD, WD1, WD2, etc.
        division_col = f"{hab_selection}_division"
        color_col = f"{hab_selection}_color"
    
    # Validate columns exist
    if division_col not in df.columns or color_col not in df.columns:
        raise ValueError(
            f"Selection '{hab_selection}' not found. "
            f"Expected columns '{division_col}' and '{color_col}' in dataframe. "
            f"Available columns: {df.columns.tolist()}"
        )
    
    # Extract division and color columns
    class_color_mapping = df[[division_col, color_col]].copy()
    class_color_mapping.columns = ['type', 'color']
    
    # Remove NaN values (some WD levels might not be populated)
    class_color_mapping = class_color_mapping.dropna()
    
    # Get unique class-color pairs
    color_df = class_color_mapping.drop_duplicates().reset_index(drop=True)
    
    # Sort by class name for consistency
    color_df = color_df.sort_values('type').reset_index(drop=True)
    
    return color_df