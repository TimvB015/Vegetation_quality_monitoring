################################################################################
##  TRAINING VALIDATION SPLIT
################################################################################
import ast
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import geopandas as gpd
from pandas.api.types import is_scalar


@dataclass
class SplitSummary:
    class_name: Any
    target_train_pxls: float
    got_train_pxls: float
    target_val_pxls: float
    got_val_pxls: float
    target_test_pxls: float
    got_test_pxls: float
    n_train_ids: int
    n_val_ids: int
    n_test_ids: int
    n_leftover_ids: int
    n_unavailable_ids_ignored: int
    n_invalid_ids_ignored: int
    n_polygons_selected: int
    n_large_polygons_selected: int
    n_seed_isolated_discarded: int


def training_validation_split_func(
    gdf: gpd.GeoDataFrame,
    idx_df: pd.DataFrame,
    *,
    year_cols: Iterable[str],
    train_years: Optional[Iterable[str]] = None,
    val_years: Optional[Iterable[str]] = None,
    test_years: Optional[Iterable[str]] = None,
    cap_pixels_by_class: Dict[Any, int],
    train_pct: float,
    val_pct: float,
    test_pct: float = 0.0,
    seed: Optional[int] = None,
    id_col: Optional[str] = None,
    area_col: str = "Shape_Area",
    pixel_area_m2: float = 100.0,
    available_col: Optional[str] = None,
    geometry_col: str = "geometry",
    large_polygon_threshold: int = 36,
    n_seed_attempts_per_class: int = 200000,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split pixel IDs stored in an index table (idx_df) into train/val(/test) under a per-class cap,
    while enforcing year-exclusive polygon usage.

    Overview
    --------
    This function samples *pixel IDs* from `idx_df` for each class (row), producing:
      - `train_df`, `val_df`, `test_df`: selected IDs per year
      - `leftover_df`: remaining selectable IDs (valid + available) not selected and not blocked by polygon exclusivity
      - `summary_df`: per-class accounting and diagnostics

    Inputs
    ------
    gdf : geopandas.GeoDataFrame
        Pixel-level GeoDataFrame containing:
          - a geometry column (default: "geometry", configurable via `geometry_col`)
          - an area column in m² (default: "Shape_Area", configurable via `area_col`)
          - optionally an availability flag column (`available_col`)
        Pixel identifiers are taken from `gdf.index` if `id_col is None`, otherwise from `gdf[id_col]`.

    idx_df : pandas.DataFrame
        Table with:
          - index: class labels
          - columns: years (given by `year_cols`)
          - each cell: list-like (or a string representation of a list) of pixel IDs for that class and year

    year_cols : iterable of str
        Year columns in `idx_df` to consider.

    train_years, val_years, test_years : iterable of str or None
        If provided, restrict sampling for that split to those years only.
        If None, defaults to all `year_cols`.
        If `test_pct == 0`, the test split is disabled (and `test_years` is ignored).

    cap_pixels_by_class : dict
        Mapping {class_label -> cap_in_pixels}. Keys must match `idx_df.index` exactly.
        This cap is the total pixel budget across all splits for that class.

    train_pct, val_pct, test_pct : float
        Fractions of the per-class cap assigned to each split. Must sum to 1.0.

    seed : int or None
        RNG seed for reproducibility.

    id_col : str or None
        Column in `gdf` containing the pixel ID. If None, `gdf.index` is treated as the ID.

    area_col : str
        Column in `gdf` with pixel area in m² (used only to convert selected area to pixel counts).

    pixel_area_m2 : float
        Area of one pixel in m². Used to convert:
            pixels = sum(area_col) / pixel_area_m2
        Must be > 0.

    available_col : str or None
        Optional boolean column in `gdf` indicating if a pixel is selectable.
        If None, all pixels are treated as available.

    geometry_col : str
        Name of the geometry column in `gdf` (renamed internally to "geometry" if needed).

    large_polygon_threshold : int
        If a polygon (see “Polygon grouping” below) has more than this many available pixels in a year,
        only up to `large_polygon_threshold` connected pixels are selected (region-growing).
        Otherwise, all available pixels from that polygon/year are selected.

    n_seed_attempts_per_class : int
        Maximum number of selection attempts per class.

    Polygon grouping and year exclusivity
    ------------------------------------
    - Each pixel ID is mapped to a polygon key by stripping a trailing "_<digits>" (e.g. "polyA_12" -> "polyA").
    - Once any pixels from a polygon key are selected in one year, that polygon key is removed from *all years*
      for that class (i.e., polygons are year-exclusive within a class).

    Large polygon patch selection (connectivity)
    --------------------------------------------
    For polygons with more than `large_polygon_threshold` available pixels in a given year:
      - pick a random seed pixel within that polygon/year subset
      - grow a connected patch (prioritizing edge-sharing neighbours; otherwise corner-touch)
      - if the seed has no touching neighbours (isolated), discard that seed and retry

    Returns
    -------
    train_df, val_df, test_df, leftover_df : pandas.DataFrame
        Each DataFrame has:
          - index == idx_df.index (classes)
          - columns == year_cols (years)
          - each cell is a Python list of pixel IDs

    summary_df : pandas.DataFrame
        Per-class summary (index == class) with the following columns:

        Pixel-budget columns (in **pixels**, not m²):
          - target_train_pxls, got_train_pxls
          - target_val_pxls,   got_val_pxls
          - target_test_pxls,  got_test_pxls

        Count/diagnostic columns:
          - n_train_ids, n_val_ids, n_test_ids, n_leftover_ids
          - n_unavailable_ids_ignored, n_invalid_ids_ignored
          - n_polygons_selected, n_large_polygons_selected, n_seed_isolated_discarded

    Notes / fixes
    -------------
    - The accumulated selection amount (`got_*_pxls`) is computed from the *year/polygon subset* only:
        added_pxls = sum(sub_poly[area_col]) / pixel_area_m2
      (This avoids overcounting that would occur if summing areas from the full gdf.)
    """
    ...

    # ---- validate pcts ----
    for name, p in [("train_pct", train_pct), ("val_pct", val_pct), ("test_pct", test_pct)]:
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"{name} must be between 0 and 1.")
    total = train_pct + val_pct + test_pct
    if not np.isclose(total, 1.0, atol=1e-12):
        raise ValueError("train_pct + val_pct + test_pct must add to 1.0 (100%).")

    if pixel_area_m2 <= 0:
        raise ValueError("pixel_area_m2 must be > 0.")

    # ---- validate cap dict keys ----
    idx_index_set = set(idx_df.index)
    cap_keys = set(cap_pixels_by_class.keys())
    if cap_keys != idx_index_set:
        missing = sorted(idx_index_set - cap_keys)
        extra = sorted(cap_keys - idx_index_set)
        raise ValueError(
            "cap_pixels_by_class keys must match idx_df.index exactly. "
            f"Missing keys: {missing}. Extra keys: {extra}."
        )

    year_cols = list(year_cols)
    for c in year_cols:
        if c not in idx_df.columns:
            raise KeyError(f"Column '{c}' not found in idx_df.")

    # ---- normalize train/val/test year sets ----
    def _norm_years(name: str, yrs: Optional[Iterable[str]]) -> List[str]:
        if yrs is None:
            return list(year_cols)
        yrs_list = list(yrs)
        unknown = [y for y in yrs_list if y not in year_cols]
        if unknown:
            raise KeyError(f"{name} contains years not in year_cols: {unknown}")
        return yrs_list

    train_years = _norm_years("train_years", train_years)
    val_years = _norm_years("val_years", val_years)

    if test_pct > 0.0:
        test_years = _norm_years("test_years", test_years)
    else:
        test_years = [] if test_years is None else _norm_years("test_years", test_years)

    if geometry_col != "geometry":
        gdf = gdf.rename(columns={geometry_col: "geometry"}).set_geometry("geometry")

    if area_col not in gdf.columns:
        raise KeyError(f"'{area_col}' not found in gdf columns.")
    if id_col is not None and id_col not in gdf.columns:
        raise KeyError(f"id_col='{id_col}' not found in gdf columns.")

    rng = np.random.default_rng(seed)

    def _is_na_scalar(x) -> bool:
        return x is None or (is_scalar(x) and pd.isna(x))

    def _parse_cell(v) -> List[Any]:
        if _is_na_scalar(v):
            return []
        if isinstance(v, (list, tuple, set, np.ndarray, pd.Series, pd.Index)):
            return [i for i in list(v) if not _is_na_scalar(i)]
        if isinstance(v, str):
            s = v.strip()
            if s.startswith("[") and s.endswith("]"):
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple, set, np.ndarray, pd.Series, pd.Index)):
                    return [i for i in list(parsed) if not _is_na_scalar(i)]
                return [] if _is_na_scalar(parsed) else [parsed]
            return [v]
        return [v]

    def _polygon_key(id_str: Any) -> str:
        s = id_str if isinstance(id_str, str) else str(id_str)
        parts = s.split("_")
        if len(parts) >= 2 and parts[-1].isdigit():
            return "_".join(parts[:-1])
        return s

    def _dedup_preserve_order(items: List[Any]) -> List[Any]:
        seen, out = set(), []
        for x in items:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    # ---- ID lookup ----
    if id_col is None:
        id_to_gdf_idx = {i: i for i in gdf.index}
    else:
        id_to_gdf_idx = pd.Series(gdf.index.values, index=gdf[id_col].values).to_dict()

    # availability
    if available_col is None:
        available = pd.Series(True, index=gdf.index)
    else:
        if available_col not in gdf.columns:
            raise KeyError(f"available_col='{available_col}' not found in gdf.")
        available = gdf[available_col].fillna(False).astype(bool)

    # ---- neighbour logic: prefer edge then touch ----
    def _edge_neighbours(sub: gpd.GeoDataFrame, seed_idx) -> List[Any]:
        geoms = sub["geometry"]
        seed = geoms.loc[seed_idx]
        sidx = geoms.sindex
        out = []
        for j in sidx.intersection(seed.bounds):
            nid = geoms.index[j]
            if nid == seed_idx:
                continue
            other = geoms.loc[nid]
            if seed.touches(other):
                inter = seed.boundary.intersection(other.boundary)
                if getattr(inter, "length", 0.0) > 0.0:
                    out.append(nid)
        return out

    def _touch_neighbours(sub: gpd.GeoDataFrame, seed_idx) -> List[Any]:
        geoms = sub["geometry"]
        seed = geoms.loc[seed_idx]
        sidx = geoms.sindex
        out = []
        for j in sidx.intersection(seed.bounds):
            nid = geoms.index[j]
            if nid == seed_idx:
                continue
            if seed.touches(geoms.loc[nid]):
                out.append(nid)
        return out

    def _grow_connected_patch(sub: gpd.GeoDataFrame, seed_idx, target_n: int) -> Optional[List[Any]]:
        if len(_touch_neighbours(sub, seed_idx)) == 0:
            return None

        selected = [seed_idx]
        selected_set = {seed_idx}
        queue = [seed_idx]
        explored = set()

        while queue and len(selected) < target_n:
            cur = queue.pop(0)
            if cur in explored:
                continue
            explored.add(cur)

            edge = _edge_neighbours(sub, cur)
            touch = _touch_neighbours(sub, cur)
            candidates = list(dict.fromkeys(edge + [n for n in touch if n not in edge]))

            for n in candidates:
                if n in selected_set:
                    continue
                selected.append(n)
                selected_set.add(n)
                queue.append(n)
                if len(selected) >= target_n:
                    break

        return selected

    # ---- global constraints: IDs and polygons not across rows ----
    id_to_row: Dict[Any, Any] = {}
    polygon_to_row: Dict[str, Any] = {}
    n_invalid_ids_global = 0

    for row_key in idx_df.index:
        for c in year_cols:
            for _id in _parse_cell(idx_df.at[row_key, c]):
                if _id in id_to_row and id_to_row[_id] != row_key:
                    raise ValueError(f"ID '{_id}' occurs in multiple rows/classes: {id_to_row[_id]} and {row_key}")
                id_to_row[_id] = row_key

                pk = _polygon_key(_id)
                if pk in polygon_to_row and polygon_to_row[pk] != row_key:
                    raise ValueError(
                        f"Polygon '{pk}' occurs in multiple rows/classes: {polygon_to_row[pk]} and {row_key}"
                    )
                polygon_to_row[pk] = row_key

                if _id not in id_to_gdf_idx:
                    n_invalid_ids_global += 1

    if n_invalid_ids_global > 0:
        warnings.warn(
            f"{n_invalid_ids_global} IDs referenced in idx_df are not present in gdf "
            f"({'gdf.index' if id_col is None else id_col}); they will be ignored."
        )

    if available_col is not None:
        n_unavail_refs = 0
        for _id in id_to_row.keys():
            gi = id_to_gdf_idx.get(_id, None)
            if gi is not None and not bool(available.loc[gi]):
                n_unavail_refs += 1
        if n_unavail_refs > 0:
            warnings.warn(f"{n_unavail_refs} referenced IDs are marked unavailable and will be ignored.")

    # ---- outputs shaped like idx_df[year_cols] ----
    def _empty_list_df() -> pd.DataFrame:
        return pd.DataFrame(index=idx_df.index, columns=year_cols, data=[[[] for _ in year_cols] for _ in idx_df.index])

    train_df = _empty_list_df()
    val_df = _empty_list_df()
    test_df = _empty_list_df()
    leftover_df = _empty_list_df()

    summaries: List[SplitSummary] = []

    # ---- per class selection loop ----
    for row_key in idx_df.index:
        cap_pixels = int(cap_pixels_by_class[row_key])

        target_train_pxls = float(cap_pixels) * float(train_pct)
        target_val_pxls = float(cap_pixels) * float(val_pct)
        target_test_pxls = float(cap_pixels) * float(test_pct)

        cell_ids_by_year: Dict[str, List[Any]] = {
            c: _dedup_preserve_order(_parse_cell(idx_df.at[row_key, c])) for c in year_cols
        }

        poly_to_ids_by_year: Dict[str, Dict[str, List[Any]]] = {c: {} for c in year_cols}
        invalid_ignored = 0
        unavailable_ignored = 0

        all_valid_av_ids: List[Any] = []
        for c in year_cols:
            for _id in cell_ids_by_year[c]:
                gi = id_to_gdf_idx.get(_id, None)
                if gi is None:
                    invalid_ignored += 1
                    continue
                if not bool(available.loc[gi]):
                    unavailable_ignored += 1
                    continue
                all_valid_av_ids.append(_id)
                pk = _polygon_key(_id)
                poly_to_ids_by_year[c].setdefault(pk, []).append(_id)

        all_valid_av_ids = _dedup_preserve_order(all_valid_av_ids)

        poly_av_count: Dict[str, int] = {}
        for _id in all_valid_av_ids:
            pk = _polygon_key(_id)
            poly_av_count[pk] = poly_av_count.get(pk, 0) + 1

        remaining_years_by_split: Dict[str, set[str]] = {
            "train": {y for y in train_years if len(poly_to_ids_by_year.get(y, {})) > 0},
            "val": {y for y in val_years if len(poly_to_ids_by_year.get(y, {})) > 0},
        }
        if test_pct > 0.0:
            remaining_years_by_split["test"] = {y for y in test_years if len(poly_to_ids_by_year.get(y, {})) > 0}

        got_train_pxls = got_val_pxls = got_test_pxls = 0.0
        n_polygons_selected = 0
        n_large_polygons_selected = 0
        n_seed_isolated_discarded = 0

        train_ids_set, val_ids_set, test_ids_set = set(), set(), set()
        selected_polygons_global: set[str] = set()
        polygon_selected_year: Dict[str, str] = {}

        def _needs(split: str) -> float:
            if split == "train":
                return max(0.0, target_train_pxls - got_train_pxls)
            if split == "val":
                return max(0.0, target_val_pxls - got_val_pxls)
            if split == "test":
                return max(0.0, target_test_pxls - got_test_pxls)
            raise ValueError(split)

        def _refresh_remaining_years_for_split(split: str):
            ys = remaining_years_by_split.get(split, set())
            remaining_years_by_split[split] = {y for y in ys if len(poly_to_ids_by_year[y]) > 0}

        def _delete_polygon_from_all_years(pk: str):
            for y in year_cols:
                poly_to_ids_by_year[y].pop(pk, None)

        def _pick_split() -> str:
            splits = ["train", "val"] + (["test"] if test_pct > 0 else [])

            eligible_splits = []
            for s in splits:
                _refresh_remaining_years_for_split(s)
                if remaining_years_by_split.get(s, set()):
                    eligible_splits.append(s)

            if not eligible_splits:
                return "done"

            needs = [(s, _needs(s)) for s in eligible_splits]
            needs.sort(key=lambda x: x[1], reverse=True)
            if needs[0][1] <= 0:
                return "done"
            top_need = needs[0][1]
            top = [s for s, n in needs if np.isclose(n, top_need)]
            return rng.choice(top)

        attempts = 0
        while attempts < n_seed_attempts_per_class:
            attempts += 1
            split = _pick_split()
            if split == "done":
                break

            years_for_split = list(remaining_years_by_split[split])
            year = rng.choice(years_for_split)

            if len(poly_to_ids_by_year[year]) == 0:
                remaining_years_by_split[split].discard(year)
                continue

            pk = rng.choice(list(poly_to_ids_by_year[year].keys()))

            if pk in selected_polygons_global:
                _delete_polygon_from_all_years(pk)
                for s in list(remaining_years_by_split.keys()):
                    _refresh_remaining_years_for_split(s)
                continue

            poly_ids_year = poly_to_ids_by_year[year][pk]
            poly_gidx = [id_to_gdf_idx[_id] for _id in poly_ids_year]

            # subset ONLY the pixels that belong to this polygon in this year
            sub_poly = gdf.loc[poly_gidx, ["geometry", area_col]].copy()

            is_large = poly_av_count.get(pk, len(sub_poly)) > large_polygon_threshold

            if not is_large:
                chosen_gidx = list(sub_poly.index)
            else:
                n_large_polygons_selected += 1
                seed_gi = rng.choice(list(sub_poly.index))

                grown = _grow_connected_patch(sub_poly, seed_gi, large_polygon_threshold)
                if grown is None:
                    n_seed_isolated_discarded += 1
                    bad_ids = {_id for _id in poly_ids_year if id_to_gdf_idx[_id] == seed_gi}
                    poly_to_ids_by_year[year][pk] = [i for i in poly_to_ids_by_year[year][pk] if i not in bad_ids]
                    if len(poly_to_ids_by_year[year][pk]) == 0:
                        del poly_to_ids_by_year[year][pk]
                        remaining_years_by_split[split].discard(year)
                    continue

                chosen_gidx = grown

            # FIX: sum area on the *year/polygon subset* (sub_poly), not on the full gdf
            added_area_m2 = float(sub_poly.loc[chosen_gidx, area_col].sum())
            added_pxls = added_area_m2 / float(pixel_area_m2)

            if split == "train":
                got_train_pxls += added_pxls
            elif split == "val":
                got_val_pxls += added_pxls
            else:
                got_test_pxls += added_pxls

            chosen_gidx_set = set(chosen_gidx)
            chosen_ids = [_id for _id in poly_ids_year if id_to_gdf_idx[_id] in chosen_gidx_set]

            if split == "train":
                train_ids_set.update(chosen_ids)
            elif split == "val":
                val_ids_set.update(chosen_ids)
            else:
                test_ids_set.update(chosen_ids)

            selected_polygons_global.add(pk)
            polygon_selected_year[pk] = year
            _delete_polygon_from_all_years(pk)
            n_polygons_selected += 1

            for s in list(remaining_years_by_split.keys()):
                _refresh_remaining_years_for_split(s)

        # ---- write outputs per year (exclusive by polygon_selected_year) ----
        for c in year_cols:
            ids = cell_ids_by_year[c]

            train_df.at[row_key, c] = [
                i for i in ids
                if (i in train_ids_set) and (polygon_selected_year.get(_polygon_key(i), None) == c)
            ]
            val_df.at[row_key, c] = [
                i for i in ids
                if (i in val_ids_set) and (polygon_selected_year.get(_polygon_key(i), None) == c)
            ]
            test_df.at[row_key, c] = [
                i for i in ids
                if (i in test_ids_set) and (polygon_selected_year.get(_polygon_key(i), None) == c)
            ]

            used = set(train_df.at[row_key, c]) | set(val_df.at[row_key, c]) | set(test_df.at[row_key, c])

            leftover_df.at[row_key, c] = [
                i for i in ids
                if (i in id_to_gdf_idx)
                and bool(available.loc[id_to_gdf_idx[i]])
                and (i not in used)
                and (_polygon_key(i) not in selected_polygons_global)
            ]

        summaries.append(
            SplitSummary(
                class_name=row_key,
                target_train_pxls=target_train_pxls,
                got_train_pxls=got_train_pxls,
                target_val_pxls=target_val_pxls,
                got_val_pxls=got_val_pxls,
                target_test_pxls=target_test_pxls,
                got_test_pxls=got_test_pxls,
                n_train_ids=sum(len(train_df.at[row_key, c]) for c in year_cols),
                n_val_ids=sum(len(val_df.at[row_key, c]) for c in year_cols),
                n_test_ids=sum(len(test_df.at[row_key, c]) for c in year_cols),
                n_leftover_ids=sum(len(leftover_df.at[row_key, c]) for c in year_cols),
                n_unavailable_ids_ignored=unavailable_ignored,
                n_invalid_ids_ignored=invalid_ignored,
                n_polygons_selected=n_polygons_selected,
                n_large_polygons_selected=n_large_polygons_selected,
                n_seed_isolated_discarded=n_seed_isolated_discarded,
            )
        )

    summary_df = pd.DataFrame([s.__dict__ for s in summaries]).set_index("class_name")
    return train_df, val_df, test_df, leftover_df, summary_df



################################################################################
##  CARTO TRAINGING GPKG
################################################################################
from pathlib import Path
from typing import Optional, Union, Sequence, List, Any, Dict

import geopandas as gpd
import pandas as pd


def idx_to_carto_training_gdf_func(
    idx_df: pd.DataFrame,
    source_gdf: gpd.GeoDataFrame,
    out_gpkg: Optional[Union[str, Path]] = None,
    *,
    idx_cols_to_include: Optional[Sequence[str]] = None,
    # For IDs like "GI_41687_1" -> polygon id "GI_41687"
    polygon_id_n_parts: int = 2,
    polygon_id_sep: str = "_",
    # Output columns
    type_colname: str = "Type",
    years_colname: str = "years",
    polygon_id_colname: str = "polygon_id",
    # writing
    single_layer: bool = True,
    single_layer_name: str = "combined",
) -> gpd.GeoDataFrame:
    """
    Convert an index table (idx_df) that stores *pixel IDs* per class/year into a
    polygon-level GeoDataFrame by unioning only the selected pixels.

    Expected input structure
    ------------------------
    - idx_df:
        * rows: class labels (e.g. "training", "validation", habitat types, etc.)
        * columns: years (or other time slices)
        * each cell: list-like of pixel IDs (strings) that must exist in source_gdf.index
          (e.g. "GI_41687_1", "GI_41687_2", ...)

    - source_gdf:
        * pixel-level GeoDataFrame indexed by pixel ID (must match the IDs in idx_df)
        * each row is a pixel geometry

    What this function does
    -----------------------
    For every (Type=row label in idx_df, years=column label in idx_df):
    1) Select ONLY the pixel IDs listed in that idx_df cell from source_gdf.
       (No expansion to all pixels of a polygon occurs.)
    2) Derive a polygon identifier from each pixel ID (by taking the first
       `polygon_id_n_parts` parts when splitting on `polygon_id_sep`).
       Example: "GI_41687_1" -> "GI_41687".
    3) Union the geometries of the selected pixels that share the same
       (polygon_id, Type, years), yielding one output feature per group.

    Output
    ------
    Returns a GeoDataFrame with columns:
      - polygon_id_colname (also set as index; can be non-unique)
      - type_colname
      - years_colname
      - geometry (union of the selected pixels)

    Notes
    -----
    - If the same polygon_id occurs in multiple years and/or types, you will get
      multiple rows with the same index value (non-unique index is expected).
    - Any pixel IDs present in idx_df but missing from source_gdf.index are reported.
    """

    if idx_cols_to_include is None:
        idx_cols = list(idx_df.columns)
    else:
        idx_cols = list(idx_cols_to_include)
        missing = [c for c in idx_cols if c not in idx_df.columns]
        if missing:
            raise KeyError(f"idx_cols_to_include contains columns not in idx_df: {missing}")

    def to_polygon_id(pixel_id: Any) -> str:
        parts = str(pixel_id).split(polygon_id_sep)
        return polygon_id_sep.join(parts[:polygon_id_n_parts]) if len(parts) >= polygon_id_n_parts else str(pixel_id)

    missing_in_source: List[Any] = []
    dissolved_blocks: List[gpd.GeoDataFrame] = []

    for year_col in idx_cols:
        year_value = str(year_col)

        for typ, cell in idx_df[year_col].items():
            if cell is None or (isinstance(cell, float) and pd.isna(cell)):
                continue
            if not isinstance(cell, (list, tuple, set, pd.Index)):
                raise TypeError(
                    f"Expected list-like in idx_df[{year_col!r}][{typ!r}], got {type(cell)}"
                )

            codes = list(cell)
            if not codes:
                continue

            codes_in = [c for c in codes if c in source_gdf.index]
            codes_missing = [c for c in codes if c not in source_gdf.index]
            if codes_missing:
                missing_in_source.extend(codes_missing)
            if not codes_in:
                continue

            pix = source_gdf.loc[codes_in].copy()
            pix[polygon_id_colname] = [to_polygon_id(i) for i in pix.index]
            pix[type_colname] = str(typ)
            pix[years_colname] = year_value

            group_cols = [polygon_id_colname, type_colname, years_colname]

            # dissolve: geometry is unioned by geopandas; we do not aggregate any extra columns
            dissolved = pix.dissolve(by=group_cols).reset_index()
            dissolved_blocks.append(dissolved)

    if dissolved_blocks:
        combined = gpd.GeoDataFrame(pd.concat(dissolved_blocks, ignore_index=True), crs=source_gdf.crs)
    else:
        combined = gpd.GeoDataFrame(
            columns=[polygon_id_colname, type_colname, years_colname, "geometry"],
            geometry="geometry",
            crs=source_gdf.crs,
        )

    # Keep only the requested output columns (explicitly drop any carried-through columns)
    keep_cols = [polygon_id_colname, type_colname, years_colname, "geometry"]
    combined = combined[[c for c in keep_cols if c in combined.columns]]

    combined = combined.set_index(polygon_id_colname)
    combined.index.name = polygon_id_colname

    if out_gpkg is not None:
        out_gpkg = Path(out_gpkg)
        out_gpkg.parent.mkdir(parents=True, exist_ok=True)

        if single_layer:
            combined.to_file(out_gpkg, layer=single_layer_name, driver="GPKG")
        else:
            for year_value in map(str, idx_cols):
                layer_gdf = combined.loc[combined[years_colname] == year_value]
                if not layer_gdf.empty:
                    layer_gdf.to_file(out_gpkg, layer=year_value, driver="GPKG")

    if missing_in_source:
        print(
            f"Warning: {len(missing_in_source)} codes not found in source_gdf.index "
            f"(showing first 10): {missing_in_source[:10]}"
        )

    return combined