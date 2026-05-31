################################################################################
## STACK RASTERS OVER THE YEARS -> BATCH
################################################################################
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence, Union, Optional, Dict, Any

import rasterio

from paths._support.path_defining_funcs import RF_paths
from functions.raster_ops_funcs import stack_rasters_func


def _read_class_map_from_raster(raster_path: Path) -> Optional[dict]:
    """
    Try to read a class-map dict from raster metadata tags.

    Looks for keys: 'CLASS_MAP' and 'CLASS-MAP'. Values are expected to be JSON
    like: {"0": "Open water", "1": "Remaining"}.

    Returns a dict (possibly with int keys) or None if unavailable/unparseable.
    """
    try:
        with rasterio.open(raster_path) as ds:
            tags = ds.tags()  # dataset-level tags
    except Exception:
        return None

    raw = tags.get("CLASS_MAP") or tags.get("CLASS-MAP")
    if not raw:
        return None

    try:
        cm = json.loads(raw)
        return cm
    except Exception:
        return None


def stacking_rasters_batch(
    RF_results__dir: Union[str, Path],
    years: Sequence[str],
    timeframe: Sequence[str],
    band_selection: Union[str, Sequence[str]],
    hab_selection: Union[str, Sequence[str]],
    train_split_attempt: Union[str, Sequence[str]],
    *,
    crs_tag: str = "UTM32631",
    suffix: str = "__rstr.tif",
    overwrite: bool = False,
    verbose: bool = True,
    add_class_map: bool = True,
) -> List[Path]:
    """
    Batch-stack Random Forest (RF) output rasters across multiple years for each
    timeframe, for all combinations of band, habitat, and train/split attempt.

    For each (band, hab, attempt, tf) combination, this function searches for
    year-specific RF output rasters under ``RF_results__dir`` and stacks the
    rasters that exist into a single multi-band GeoTIFF (one band per year, in
    the order provided by ``years``).

    Optionally, a class label mapping (e.g. ``{"0": "Open water", "1": "Remaining"}``)
    is copied from the input rasters into the produced stack if your
    ``stack_rasters_func`` supports it.

    Input layout (relative to ``RF_results__dir``)
    ----------------------------------------------
    Expected per-year raster files are searched at::

        {band}/{hab}/{year}_{tf}/
            RF_out__{band}__{hab}_{year}_{tf}_{attempt}__{crs_tag}{suffix}

    Output layout
    -------------
    Stacks are written to::

        {band}_stacked/{hab}/

    with filenames::

        stack_{band}__{hab}_{attempt}_{tf}{suffix}

    Parameters
    ----------
    RF_results__dir
        Root directory containing RF result folders.
    years
        Year labels to look for (and to use as band names in the stack). Order
        determines the band order in the output stack.
    timeframe
        Timeframe identifiers (e.g., seasons or periods). For each element ``tf``,
        inputs are searched in ``{year}_{tf}`` subfolders.
    band_selection
        Band name(s) to process. Can be a single band string or a sequence of
        band strings.
    hab_selection
        Habitat class name(s) to process. Can be a single habitat string or a
        sequence of habitat strings.
    train_split_attempt
        Attempt identifier(s) to process (e.g., different train/test splits or
        run IDs). Can be a single string or a sequence of strings.
    crs_tag
        CRS tag embedded in filenames (default ``"UTM32631"``).
    suffix
        Filename suffix for rasters (default ``"__rstr.tif"``).
    overwrite
        If ``True``, re-create stacks even if the output file already exists.
        If ``False``, existing outputs are left in place and still reported in
        the returned list.
    verbose
        If ``True``, print progress and missing-input messages.
    add_class_map
        If ``True``, attempts to read a class mapping from the metadata tags of
        the first available input raster for a given stack. The function looks
        for dataset-level tag keys ``CLASS_MAP`` or ``CLASS-MAP`` containing a
        JSON dictionary. If found, the mapping is forwarded to
        ``stack_rasters_func`` so it can be written to the output (e.g., as tags
        on each layer/band). If no mapping is found (or parsing fails), stacking
        proceeds without class labels.

    Returns
    -------
    List[Path]
        Paths to stack outputs for each processed (band, hab, attempt, tf)
        combination. Includes paths that already existed when ``overwrite=False``.

    Notes
    -----
    - Missing per-year rasters are skipped; stacking proceeds as long as at least
      one input raster exists for the given (band, hab, attempt, tf).
    - The actual stacking and any per-band metadata writing is performed by
      ``stack_rasters_func``; this function only discovers inputs, builds output
      paths, and optionally forwards a parsed class map.
    """

    def _as_list(x: Union[str, Sequence[str]]) -> List[str]:
        return [x] if isinstance(x, str) else list(x)

    RF_results__dir = Path(RF_results__dir)
    bands = _as_list(band_selection)
    habs = _as_list(hab_selection)
    attempts = _as_list(train_split_attempt)

    created: List[Path] = []

    for band in bands:
        for hab in habs:
            for attempt in attempts:

                # NOTE: RF_paths currently returns [] when filename is None.
                # Use it only for directory creation OR update RF_paths to optionally return out_dir.
                stacked_rasters_dir = RF_results__dir / f"{band}_stacked" / hab
                stacked_rasters_dir.mkdir(parents=True, exist_ok=True)

                for tf in timeframe:
                    rasters_to_stack_list: List[Path] = []

                    for yr in years:
                        rel_file = Path(f"{yr}_{tf}") / (
                            f"RF_out__{band}__{hab}_{yr}_{tf}_{attempt}__{crs_tag}{suffix}"
                        )

                        raster_path = RF_results__dir / band / hab / rel_file

                        if raster_path.exists():
                            rasters_to_stack_list.append(raster_path)
                        elif verbose:
                            print(f"[skip missing] {raster_path}")

                    if len(rasters_to_stack_list) == 0:
                        if verbose:
                            print(f"[no inputs] band={band}, hab={hab}, attempt={attempt}, tf={tf}")
                        continue

                    out_name = f"stack_{band}__{hab}_{attempt}_{tf}{suffix}"
                    out_path = stacked_rasters_dir / out_name

                    if out_path.exists() and not overwrite:
                        if verbose:
                            print(f"[exists] {out_path}")
                        created.append(out_path)
                        continue

                    class_map = None
                    if add_class_map:
                        class_map = _read_class_map_from_raster(rasters_to_stack_list[0])
                        if verbose:
                            if class_map is None:
                                print(f"[no class map] {rasters_to_stack_list[0]}")
                            else:
                                print(f"[class map] using from {rasters_to_stack_list[0]}")

                    if verbose:
                        print(f"[stack] -> {out_path}  (n={len(rasters_to_stack_list)})")

                    stack_rasters_func(
                        raster_paths=rasters_to_stack_list,
                        out_dir=stacked_rasters_dir,
                        out_name=out_name,
                        band_names=list(years),
                        overwrite=overwrite,
                        rstr_classes=class_map,
                    )

                    created.append(out_path)

    return created



################################################################################
## STACK RASTERS OVER THE QUARTALS -> BATCH
################################################################################
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Union, Optional, Dict, Any

import rasterio

from paths._support.path_defining_funcs import RF_paths
from functions.raster_ops_funcs import stack_rasters_func


def stacking_timeframes_per_year_batch(
    RF_results__dir: Union[str, Path],
    years: Sequence[str],
    timeframes: Sequence[str],
    band_selection: Union[str, Sequence[str]],
    hab_selection: Union[str, Sequence[str]],
    train_split_attempt: Union[str, Sequence[str]],
    timeframe_descriptor: str,
    *,
    crs_tag: str = "UTM32631",
    suffix: str = "__rstr.tif",
    overwrite: bool = False,
    verbose: bool = True,
    add_class_map: bool = True,
) -> List[Path]:
    """
    Batch-stack Random Forest (RF) output rasters across multiple timeframes for each
    year, for all combinations of band, habitat, and train/split attempt.

    For each (band, hab, attempt, year) combination, this function searches for
    timeframe-specific RF output rasters under ``RF_results__dir`` and stacks the
    rasters that exist into a single multi-band GeoTIFF (one band per timeframe, in
    the order provided by ``timeframes``).

    Optionally, a class label mapping (e.g. ``{"0": "Open water", "1": "Remaining"}``)
    is copied from the input rasters into the produced stack if your
    ``stack_rasters_func`` supports it.

    Input layout (relative to ``RF_results__dir``)
    ----------------------------------------------
    Expected per-timeframe raster files are searched at::

        {band}/{hab}/{year}_{timeframe}/
            RF_out__{band}__{hab}_{year}_{timeframe}_{attempt}__{crs_tag}{suffix}

    Output layout
    -------------
    Stacks are written to::

        {band}_stacked/{hab}/

    with filenames::

        stack_{band}__{hab}_{attempt}_{timeframe_descriptor}_{year}{suffix}

    Example: ``stack_b28ndvwi__WD1_at1_Q1234_2018__rstr.tif``

    Parameters
    ----------
    RF_results__dir
        Root directory containing RF result folders.
    years
        Year labels to process. One output stack is created per year.
    timeframes
        Timeframe identifiers (e.g., seasons or quarters like 'Q1', 'Q2', 'Q3', 'Q4').
        For each timeframe ``tf``, inputs are searched in ``{year}_{tf}`` subfolders.
        Order determines the band order in the output stack.
    band_selection
        Band name(s) to process. Can be a single band string or a sequence of
        band strings.
    hab_selection
        Habitat class name(s) to process. Can be a single habitat string or a
        sequence of habitat strings.
    train_split_attempt
        Attempt identifier(s) to process (e.g., different train/test splits or
        run IDs). Can be a single string or a sequence of strings.
    timeframe_descriptor
        Descriptor to use in the output filename to indicate what timeframes are
        included (e.g., 'Q1234', 'all_quarters', 'seasonal').
    crs_tag
        CRS tag embedded in filenames (default ``"UTM32631"``).
    suffix
        Filename suffix for rasters (default ``"__rstr.tif"``).
    overwrite
        If ``True``, re-create stacks even if the output file already exists.
        If ``False``, existing outputs are left in place and still reported in
        the returned list.
    verbose
        If ``True``, print progress and missing-input messages.
    add_class_map
        If ``True``, attempts to read a class mapping from the metadata tags of
        the first available input raster for a given stack. The function looks
        for dataset-level tag keys ``CLASS_MAP`` or ``CLASS-MAP`` containing a
        JSON dictionary. If found, the mapping is forwarded to
        ``stack_rasters_func`` so it can be written to the output (e.g., as tags
        on each layer/band). If no mapping is found (or parsing fails), stacking
        proceeds without class labels.

    Returns
    -------
    List[Path]
        Paths to stack outputs for each processed (band, hab, attempt, year)
        combination. Includes paths that already existed when ``overwrite=False``.

    Notes
    -----
    - Missing per-timeframe rasters are skipped; stacking proceeds as long as at least
      one input raster exists for the given (band, hab, attempt, year).
    - The actual stacking and any per-band metadata writing is performed by
      ``stack_rasters_func``; this function only discovers inputs, builds output
      paths, and optionally forwards a parsed class map.
    
    Examples
    --------
    >>> results = stacking_timeframes_per_year_batch(
    ...     RF_results__dir='path/to/RF_results',
    ...     years=['2017', '2018', '2019'],
    ...     timeframes=['Q1', 'Q2', 'Q3', 'Q4'],
    ...     band_selection='b28ndvwi',
    ...     hab_selection='WD1',
    ...     train_split_attempt='at1',
    ...     timeframe_descriptor='Q1234',
    ... )
    # Creates:
    # - stack_b28ndvwi__WD1_at1_Q1234_2017__rstr.tif
    # - stack_b28ndvwi__WD1_at1_Q1234_2018__rstr.tif
    # - stack_b28ndvwi__WD1_at1_Q1234_2019__rstr.tif
    """

    def _as_list(x: Union[str, Sequence[str]]) -> List[str]:
        return [x] if isinstance(x, str) else list(x)

    RF_results__dir = Path(RF_results__dir)
    bands = _as_list(band_selection)
    habs = _as_list(hab_selection)
    attempts = _as_list(train_split_attempt)

    created: List[Path] = []

    for band in bands:
        for hab in habs:
            for attempt in attempts:

                # Create output directory
                stacked_rasters_dir = RF_results__dir / f"{band}_stacked" / hab
                stacked_rasters_dir.mkdir(parents=True, exist_ok=True)

                # Process each year separately
                for yr in years:
                    rasters_to_stack_list: List[Path] = []

                    # Collect rasters for each timeframe for this year
                    for tf in timeframes:
                        rel_file = Path(f"{yr}_{tf}") / (
                            f"RF_out__{band}__{hab}_{yr}_{tf}_{attempt}__{crs_tag}{suffix}"
                        )

                        raster_path = RF_results__dir / band / hab / rel_file

                        if raster_path.exists():
                            rasters_to_stack_list.append(raster_path)
                        elif verbose:
                            print(f"[skip missing] {raster_path}")

                    if len(rasters_to_stack_list) == 0:
                        if verbose:
                            print(f"[no inputs] band={band}, hab={hab}, attempt={attempt}, year={yr}")
                        continue

                    # Output filename: stack_{band}__{hab}_{attempt}_{timeframe_descriptor}_{year}__rstr.tif
                    out_name = f"stack_{band}__{hab}_{attempt}_{timeframe_descriptor}__{yr}{suffix}"
                    out_path = stacked_rasters_dir / out_name

                    if out_path.exists() and not overwrite:
                        if verbose:
                            print(f"[exists] {out_path}")
                        created.append(out_path)
                        continue

                    class_map = None
                    if add_class_map:
                        class_map = _read_class_map_from_raster(rasters_to_stack_list[0])
                        if verbose:
                            if class_map is None:
                                print(f"[no class map] {rasters_to_stack_list[0]}")
                            else:
                                print(f"[class map] using from {rasters_to_stack_list[0]}")

                    if verbose:
                        print(f"[stack] -> {out_path}  (n={len(rasters_to_stack_list)} timeframes for year {yr})")

                    stack_rasters_func(
                        raster_paths=rasters_to_stack_list,
                        out_dir=stacked_rasters_dir,
                        out_name=out_name,
                        band_names=list(timeframes[:len(rasters_to_stack_list)]),  # Use timeframe names as band names
                        overwrite=overwrite,
                        rstr_classes=class_map,
                    )

                    created.append(out_path)

    return created