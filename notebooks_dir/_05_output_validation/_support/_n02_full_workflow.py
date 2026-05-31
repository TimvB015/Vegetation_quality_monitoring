###############################################################################
## FULL WORKFLOW FOR CARTO
###############################################################################
from notebooks_dir._05_output_validation._support._n02_funcs import (
    carto_performance_df_finder,
    build_class_display_names_from_habitat_df,
    build_class_colors_from_habitat_df,
    plot_carto_acc_f1_heatmaps,
)

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd


def make_and_save_carto_performance_heatmap(
    carto_results_dfs_dir,
    carto_heatmaps_dir,
    hab_selection: str,
    train_split_attempt: str,
    habitat_reference_df: pd.DataFrame,
    years=None,                 # <- NEW (None or list of years)
    overwrite: bool = False,
    dpi: int = 300,
):
    """
    Create (or reuse) the CARTO model performance heatmap and return the plot.

    Workflow
    --------
    1) Locate the CARTO performance dataframe pickle in `carto_results_dfs_dir`
       using `carto_performance_df_finder(...)`.
    2) Load it with `pd.read_pickle`.
    3) Build `class_colors` and `class_display_names` from `habitat_reference_df`.
    4) Create the two-panel (Accuracy/F1) heatmap with `plot_carto_acc_f1_heatmaps(...)`,
       optionally forcing a specific set of years via `years=...`.
    5) Save the figure as a PNG into `carto_heatmaps_dir` named:
       `carto_model_performance_{hab_selection}_{train_split_attempt}.png`

    Years behavior
    --------------
    - If `years=None` (default): plot only the years present in the performance dataframe.
    - If `years` is a list/sequence of ints: plot exactly those years (in that order),
      including years with no data (they will appear as empty columns).

    Caching behavior
    ----------------
    If `overwrite=False` and the PNG already exists, the PNG is *imported* and
    displayed (as an image-only figure) instead of being re-created. In that
    case, a message is printed: "Imported existing heatmap: <path>".

    Parameters
    ----------
    carto_results_dfs_dir :
        Directory containing the performance result `.pkl` files.
    carto_heatmaps_dir :
        Directory where the heatmap image will be saved.
    hab_selection :
        Habitat selection key, e.g. "WD1".
    train_split_attempt :
        Split attempt id, e.g. "at1".
    habitat_reference_df :
        Reference dataframe used to derive display names and class colors.
    years :
        None or a sequence of years (ints) to force on the x-axis.
    overwrite :
        If True, always regenerate and overwrite the PNG.
    dpi :
        DPI used when saving the PNG.

    Returns
    -------
    fig :
        A Matplotlib Figure. If the file was imported, this is a figure that
        displays the saved PNG. If created, this is the original heatmap figure.
    """
    heatmaps_dir = Path(carto_heatmaps_dir)
    heatmaps_dir.mkdir(parents=True, exist_ok=True)

    out_path = heatmaps_dir / f"carto_model_performance_{hab_selection}_{train_split_attempt}.png"

    # If cached and not overwriting: import the png and display it
    if out_path.exists() and not overwrite:
        print(f"Imported existing heatmap: {out_path}")
        img = mpimg.imread(out_path)

        fig, ax = plt.subplots(figsize=(img.shape[1] / 100, img.shape[0] / 100), dpi=100)
        ax.imshow(img)
        ax.axis("off")
        fig.tight_layout(pad=0)
        return fig

    # Find + load performance df
    carto_performance_df_path = carto_performance_df_finder(
        df_dir=carto_results_dfs_dir,
        hab_selection=hab_selection,
        train_split_attempt=train_split_attempt,
    )
    carto_performance_df = pd.read_pickle(carto_performance_df_path)

    # Build colors + display names
    class_colors = build_class_colors_from_habitat_df(
        colors_df=habitat_reference_df,
        division_col=f"{hab_selection}_division",
        color_col=f"{hab_selection}_color",
    )
    class_display_names = build_class_display_names_from_habitat_df(
        colors_df=habitat_reference_df,
        division_col=f"{hab_selection}_division",
    )

    # Create plot
    fig, _ = plot_carto_acc_f1_heatmaps(
        df=carto_performance_df,
        title=f"Carto performance {hab_selection} {train_split_attempt}",
        class_display_names=class_display_names,
        class_colors=class_colors,
        years=years,   # <- NEW
    )

    # Save
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    return fig



###############################################################################
## FULL WORKFLOW FOR RF YEARLY AVG
###############################################################################
from paths._support.path_defining_funcs import RF_paths

from notebooks_dir._05_output_validation._support._n02_funcs import (
    RF_performance_df_finder,
    read_RF_performance_csv,
    resample_RF_performance_to_year_avgs_df,
    build_class_display_names_from_habitat_df,
    build_class_colors_from_habitat_df,
    plot_RF_performance_acc_f1_heatmaps,
)

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from typing import Optional, Sequence


def make_and_save_RF_year_avg_performance_heatmap(
    RF_results__dir,
    RF_heatmaps_dir,
    train_year: str,
    model_selection: str,
    band_selection: str,
    hab_selection: str,
    train_split_attempt: str,
    habitat_reference_df: pd.DataFrame,
    years: Optional[Sequence[int]] = None,  # <- NEW (None or explicit year list)
    overwrite: bool = False,
    dpi: int = 300,
):
    """
    Create (or reuse) an RF year-average performance heatmap (Accuracy/F1), optionally
    forcing a specific set of year columns.

    Workflow
    --------
    1) Locate RF performance CSV using ``RF_performance_df_finder(...)``.
    2) Load it with ``read_RF_performance_csv(...)``.
    3) Resample to year averages using ``resample_RF_performance_to_year_avgs_df(...)``.
    4) Build `class_colors` and `class_display_names` from `habitat_reference_df`.
    5) Plot via ``plot_RF_performance_acc_f1_heatmaps(..., years=years)``.
    6) Save PNG into `RF_heatmaps_dir` using:
       ``RF_model_performance__{hab_selection}_{train_split_attempt}__year_avg.png``.

    Year handling
    -------------
    - If `years=None` (default): the plot uses only the years present in the resampled dataframe.
    - If `years` is provided (sequence of ints): the plot includes exactly those years
      (in that order). Years without data will appear as empty/NaN columns.

    Caching behavior
    ----------------
    If `overwrite=False` and the PNG already exists, it is imported and returned as an
    image-only figure, and a message is printed: ``Imported existing heatmap: <path>``.

    Parameters
    ----------
    RF_results__dir :
        Directory containing RF result files.
    RF_heatmaps_dir :
        Output directory where the heatmap PNG will be written.
    train_year, model_selection, band_selection, hab_selection, train_split_attempt : str
        Identifiers used to locate the performance file and build output filenames.
    habitat_reference_df : pd.DataFrame
        Reference dataframe used to derive display names and class colors.
    years : Optional[Sequence[int]], default None
        Optional explicit list of years to force on the x-axis. Strings like "2020" are accepted.
    overwrite : bool, default False
        If True, always regenerate and overwrite the PNG.
    dpi : int, default 300
        DPI used when saving the PNG.

    Returns
    -------
    fig : matplotlib.figure.Figure
        If imported, a figure displaying the saved PNG. If created, the original heatmap figure.
    """
    # normalize years argument (accept strings like "2020")
    if years is not None:
        years = [int(y) for y in years]

    out_path = RF_paths(
        base_dir=RF_heatmaps_dir,
        band_selection=band_selection,
        hab_selection=hab_selection,
        filename=f"RF_model_performance__{hab_selection}_{train_split_attempt}__year_avg.png"
    )[0]

    # Cache import
    if out_path.exists() and not overwrite:
        print(f"Imported existing heatmap: {out_path}")
        img = mpimg.imread(out_path)
        fig, ax = plt.subplots(figsize=(img.shape[1] / 100, img.shape[0] / 100), dpi=100)
        ax.imshow(img)
        ax.axis("off")
        fig.tight_layout(pad=0)
        return fig

    # Find + load performance df
    performance_df_path = RF_performance_df_finder(
        RF_results__dir=RF_results__dir,
        train_year=train_year,
        model_selection=model_selection,
        band_selection=band_selection,
        hab_selection=hab_selection,
        train_split_attempt=train_split_attempt,
    )
    performance_df = read_RF_performance_csv(performance_df_path)

    # Resample to year averages
    resampled_year__df = resample_RF_performance_to_year_avgs_df(
        df=performance_df,
        adjust_weighted=None,
        year_col="clas_year",
        timeframe_col="timeframe",
    )

    # Colors + display names
    class_colors = build_class_colors_from_habitat_df(
        colors_df=habitat_reference_df,
        division_col=f"{hab_selection}_division",
        color_col=f"{hab_selection}_color",
    )
    class_display_names = build_class_display_names_from_habitat_df(
        colors_df=habitat_reference_df,
        division_col=f"{hab_selection}_division",
    )

    # Plot
    fig, _ = plot_RF_performance_acc_f1_heatmaps(
        df=resampled_year__df,
        title=f"RF performance {hab_selection} {train_split_attempt} year avg [{band_selection}]",
        class_display_names=class_display_names,
        class_colors=class_colors,
        years=years,  # <- NEW
    )

    # Save
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    return fig



###############################################################################
## FULL WORKFLOW FOR RF TIMEFRAMES
###############################################################################
from notebooks_dir._05_output_validation._support._n02_funcs import (
    RF_performance_df_finder,
    read_RF_performance_csv,
    resample_RF_performance_to_timeframe_df,
    build_class_display_names_from_habitat_df,
    build_class_colors_from_habitat_df,
    plot_RF_performance_acc_f1_heatmaps,
)

from paths._support.path_defining_funcs import RF_paths

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from typing import Optional, Sequence


def make_and_save_RF_timeframe_performance_heatmap(
    RF_results__dir,
    RF_heatmaps_dir,
    train_year: str,
    model_selection: str,
    band_selection: str,
    hab_selection: str,
    train_split_attempt: str,
    habitat_reference_df: pd.DataFrame,
    timeframe: Optional[list[str]] = None,
    years: Optional[Sequence[int]] = None,
    overwrite: bool = False,
    dpi: int = 300,
):
    """
    Create (or reuse) RF performance heatmaps per timeframe, optionally forcing year columns.

    For each timeframe (default: Q1..Q4):
      1) Load the RF performance dataframe.
      2) Resample to a timeframe-specific dataframe via
         ``resample_RF_performance_to_timeframe_df(..., timeframe=<tf>)``.
      3) Plot a two-panel (Accuracy/F1) heatmap via
         ``plot_RF_performance_acc_f1_heatmaps(..., years=years)``.
      4) Save one PNG per timeframe.

    Year handling
    -------------
    - If `years=None` (default): each timeframe plot uses only the years present in the
      resampled timeframe dataframe.
    - If `years` is provided (sequence of ints): each timeframe plot will include exactly
      those years (in that order). Years without data will still appear as empty/NaN columns.

    Caching behavior (per timeframe PNG)
    ------------------------------------
    If `overwrite=False` and a PNG already exists for a timeframe, the PNG is imported and
    returned as an image-only figure instead of being re-created. A message is printed:
    ``Imported existing heatmap: <path>``.

    Parameters
    ----------
    RF_results__dir :
        Directory containing RF result files.
    RF_heatmaps_dir :
        Output directory where heatmap PNGs will be written.
    train_year, model_selection, band_selection, hab_selection, train_split_attempt : str
        Identifiers used to locate the performance file and build output filenames.
    habitat_reference_df : pd.DataFrame
        Reference dataframe used to derive display names and class colors.
    timeframe : Optional[list[str]], default None
        Timeframes to generate, e.g. ["Q1","Q2","Q3","Q4"]. If None, defaults to Q1..Q4.
    years : Optional[Sequence[int]], default None
        Optional explicit list of years to force on the x-axis for all timeframe plots.
    overwrite : bool, default False
        If True, always regenerate and overwrite PNGs.
    dpi : int, default 300
        DPI used when saving the PNGs.

    Returns
    -------
    figs_by_timeframe : dict[str, matplotlib.figure.Figure]
        Mapping from timeframe (e.g. "Q1") to the corresponding Matplotlib Figure.
        If a PNG was imported, the returned figure displays that PNG.
    """
    if timeframe is None:
        timeframe = ["Q1", "Q2", "Q3", "Q4"]

    # normalize years argument (accept strings like "2020")
    if years is not None:
        years = [int(y) for y in years]

    # Find + load performance df once (same for all timeframes)
    performance_df_path = RF_performance_df_finder(
        RF_results__dir=RF_results__dir,
        train_year=train_year,
        model_selection=model_selection,
        band_selection=band_selection,
        hab_selection=hab_selection,
        train_split_attempt=train_split_attempt,
    )
    performance_df = read_RF_performance_csv(performance_df_path)

    # Colors + display names once
    class_colors = build_class_colors_from_habitat_df(
        colors_df=habitat_reference_df,
        division_col=f"{hab_selection}_division",
        color_col=f"{hab_selection}_color",
    )
    class_display_names = build_class_display_names_from_habitat_df(
        colors_df=habitat_reference_df,
        division_col=f"{hab_selection}_division",
    )

    figs_by_timeframe: dict[str, plt.Figure] = {}

    for tf in timeframe:
        out_path = RF_paths(
            base_dir=RF_heatmaps_dir,
            band_selection=band_selection,
            hab_selection=hab_selection,
            filename=f"RF_model_performance__{hab_selection}_{train_split_attempt}__{band_selection}_{tf}.png"
        )[0]

        # Cache import
        if out_path.exists() and not overwrite:
            print(f"Imported existing heatmap: {out_path}")
            img = mpimg.imread(out_path)
            fig, ax = plt.subplots(figsize=(img.shape[1] / 100, img.shape[0] / 100), dpi=100)
            ax.imshow(img)
            ax.axis("off")
            fig.tight_layout(pad=0)
            figs_by_timeframe[tf] = fig
            continue

        # Resample for this timeframe
        timeframe_df = resample_RF_performance_to_timeframe_df(
            df=performance_df,
            timeframe=tf,
            year_col="clas_year",
        )

        # Plot (pass years through)
        fig, _ = plot_RF_performance_acc_f1_heatmaps(
            df=timeframe_df,
            title=f"RF performance {hab_selection} {train_split_attempt} {tf} [{band_selection}]",
            class_display_names=class_display_names,
            class_colors=class_colors,
            years=years,  # <- NEW
        )

        # Save
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        figs_by_timeframe[tf] = fig

    return figs_by_timeframe