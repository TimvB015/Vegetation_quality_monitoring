from __future__ import annotations

###############################################################################
## FINDING THE CARTO OUTPUT FILES
###############################################################################
from pathlib import Path
from typing import Union


def carto_performance_df_finder(
    df_dir: Union[str, Path],
    hab_selection: str,
    train_split_attempt: Union[str, int],
) -> Path:
    """
    Build the path to the performance dataframe pickle file.

    Returns:
        df_dir / f"classification_results_stats_{hab_selection}_{train_split_attempt}.pkl"
    """
    df_dir = Path(df_dir)
    return df_dir / f"classification_results_stats__{hab_selection}_{train_split_attempt}.pkl"



###############################################################################
## FINDING THE RF OUTPUT FILES
###############################################################################
from pathlib import Path
from typing import Union


def RF_performance_df_finder(
    RF_results__dir: Union[str, Path],
    train_year: Union[str, int],
    model_selection: str,
    band_selection: str,
    hab_selection: str,
    train_split_attempt: Union[str, int],
) -> Path:
    """
    Build the path to the RF validation performance dataframe pickle file.

    Returns:
        RF_results__dir / band_selection / hab_selection / "performance_dfs" /
        f"RF_validation__{band_selection}__{hab_selection}__train{train_year}__{model_selection}__{train_split_attempt}__UTM32631.pkl"
    """
    RF_results__dir = Path(RF_results__dir)
    return (
        RF_results__dir
        / str(band_selection)
        / str(hab_selection)
        / "performance_dfs"
        / f"RF_validation__{band_selection}__{hab_selection}_{train_split_attempt}__train{train_year}__{model_selection}__UTM32631.csv"
    )



###############################################################################
## READ PERFORMANCE CSV
###############################################################################
from pathlib import Path
from typing import Union
import pandas as pd


def read_RF_performance_csv(csv_path: Union[str, Path]) -> pd.DataFrame:
    """
    Read an RF performance CSV and return it as a DataFrame.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        pandas.DataFrame
    """
    csv_path = Path(csv_path)
    return pd.read_csv(csv_path)



###############################################################################
## RESAMPLE RF PERFORMANCE TO YEARLY AVERAGES
###############################################################################
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

def resample_RF_performance_to_year_avgs_df(
    df: pd.DataFrame,
    adjust_weighted: Optional[Union[List[float], Dict[str, float]]] = None,
    year_col: str = "clas_year",
    timeframe_col: str = "timeframe",
    train_year_col: str = "train_year",   # not used for grouping; only for output ordering/keeping
    round_decimals: int = 2,
) -> pd.DataFrame:
    """
    Resample RF performance rows (clas_year x timeframe) to yearly rows using weighted averages.

    Output formatting:
      - train_year (if present) -> int (nullable Int64) and placed as first column
      - clas_year -> int (nullable Int64)
      - all n / n_* columns -> int (nullable Int64)
      - all other numeric columns -> rounded to `round_decimals`

    Note: aggregation is done per `clas_year` only. If `train_year` exists in the input,
    the first value per `clas_year` is kept.
    """
    for col in (year_col, timeframe_col):
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}'.")

    d = df.copy()

    # --- detect timeframe type and canonical order ---
    tf_raw = d[timeframe_col].dropna().unique().tolist()
    tf_str = [str(x).strip() for x in tf_raw]

    def _is_quartal_label(s: str) -> bool:
        return s.upper() in {"Q1", "Q2", "Q3", "Q4"}

    quartal_like = (len(tf_str) > 0) and all(_is_quartal_label(s) for s in tf_str)

    if quartal_like:
        tf_order = ["Q1", "Q2", "Q3", "Q4"]
        expected_k = 4
        d[timeframe_col] = d[timeframe_col].astype(str).str.strip().str.upper()
    else:
        tf_num = pd.to_numeric(d[timeframe_col], errors="coerce")
        if tf_num.isna().any():
            raise ValueError(
                f"Cannot infer month-like timeframes from '{timeframe_col}'. "
                "Expected quartals 'Q1'..'Q4' or months as numbers."
            )
        d[timeframe_col] = tf_num.astype(int)
        vals = sorted(set(d[timeframe_col].dropna().astype(int).tolist()))
        tf_order = list(range(0, 12)) if 0 in vals else list(range(1, 13))
        expected_k = 12

    # --- weights aligned to tf_order ---
    def _normalize_weights(adjust):
        if adjust is None:
            w = np.ones(expected_k, dtype=float) / expected_k
            return {tf_order[i]: float(w[i]) for i in range(expected_k)}

        if isinstance(adjust, dict):
            wmap = {}
            for k, v in adjust.items():
                kk = str(k).strip()
                kk = kk.upper() if quartal_like else int(float(kk))
                wmap[kk] = float(v)

            missing = [t for t in tf_order if t not in wmap]
            extra = [t for t in wmap.keys() if t not in set(tf_order)]
            if missing or extra:
                raise ValueError(f"adjust_weighted keys mismatch. Missing={missing}, extra={extra}")

            s = float(sum(wmap.values()))
            if not np.isclose(s, 1.0):
                raise ValueError(f"Weights must sum to 1. Got {s}.")
            return wmap

        wlist = [float(x) for x in list(adjust)]
        if len(wlist) != expected_k:
            raise ValueError(f"Expected {expected_k} weights, got {len(wlist)}.")
        s = float(sum(wlist))
        if not np.isclose(s, 1.0):
            raise ValueError(f"Weights must sum to 1. Got {s}.")
        return {tf_order[i]: wlist[i] for i in range(expected_k)}

    wmap = _normalize_weights(adjust_weighted)
    d["_w"] = d[timeframe_col].map(wmap).astype(float)

    group_cols = [year_col]

    numeric_cols = d.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in {"_w"}]

    def _is_n_col(c: str) -> bool:
        return c == "n" or c.startswith("n_")

    n_cols = [c for c in numeric_cols if _is_n_col(c)]
    metric_cols = [c for c in numeric_cols if c not in n_cols]

    def _wmean(x: pd.Series, w: pd.Series) -> float:
        m = x.notna() & w.notna()
        if not bool(m.any()):
            return np.nan
        ww = w[m].to_numpy(dtype=float)
        xx = x[m].to_numpy(dtype=float)
        sww = ww.sum()
        if sww == 0:
            return np.nan
        return float((xx * ww).sum() / sww)

    out_rows = []
    for (clas_year,), g in d.groupby(group_cols, dropna=False):
        row = {year_col: clas_year}
        w = g["_w"]

        # keep train_year (first) if present
        if train_year_col in g.columns:
            row[train_year_col] = g[train_year_col].iloc[0]

        # sum n-columns
        for c in n_cols:
            row[c] = float(pd.to_numeric(g[c], errors="coerce").fillna(0).mean())

        # weighted mean for other numeric metrics
        for c in metric_cols:
            row[c] = _wmean(pd.to_numeric(g[c], errors="coerce"), w)

        # carry over other non-numeric columns (first)
        for c in g.columns:
            if c in group_cols or c in {timeframe_col, "_w"} or c in n_cols or c in metric_cols or c == train_year_col:
                continue
            row[c] = g[c].iloc[0]

        out_rows.append(row)

    out = pd.DataFrame(out_rows).sort_values(group_cols).reset_index(drop=True)

    # --- formatting: ints ---
    out[year_col] = pd.to_numeric(out[year_col], errors="coerce").round().astype("Int64")
    if train_year_col in out.columns:
        out[train_year_col] = pd.to_numeric(out[train_year_col], errors="coerce").round().astype("Int64")

    for c in n_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round().astype("Int64")

    # --- formatting: rounding other numeric cols ---
    num_cols_out = out.select_dtypes(include=[np.number]).columns.tolist()
    num_cols_out = [c for c in num_cols_out if c not in {year_col, train_year_col, *n_cols}]
    out[num_cols_out] = out[num_cols_out].round(round_decimals)

    # --- reorder columns: train_year first (if present), then clas_year, then rest ---
    front = [c for c in [train_year_col, year_col] if c in out.columns]
    rest = [c for c in out.columns if c not in front]
    out = out[front + rest]

    return out



###############################################################################
## RESAMPLE RF PERFORMANCE DF TO TIMEFRAME
###############################################################################
import numpy as np
import pandas as pd


def resample_RF_performance_to_timeframe_df(
    df: pd.DataFrame,
    timeframe: str,
    year_col: str = "clas_year",
    timeframe_col: str = "timeframe",
    round_decimals: int = 2,
) -> pd.DataFrame:
    """
    Filter/resample RF performance data to a single timeframe (e.g. 'Q1').

    Keeps one row per clas_year for the requested timeframe.
    If duplicates exist for the same (clas_year, timeframe), numeric columns are averaged
    (and n/n_* columns are summed).

    Output formatting:
      - if a 'train_year' column exists: it is kept, cast to Int64, and placed first
      - clas_year cast to Int64
      - timeframe column is kept (constant for all rows)
      - n / n_* columns cast to Int64
      - all other numeric columns rounded to `round_decimals`
    """
    for col in (year_col, timeframe_col):
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}'.")

    d = df.copy()

    # normalize timeframe for matching
    tf_req_raw = str(timeframe).strip()
    if d[timeframe_col].dtype == object:
        d[timeframe_col] = d[timeframe_col].astype(str).str.strip()

    # quartal vs month-like matching
    quartals = {"Q1", "Q2", "Q3", "Q4"}
    if tf_req_raw.upper() in quartals:
        tf_req = tf_req_raw.upper()
        d[timeframe_col] = d[timeframe_col].astype(str).str.strip().str.upper()
    else:
        tf_req_num = pd.to_numeric(pd.Series([tf_req_raw]), errors="coerce").iloc[0]
        if pd.isna(tf_req_num):
            raise ValueError("timeframe must be like 'Q1'..'Q4' or a numeric month.")
        tf_req = int(tf_req_num)
        d[timeframe_col] = pd.to_numeric(d[timeframe_col], errors="coerce").astype("Int64")

    d = d[d[timeframe_col] == tf_req].copy()

    group_cols = [year_col]
    numeric_cols = d.select_dtypes(include=[np.number]).columns.tolist()

    def _is_n_col(c: str) -> bool:
        return c == "n" or c.startswith("n_")

    n_cols = [c for c in numeric_cols if _is_n_col(c)]
    metric_cols = [c for c in numeric_cols if c not in n_cols]

    # aggregate if duplicates exist
    agg = {timeframe_col: "first"}  # keep timeframe column
    agg.update({c: "sum" for c in n_cols})
    agg.update({c: "mean" for c in metric_cols})

    # keep train_year if present (first)
    if "train_year" in d.columns:
        agg["train_year"] = "first"

    # other non-numeric: take first
    protected = set(group_cols + [timeframe_col] + numeric_cols + (["train_year"] if "train_year" in d.columns else []))
    other_cols = [c for c in d.columns if c not in protected]
    for c in other_cols:
        agg[c] = "first"

    out = (
        d.groupby(group_cols, dropna=False, as_index=False)
         .agg(agg)
         .sort_values(group_cols)
         .reset_index(drop=True)
    )

    # formatting: ints
    out[year_col] = pd.to_numeric(out[year_col], errors="coerce").round().astype("Int64")
    if "train_year" in out.columns:
        out["train_year"] = pd.to_numeric(out["train_year"], errors="coerce").round().astype("Int64")

    for c in n_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round().astype("Int64")

    # formatting: other numeric columns rounded
    num_cols_out = out.select_dtypes(include=[np.number]).columns.tolist()
    exclude = {year_col, *n_cols}
    if "train_year" in out.columns:
        exclude.add("train_year")
    num_cols_out = [c for c in num_cols_out if c not in exclude]
    out[num_cols_out] = out[num_cols_out].round(round_decimals)

    # reorder columns: train_year first (if present), then clas_year, then timeframe, then rest
    front = [c for c in ["train_year", year_col, timeframe_col] if c in out.columns]
    rest = [c for c in out.columns if c not in front]
    out = out[front + rest]

    return out



###############################################################################
## NAMESTRIP FUNCTION
###############################################################################
import numpy as np

def name_strip(x: str) -> str:
    """Lowercase and remove underscores/spaces (robust for None/NaN)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip().lower()
    s = s.replace("_", "").replace(" ", "")
    return s




###############################################################################
## BUILD CLASS NAMES FROM HABITAT DF
###############################################################################
from typing import Dict
import pandas as pd


def build_class_display_names_from_habitat_df(
    colors_df: pd.DataFrame,
    division_col: str,
) -> Dict[str, str]:
    """
    Returns: {stripped_division_name: original_division_name_as_in_colors_df}
    Example: {"openwater": "Open water", "wetnature": "Wet Nature"}
    """
    tmp = colors_df[[division_col]].copy().dropna()
    tmp["_key"] = tmp[division_col].map(name_strip)
    tmp = tmp[tmp["_key"] != ""]
    tmp = tmp.drop_duplicates(subset=["_key"])  # keep first occurrence (as in your reference df)
    return tmp.set_index("_key")[division_col].to_dict()



###############################################################################
## BUILD CLASS COLORS FROM HABITAT DF
###############################################################################
import pandas as pd
from typing import Dict


def build_class_colors_from_habitat_df(
    colors_df: pd.DataFrame,
    division_col: str,
    color_col: str,
) -> Dict[str, str]:
    """
    Returns: dict {stripped_division_name: hex_color}

    Example:
      "Open water" -> key "openwater"
      "Wet Nature" -> key "wetnature"
    """
    tmp = colors_df[[division_col, color_col]].copy()
    tmp = tmp.dropna(subset=[division_col, color_col])

    tmp["_key"] = tmp[division_col].map(name_strip)

    # keep only non-empty keys
    tmp = tmp[tmp["_key"] != ""]

    # one color per unique stripped name (take first occurrence)
    tmp = tmp.drop_duplicates(subset=["_key"])

    return tmp.set_index("_key")[color_col].to_dict()



###############################################################################
## CARTO HEATMAP PLOTTER
###############################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from typing import Dict, Tuple, Optional, Sequence


def plot_carto_acc_f1_heatmaps(
    df: pd.DataFrame,
    title: str,
    class_display_names: Dict[str, str],
    class_colors: Dict[str, str],
    years: Optional[Sequence[int]] = None,
):
    """
    Plot CARTO model performance heatmaps (Accuracy and F1) per class over years.

    Expects the input dataframe `df` to have:
      - a year column named ``class_year``
      - per-class metric columns in canonical form:
          * ``acc_<class>``
          * ``f1_<class>``
          * ``n_<class>``   (number of validation samples)
    Classes are detected from ``n_<class>`` columns that also have corresponding
    ``acc_<class>`` and ``f1_<class>`` columns.

    Internally, class names are converted to *stripped keys* using ``name_strip``
    (e.g. ``"open_water" -> "openwater"``). The dictionaries ``class_display_names``
    and ``class_colors`` must be keyed by these stripped keys.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing year + per-class metric columns.
    title : str
        Figure title (suptitle).
    class_display_names : Dict[str, str]
        Mapping from stripped class key -> display label used on the y-axis.
    class_colors : Dict[str, str]
        Mapping from stripped class key -> hex color used to color the cells.
    years : Optional[Sequence[int]], default None
        Controls which year-columns appear in the heatmap:
          - If None: plot only the years present in ``df["class_year"]``.
          - If a sequence of ints: plot exactly those years (in that order).
            Years not present in the input data are included as empty columns
            (their cells remain NaN/blank).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Matplotlib figure.
    (ax_acc, ax_f1) : Tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]
        The axes for the Accuracy and F1 heatmaps.
    """
    # --- hardcoded settings ---
    year_col = "clas_year"
    acc_prefix, f1_prefix, n_prefix = "acc", "f1", "n"
    alpha_min = 0.08
    facecolor_empty = "#f2f2f2"
    edgecolor = "white"
    linewidth = 1.2
    text_size = 8

    d = df.copy()
    d[year_col] = pd.to_numeric(d[year_col], errors="coerce")
    d = d[d[year_col].notna()].sort_values(year_col)
    d[year_col] = d[year_col].astype(int)

    # detect classes from n_<cls> that also have acc_<cls> and f1_<cls>
    raw_classes = []
    for c in d.columns:
        if c.startswith(f"{n_prefix}_"):
            cls = c[len(f"{n_prefix}_"):]  # e.g. "open_water"
            if f"{acc_prefix}_{cls}" in d.columns and f"{f1_prefix}_{cls}" in d.columns:
                raw_classes.append(cls)

    if not raw_classes:
        raise ValueError("No classes found. Need columns like n_<cls>, acc_<cls>, f1_<cls>.")

    # map raw -> stripped key
    raw_to_key = {rc: name_strip(rc) for rc in sorted(set(raw_classes))}
    keys_in_df = sorted(set(raw_to_key.values()))

    def _display_for_key(k: str) -> str:
        return class_display_names.get(k, k)

    # colors
    missing = [k for k in keys_in_df if k not in class_colors or not class_colors[k]]
    if missing:
        key_to_raw = {}
        for raw, key in raw_to_key.items():
            key_to_raw.setdefault(key, []).append(raw)

        details = "\n".join(
            f"  - '{k}' (from df classes: {sorted(key_to_raw.get(k, []))})"
            for k in missing
        )
        raise ValueError(
            "No color defined for one or more classes found in df.\n"
            "Missing stripped keys:\n"
            f"{details}\n\n"
            "Define these in class_colors (keys must be stripped with name_strip)."
        )

    # normalize years argument
    if years is not None:
        years = [int(y) for y in years]

    # build matrices (rows are stripped keys)
    def _pivot(metric_prefix: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        rows = []
        for raw_cls, key in raw_to_key.items():
            tmp = d[[year_col, f"{metric_prefix}_{raw_cls}", f"{n_prefix}_{raw_cls}"]].copy()
            tmp.columns = [year_col, "score", "n"]
            tmp["key"] = key
            rows.append(tmp)

        long = pd.concat(rows, ignore_index=True)
        long["n"] = pd.to_numeric(long["n"], errors="coerce").fillna(0)

        # mask where no validation data
        long.loc[long["n"] <= 0, "score"] = np.nan

        # if multiple raw classes collapse to same stripped key, average score and sum n
        score_mat = long.pivot_table(index="key", columns=year_col, values="score", aggfunc="mean")
        n_mat = long.pivot_table(index="key", columns=year_col, values="n", aggfunc="sum")

        # add empty year-columns if requested
        if years is not None:
            score_mat = score_mat.reindex(columns=years)
            n_mat = n_mat.reindex(columns=years)

        score_mat = score_mat.reindex(keys_in_df)
        n_mat = n_mat.reindex(keys_in_df)
        return score_mat, n_mat

    acc_mat, n_mat = _pivot(acc_prefix)
    f1_mat, _ = _pivot(f1_prefix)

    years_used = acc_mat.columns.tolist()
    figsize = (1.2 + 0.9 * max(len(years_used), 1), 2.6 + 0.55 * len(keys_in_df))

    fig, (ax_acc, ax_f1) = plt.subplots(
        nrows=2, ncols=1, figsize=figsize, sharex=True, sharey=True,
        constrained_layout=True, gridspec_kw={"hspace": 0.06}
    )

    def _draw(ax, mat: pd.DataFrame, subtitle: str):
        ax.set_title(subtitle, loc="center", fontsize=12, fontweight="bold")
        ax.set_facecolor(facecolor_empty)

        for i, key in enumerate(mat.index):
            rgb = mcolors.to_rgb(class_colors[key])
            for j, _yr in enumerate(mat.columns):
                val = mat.iat[i, j]
                nval = n_mat.iat[i, j] if (i < n_mat.shape[0] and j < n_mat.shape[1]) else np.nan

                if pd.isna(val):
                    continue

                a = float(np.clip(val, 0.0, 1.0))
                a = max(alpha_min, a)

                rect = plt.Rectangle(
                    (j - 0.5, i - 0.5), 1.0, 1.0,
                    facecolor=rgb, alpha=a,
                    edgecolor=edgecolor, linewidth=linewidth
                )
                ax.add_patch(rect)

                tcol = "white" if a >= 0.55 else "#1a1a1a"
                n_txt = "" if pd.isna(nval) else f"\n(n={int(nval)})"
                ax.text(
                    j, i, f"{val:.2f}{n_txt}",
                    ha="center", va="center",
                    fontsize=text_size, color=tcol
                )

        ax.set_xlim(-0.5, mat.shape[1] - 0.5)
        ax.set_ylim(mat.shape[0] - 0.5, -0.5)

        ax.set_yticks(np.arange(mat.shape[0]))
        ax.set_yticklabels([_display_for_key(k) for k in mat.index])
        for lab in ax.get_yticklabels():
            lab.set_fontstyle("italic")

        ax.set_xticks(np.arange(mat.shape[1]))
        ax.set_xticklabels(mat.columns.tolist(), rotation=45, ha="right")

        ax.set_xticks(np.arange(-0.5, mat.shape[1], 1), minor=True)  # kept style
        ax.set_yticks(np.arange(-0.5, mat.shape[0], 1), minor=True)
        ax.grid(which="minor", color=edgecolor, linewidth=linewidth)
        ax.tick_params(which="minor", bottom=False, left=False)

        ax.set_ylabel("")

    _draw(ax_acc, acc_mat, "Accuracy")
    _draw(ax_f1, f1_mat, "F1")

    ax_f1.set_xlabel("Year")
    fig.suptitle(title, y=1.06, fontsize=14, fontweight="bold", fontstyle="italic")

    return fig, (ax_acc, ax_f1)



###############################################################################
## RF YEAR AVG HEATMAP PLOTTER
###############################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from typing import Dict, Tuple, Optional, Sequence


def plot_RF_performance_acc_f1_heatmaps(
    df: pd.DataFrame,
    title: str,
    class_display_names: Dict[str, str],
    class_colors: Dict[str, str],
    year_col: str = "clas_year",
    acc_prefix: str = "acc",
    f1_prefix: str = "f1",
    n_prefix: str = "n",
    years: Optional[Sequence[int]] = None,
):
    """
    Plot Random Forest performance heatmaps (Accuracy and F1) per class over years.

    The function produces a two-panel figure:
      1) Accuracy heatmap
      2) F1 heatmap

    Input expectations
    ------------------
    The dataframe must contain:
      - a year column (default: ``clas_year``; configurable via `year_col`)
      - per-class metric columns in the form:
          * ``{acc_prefix}_<class>``  (default: ``acc_<class>``)
          * ``{f1_prefix}_<class>``   (default: ``f1_<class>``)
          * ``{n_prefix}_<class>``    (default: ``n_<class>``; number of validation samples)

    Classes are detected from ``{n_prefix}_<class>`` columns that also have matching
    accuracy and F1 columns.

    Internally, class names are converted to *stripped keys* using ``name_strip``.
    The dictionaries `class_display_names` and `class_colors` must be keyed by these
    stripped keys.

    Year handling
    -------------
    - If `years` is None (default): the x-axis includes only the years present in `df[year_col]`.
    - If `years` is provided (sequence of ints): the x-axis includes exactly those years
      (in the provided order). Years without data are still included and shown as empty
      (NaN) columns.

    Masking rule
    ------------
    Cells with ``n <= 0`` are treated as “no validation data” and are not drawn (left empty).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with year column and per-class metric columns.
    title : str
        Figure title (suptitle).
    class_display_names : Dict[str, str]
        Mapping: stripped class key -> display label used on the y-axis.
    class_colors : Dict[str, str]
        Mapping: stripped class key -> color (e.g. hex string) used for the heatmap cells.
    year_col : str, default "clas_year"
        Name of the year column in `df`.
    acc_prefix : str, default "acc"
        Prefix used for accuracy columns.
    f1_prefix : str, default "f1"
        Prefix used for F1 columns.
    n_prefix : str, default "n"
        Prefix used for sample count columns.
    years : Optional[Sequence[int]], default None
        Optional explicit year list to force on the x-axis.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Matplotlib figure.
    (ax_acc, ax_f1) : Tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]
        Axes for the Accuracy and F1 panels.
    """
    # --- plot settings ---
    alpha_min = 0.08
    facecolor_empty = "#f2f2f2"
    edgecolor = "white"
    linewidth = 1.2
    text_size = 8

    d = df.copy()
    if year_col not in d.columns:
        raise ValueError(f"Missing required column '{year_col}'.")

    d[year_col] = pd.to_numeric(d[year_col], errors="coerce")
    d = d[d[year_col].notna()].sort_values(year_col)
    d[year_col] = d[year_col].astype(int)

    # normalize years argument
    if years is not None:
        years = [int(y) for y in years]

    # detect classes from n_<cls> that also have acc_<cls> and f1_<cls>
    raw_classes = []
    for c in d.columns:
        if c.startswith(f"{n_prefix}_"):
            cls = c[len(f"{n_prefix}_"):]  # e.g. "Open_water"
            if f"{acc_prefix}_{cls}" in d.columns and f"{f1_prefix}_{cls}" in d.columns:
                raw_classes.append(cls)

    if not raw_classes:
        raise ValueError("No classes found. Need columns like n_<cls>, acc_<cls>, f1_<cls>.")

    # map raw -> stripped key
    raw_to_key = {rc: name_strip(rc) for rc in sorted(set(raw_classes))}
    keys_in_df = sorted(set(raw_to_key.values()))

    # ensure colors exist
    missing = [k for k in keys_in_df if k not in class_colors or not class_colors[k]]
    if missing:
        key_to_raw = {}
        for raw, key in raw_to_key.items():
            key_to_raw.setdefault(key, []).append(raw)

        details = "\n".join(
            f"  - '{k}' (from df classes: {sorted(key_to_raw.get(k, []))})"
            for k in missing
        )
        raise ValueError(
            "No color defined for one or more classes found in df.\n"
            "Missing stripped keys:\n"
            f"{details}\n\n"
            "Define these in class_colors (keys must be stripped with name_strip)."
        )

    def _display_for_key(k: str) -> str:
        return class_display_names.get(k, k)

    # build matrices (rows are stripped keys)
    def _pivot(metric_prefix: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        rows = []
        for raw_cls, key in raw_to_key.items():
            tmp = d[[year_col, f"{metric_prefix}_{raw_cls}", f"{n_prefix}_{raw_cls}"]].copy()
            tmp.columns = [year_col, "score", "n"]
            tmp["key"] = key
            rows.append(tmp)

        long = pd.concat(rows, ignore_index=True)
        long["n"] = pd.to_numeric(long["n"], errors="coerce").fillna(0)

        # mask where no validation data
        long.loc[long["n"] <= 0, "score"] = np.nan

        # if multiple raw classes collapse to same stripped key, average score and sum n
        score_mat = long.pivot_table(index="key", columns=year_col, values="score", aggfunc="mean")
        n_mat = long.pivot_table(index="key", columns=year_col, values="n", aggfunc="sum")

        # enforce requested year columns (including empty ones)
        if years is not None:
            score_mat = score_mat.reindex(columns=years)
            n_mat = n_mat.reindex(columns=years)

        score_mat = score_mat.reindex(keys_in_df)
        n_mat = n_mat.reindex(keys_in_df)
        return score_mat, n_mat

    acc_mat, n_mat = _pivot(acc_prefix)
    f1_mat, _ = _pivot(f1_prefix)

    years_used = acc_mat.columns.tolist()
    figsize = (1.2 + 0.9 * max(len(years_used), 1), 2.6 + 0.55 * len(keys_in_df))

    fig, (ax_acc, ax_f1) = plt.subplots(
        nrows=2, ncols=1, figsize=figsize, sharex=True, sharey=True,
        constrained_layout=True, gridspec_kw={"hspace": 0.06}
    )

    def _draw(ax, mat: pd.DataFrame, subtitle: str):
        ax.set_title(subtitle, loc="center", fontsize=12, fontweight="bold")
        ax.set_facecolor(facecolor_empty)

        for i, key in enumerate(mat.index):
            rgb = mcolors.to_rgb(class_colors[key])
            for j, _yr in enumerate(mat.columns):
                val = mat.iat[i, j]
                nval = n_mat.iat[i, j] if (i < n_mat.shape[0] and j < n_mat.shape[1]) else np.nan

                if pd.isna(val):
                    continue

                a = float(np.clip(val, 0.0, 1.0))
                a = max(alpha_min, a)

                rect = plt.Rectangle(
                    (j - 0.5, i - 0.5), 1.0, 1.0,
                    facecolor=rgb, alpha=a,
                    edgecolor=edgecolor, linewidth=linewidth
                )
                ax.add_patch(rect)

                tcol = "white" if a >= 0.55 else "#1a1a1a"
                n_txt = "" if pd.isna(nval) else f"\n(n={int(nval)})"
                ax.text(
                    j, i, f"{val:.2f}{n_txt}",
                    ha="center", va="center",
                    fontsize=text_size, color=tcol
                )

        ax.set_xlim(-0.5, mat.shape[1] - 0.5)
        ax.set_ylim(mat.shape[0] - 0.5, -0.5)

        ax.set_yticks(np.arange(mat.shape[0]))
        ax.set_yticklabels([_display_for_key(k) for k in mat.index])
        for lab in ax.get_yticklabels():
            lab.set_fontstyle("italic")

        ax.set_xticks(np.arange(mat.shape[1]))
        ax.set_xticklabels(mat.columns.tolist(), rotation=45, ha="right")

        ax.set_xticks(np.arange(-0.5, mat.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, mat.shape[0], 1), minor=True)
        ax.grid(which="minor", color=edgecolor, linewidth=linewidth)
        ax.tick_params(which="minor", bottom=False, left=False)

        ax.set_ylabel("")

    _draw(ax_acc, acc_mat, "Accuracy")
    _draw(ax_f1, f1_mat, "F1")

    ax_f1.set_xlabel("Year")
    fig.suptitle(title, y=1.06, fontsize=14, fontweight="bold", fontstyle="italic")

    return fig, (ax_acc, ax_f1)