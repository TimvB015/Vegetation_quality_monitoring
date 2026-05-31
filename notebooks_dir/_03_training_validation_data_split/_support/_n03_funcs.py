from __future__ import annotations

################################################################################
##  FULL WORKFLOW
################################################################################
import re
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from functions.gpkg_funcs import reproject_gpkg_func
from functions.training_validation_split_funcs import (
    training_validation_split_func,
    idx_to_carto_training_gdf_func,
)
from functions.idx_to_gdf_or_plot_funcs import idx_df_to_gdf_func
from functions.idx_pixel_count_funcs import pixel_count_df_func, pixel_vis_plus_differences_func


def train_val_split_automation(
    *,
    run_id: str,
    gdf,
    idx_df: pd.DataFrame,
    output_dir: Path,
    cap_pixels_by_class: Dict[Any, Any],
    train_pct: float = 0.7,
    val_pct: float = 0.3,
    test_pct: float = 0.0,
    area_col: str = "Shape_Area",
    large_polygon_threshold: int = 36,
    seed: int = 42,
    overwrite: bool = False,
    ml_epsg: str = "EPSG:32631",
    pixels_row_order: list = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Automates a train/validation split workflow with consistent, safe I/O behavior.

    Outputs managed by this function
    -------------------------------
    Pickled dataframes (the “split dfs”):
      - training_df      -> output_dir/02_training_validation_idx_dfs/idx__...training...__df.pkl
      - validation_df    -> output_dir/02_training_validation_idx_dfs/idx__...validation...__df.pkl
      - summary_df       -> output_dir/02_training_validation_split_sum_df/sum__...__df.pkl

    Associated GeoPackages (“associated gpkgs”):
      - Carto training polys (RD):
          output_dir/03_training_data_gpkgs/Carto [RD]/training_polys__{run_id}__gpkg.gpkg
      - Carto validation pixels (RD):
          output_dir/03_validation_data_gpkgs/Carto [RD]/validation_pixels__{run_id}__gpkg.gpkg
      - ML training pixels (reprojected to ml_epsg):
          output_dir/03_training_data_gpkgs/ML_approach [UTM32631]/training_pixels__...__{utm_tag}__gpkg.gpkg
      - ML validation pixels (reprojected to ml_epsg):
          output_dir/03_validation_data_gpkgs/ML_approach [UTM32631]/validation_pixels__...__{utm_tag}__gpkg.gpkg

    Core logic (overwrite + existence rules)
    ---------------------------------------
    1) Split dfs creation/loading
       - overwrite=True:
           Always recompute the split using training_validation_split_func and overwrite/save
           training_df, validation_df, and summary_df.

       - overwrite=False:
           a) If ALL THREE split dfs exist:
              Load them and continue.

           b) If ONLY SOME of the split dfs exist:
              Interrupt (raise RuntimeError). Partial presence is treated as an inconsistent state.

           c) If NONE of the split dfs exist:
              - If ANY associated gpkg exists:
                  Interrupt (raise RuntimeError) because outputs exist without the split provenance.
                  (User should set overwrite=True to recreate consistently, or delete the gpkgs.)
              - If NO associated gpkg exists:
                  Compute the split and save all three split dfs.

    2) GeoPackage creation (always driven by the split dfs)
       After split dfs are available (loaded or newly created):
         - For each gpkg:
             * If it exists and overwrite=False -> skip with a confirmation print.
             * If it does not exist OR overwrite=True -> create it using the split dfs.

       Notes:
         - Carto training polys are created with idx_to_carto_training_gdf_func (RD).
         - Validation RD pixels are created with idx_df_to_gdf_func (RD).
         - ML pixel gpkgs are created by building an RD gdf via idx_df_to_gdf_func and then
           reprojecting/saving with reproject_gpkg_func to ml_epsg.

    3) Return values
       The function computes pixel-count summaries from the (loaded or created) training_df and
       validation_df and returns:
         (summary_df, idx_train_pixel_area_df_exp, idx_validation_pixel_area_df_exp)

    Parameters
    ----------
    run_id : str
        Identifier used in output filenames.
    gdf :
        Source GeoDataFrame used to create gpkgs and compute pixel counts.
    idx_df : pd.DataFrame
        Index dataframe used as input to the splitting routine.
    output_dir : Path
        Root output folder for pickles and gpkgs.
    cap_pixels_by_class : Dict[Any, Any]
        Passed through to training_validation_split_func.
    overwrite : bool
        If True, recompute and overwrite all split dfs and recreate gpkgs.
        If False, reuse existing outputs when safe and error on inconsistent states.
    ml_epsg : str
        Target EPSG for ML pixel outputs (e.g. "EPSG:32631").
    pixels_row_order : list
        Optional list defining the row order for the pixel count summary dataframes. (e.g. ['Open water', 'Wet Nature', 'Remaining'])

    Raises
    ------
    RuntimeError
        - overwrite=False and only a subset of the split dfs exists.
        - overwrite=False, all split dfs missing, but one or more associated gpkgs exists.
    """

    # -------------------------
    # Helpers
    # -------------------------
    def _strip_trailing_rd(rid: str) -> str:
        return rid[:-4] if rid.endswith("__RD") else rid

    def _remove_approach_tokens(base: str) -> str:
        return base.replace("_cart_", "_").replace("_ML_", "_")

    def _insert_after_tmp_token(base: str, split_token: str) -> str:
        parts = base.split("_")
        tmp_idx = None
        for i, tok in enumerate(parts):
            if re.fullmatch(r"tmp\d+", tok):
                tmp_idx = i
                break
        if tmp_idx is None:
            raise ValueError(
                f"Could not find a token like 'tmp<digits>' in: {base!r}. "
                f"Cannot place '{split_token}'."
            )
        return "_".join(parts[: tmp_idx + 1] + [split_token] + parts[tmp_idx + 1 :])

    def _ensure_parent(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

    def _save_pickle(df: pd.DataFrame, path: Path) -> None:
        _ensure_parent(path)
        df.to_pickle(path)

    def _load_pickle(path: Path) -> pd.DataFrame:
        return pd.read_pickle(path)

    def _as_utm_tag(epsg: str) -> str:
        m = re.fullmatch(r"EPSG:(\d+)", epsg.strip().upper())
        return f"UTM{m.group(1)}" if m else epsg.replace(":", "")

    # -------------------------
    # Paths / folders
    # -------------------------
    output_dir = Path(output_dir)

    idx_dir_shared = output_dir / "02_training_validation_idx_dfs"
    sum_dir = output_dir / "02_training_validation_split_sum_df"

    train_gpkg_dir_carto = output_dir / "03_training_data_gpkgs/Carto [RD]"
    val_gpkg_dir_carto = output_dir / "03_validation_data_gpkgs/Carto [RD]"

    train_gpkg_dir_ml = output_dir / "03_training_data_gpkgs/ML_approach [UTM32631]"
    val_gpkg_dir_ml = output_dir / "03_validation_data_gpkgs/ML_approach [UTM32631]"

    for d in [
        idx_dir_shared,
        sum_dir,
        train_gpkg_dir_carto,
        val_gpkg_dir_carto,
        train_gpkg_dir_ml,
        val_gpkg_dir_ml,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    base_for_idx = _remove_approach_tokens(_strip_trailing_rd(run_id))
    training_idx_stem = _insert_after_tmp_token(base_for_idx, "training")
    validation_idx_stem = _insert_after_tmp_token(base_for_idx, "validation")

    training_idx_path = idx_dir_shared / f"idx__{training_idx_stem}__df.pkl"
    validation_idx_path = idx_dir_shared / f"idx__{validation_idx_stem}__df.pkl"
    sum_path = sum_dir / f"sum__{base_for_idx}__df.pkl"

    # --- GPKG paths we consider "associated" with the split ---
    base_for_ml = _strip_trailing_rd(run_id).replace("_cart_", "_ML_")
    utm_tag = _as_utm_tag(ml_epsg)

    training_polys_gpkg_path = train_gpkg_dir_carto / f"training_polys__{run_id}__gpkg.gpkg"
    validation_pixels_rd_gpkg_path = val_gpkg_dir_carto / f"validation_pixels__{run_id}__gpkg.gpkg"

    training_pixels_utm_gpkg_path = (
        train_gpkg_dir_ml / f"training_pixels__{base_for_ml}__{utm_tag}__gpkg.gpkg"
    )
    validation_pixels_utm_gpkg_path = (
        val_gpkg_dir_ml / f"validation_pixels__{base_for_ml}__{utm_tag}__gpkg.gpkg"
    )

    associated_gpkg_paths = [
        training_polys_gpkg_path,
        validation_pixels_rd_gpkg_path,
        training_pixels_utm_gpkg_path,
        validation_pixels_utm_gpkg_path,
    ]

    # -------------------------
    # Step 0: Examine existence
    # -------------------------
    split_paths = {
        "training_df": training_idx_path,
        "validation_df": validation_idx_path,
        "summary_df": sum_path,
    }
    split_exists = {k: p.exists() for k, p in split_paths.items()}
    all_three_exist = all(split_exists.values())
    all_three_missing = not any(split_exists.values())
    partial_missing = (not all_three_exist) and (not all_three_missing)

    any_associated_gpkg_exists = any(p.exists() for p in associated_gpkg_paths)

    # -------------------------
    # Step 1: Load or create split dfs
    # -------------------------
    if overwrite:
        split_list = training_validation_split_func(
            gdf=gdf,
            idx_df=idx_df,
            year_cols=idx_df.columns,
            cap_pixels_by_class=cap_pixels_by_class,
            train_pct=train_pct,
            val_pct=val_pct,
            test_pct=test_pct,
            seed=seed,
            area_col=area_col,
            pixel_area_m2=100.0,
            large_polygon_threshold=large_polygon_threshold,
        )
        training_df, validation_df, summary_df = split_list[0], split_list[1], split_list[4]
        _save_pickle(training_df, training_idx_path)
        _save_pickle(validation_df, validation_idx_path)
        _save_pickle(summary_df, sum_path)
        print(
            "[train_val_split_automation] overwrite=True -> computed split and saved:\n"
            f"  - {training_idx_path}\n"
            f"  - {validation_idx_path}\n"
            f"  - {sum_path}"
        )

    else:
        # overwrite=False
        if all_three_exist:
            training_df = _load_pickle(training_idx_path)
            validation_df = _load_pickle(validation_idx_path)
            summary_df = _load_pickle(sum_path)
            print(
                "[train_val_split_automation] overwrite=False and all split files exist -> loaded:\n"
                f"  - {training_idx_path}\n"
                f"  - {validation_idx_path}\n"
                f"  - {sum_path}"
            )

        elif partial_missing:
            missing = [name for name, ex in split_exists.items() if not ex]
            raise RuntimeError(
                "[train_val_split_automation] Cannot continue: overwrite=False, but only part of the split files exist.\n"
                f"Missing: {missing}\n"
                "Either restore the missing pickle(s) or set overwrite=True to recreate."
            )

        else:
            # all_three_missing is True
            if any_associated_gpkg_exists:
                existing = [str(p) for p in associated_gpkg_paths if p.exists()]
                raise RuntimeError(
                    "[train_val_split_automation] Cannot continue: split dfs are missing, but one or more associated GPKGs exist.\n"
                    "This is likely an inconsistent output state.\n"
                    f"Existing GPKGs:\n  - " + "\n  - ".join(existing) + "\n"
                    "Either:\n"
                    "  * set overwrite=True to recreate everything consistently, or\n"
                    "  * delete the existing GPKGs and rerun with overwrite=False."
                )

            # No gpkg exist -> safe to create everything
            split_list = training_validation_split_func(
                gdf=gdf,
                idx_df=idx_df,
                year_cols=idx_df.columns,
                cap_pixels_by_class=cap_pixels_by_class,
                train_pct=train_pct,
                val_pct=val_pct,
                test_pct=test_pct,
                seed=seed,
                area_col=area_col,
                pixel_area_m2=100.0,
                large_polygon_threshold=large_polygon_threshold,
            )
            training_df, validation_df, summary_df = split_list[0], split_list[1], split_list[4]
            _save_pickle(training_df, training_idx_path)
            _save_pickle(validation_df, validation_idx_path)
            _save_pickle(summary_df, sum_path)
            print(
                "[train_val_split_automation] overwrite=False and no split files + no associated GPKGs -> computed split and saved:\n"
                f"  - {training_idx_path}\n"
                f"  - {validation_idx_path}\n"
                f"  - {sum_path}"
            )

    # -------------------------
    # Step 2: Carto training polys (RD)
    # -------------------------
    if training_polys_gpkg_path.exists() and not overwrite:
        print(f"[train_val_split_automation] overwrite=False and exists -> kept: {training_polys_gpkg_path}")
    else:
        _ensure_parent(training_polys_gpkg_path)
        idx_to_carto_training_gdf_func(
            idx_df=training_df,
            source_gdf=gdf,
            out_gpkg=training_polys_gpkg_path,
        )
        print(f"[train_val_split_automation] Created/overwritten: {training_polys_gpkg_path}")

    # -------------------------
    # Step 3: ML training pixels (UTM ml_epsg)
    # -------------------------
    if training_pixels_utm_gpkg_path.exists() and not overwrite:
        print(f"[train_val_split_automation] overwrite=False and exists -> kept: {training_pixels_utm_gpkg_path}")
    else:
        training_pixels_rd_gdf = idx_df_to_gdf_func(idx_df=training_df, source_gdf=gdf)
        _ensure_parent(training_pixels_utm_gpkg_path)
        reproject_gpkg_func(
            gdf=training_pixels_rd_gdf,
            output_epsg=ml_epsg,
            output_gpkg_path=training_pixels_utm_gpkg_path,
            return_gdf=False,
        )
        print(f"[train_val_split_automation] Created/overwritten: {training_pixels_utm_gpkg_path}")

    # -------------------------
    # Step 4: Validation gpkg (RD + UTM)
    # -------------------------
    if validation_pixels_utm_gpkg_path.exists() and not overwrite:
        print(f"[train_val_split_automation] overwrite=False and exists -> kept: {validation_pixels_utm_gpkg_path}")
    else:
        # Ensure RD exists (create if missing OR overwrite=True)
        if (not validation_pixels_rd_gpkg_path.exists()) or overwrite:
            _ensure_parent(validation_pixels_rd_gpkg_path)
            idx_df_to_gdf_func(
                idx_df=validation_df,
                source_gdf=gdf,
                out_gpkg=validation_pixels_rd_gpkg_path,
            )
            print(f"[train_val_split_automation] Created/overwritten (RD): {validation_pixels_rd_gpkg_path}")
        else:
            print(f"[train_val_split_automation] overwrite=False and exists -> kept (RD): {validation_pixels_rd_gpkg_path}")

        # Create/recreate UTM
        validation_pixels_rd_gdf = idx_df_to_gdf_func(idx_df=validation_df, source_gdf=gdf)
        _ensure_parent(validation_pixels_utm_gpkg_path)
        reproject_gpkg_func(
            gdf=validation_pixels_rd_gdf,
            output_epsg=ml_epsg,
            output_gpkg_path=validation_pixels_utm_gpkg_path,
            return_gdf=False,
        )
        print(f"[train_val_split_automation] Created/overwritten (UTM): {validation_pixels_utm_gpkg_path}")

    # -------------------------
    # Step 5: Pixel count dfs + return
    # -------------------------
    def _extract_wd_at_tokens(rid: str) -> Tuple[str, str]:
        """Extract WD and AT tokens from run_id."""
        stripped = _strip_trailing_rd(rid)
        base = stripped.rsplit("__", 1)[0] if "__" in stripped else stripped
        cleaned = _remove_approach_tokens(base)
        parts = cleaned.split("_")
    
        wd_token = parts[0] if parts else "WD?"
        at_token = next((p for p in parts if p.startswith("at")), "at?")
    
        return wd_token, at_token

    wd_token, at_token = _extract_wd_at_tokens(run_id)

    idx_train_pixel_area_df, _ = pixel_count_df_func(gdf=gdf, idx_df=training_df, area_col=area_col)
    idx_train_pixel_area_df_exp = pixel_vis_plus_differences_func(
        df=idx_train_pixel_area_df,
        years=idx_train_pixel_area_df.columns,
        title=f"{wd_token} {at_token} training pixel count",
        row_order=pixels_row_order,
        idx_df=training_df,
        original_idx_df=idx_df,
    )

    idx_validation_pixel_area_df, _ = pixel_count_df_func(gdf=gdf, idx_df=validation_df, area_col=area_col)
    idx_validation_pixel_area_df_exp = pixel_vis_plus_differences_func(
        df=idx_validation_pixel_area_df,
        years=idx_validation_pixel_area_df.columns,
        title=f"{wd_token} {at_token} validation pixel count",
        row_order=pixels_row_order,
        idx_df=validation_df,
        original_idx_df=idx_df,
    )

    return summary_df, idx_train_pixel_area_df_exp, idx_validation_pixel_area_df_exp