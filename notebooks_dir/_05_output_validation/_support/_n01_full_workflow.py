import shutil
import rasterio
from pathlib import Path

from functions.raster_ops_funcs import (
    stack_rasters_func,
)

from functions.gpkg_funcs import (
    import_gpkg_func,
    reproject_gpkg_func,
)

from notebooks_dir._05_output_validation._support._n01_funcs import (
    run_id_read,
    find_gelderland_zip_folder,
    read_carto_README_file,
    extract_carto_tifs_from_zip,
    classification_accuracy_per_year,
)

###############################################################################
## FULL WORKFLOW
###############################################################################
def building_carto_clas_acc_dfs(
    *,
    carto_output_processing__dir,
    carto_validation_data__dir,
    carto_validation_label_col: str = "type",
    carto_validation_year_col: str = "years",
    run_id: str = "WDX_plusOW_p80_tmp2_cart_atX__gelderland__RD"
):
    """
    End-to-end pipeline:
      1) Parse run_id -> hab_selection, train_split_attempt
      2) Find zip folder with Carto outputs
      3) Read class map from README
      4) Extract yearly GeoTIFFs from zip
      5) Stack rasters into a multi-band GeoTIFF (one band per year)
      6) Load validation gpkg and reproject to EPSG:32631
      7) Read nodata from stacked raster
      8) Compute classification accuracy metrics per year
      9) Save df_metrics as pickle to 03_classification_acc_dfs
     10) Return df_metrics
    """
    # --- setting dir / parsing ---
    carto_output_processing__dir = Path(carto_output_processing__dir)
    carto_validation_data__dir = Path(carto_validation_data__dir)

    parsed = run_id_read(run_id)
    hab_selection = parsed["hab_selection"]
    train_split_attempt = parsed["train_split_attempt"]

    # --- input zip folder ---
    input_zip_folder = find_gelderland_zip_folder(
        input_dir=carto_output_processing__dir,
        hab_selection=hab_selection,
        train_split_attempt=train_split_attempt,
    )

    # --- stacked raster output ---
    stacked_rstrs_dir = carto_output_processing__dir / "02_stacked_rstrs"
    stacked_rstrs_dir.mkdir(exist_ok=True, parents=True)

    stacked_rstr_name = f"{hab_selection}_{train_split_attempt}_gelderland_stacked_rstrs.tif"
    stacked_rstr_path = stacked_rstrs_dir / stacked_rstr_name

    # --- validation data ---
    validation_data__gpkg_path = (
        carto_validation_data__dir / f"validation_pixels__{run_id}__gpkg.gpkg"
    )

    # --- classification acc df output dir/path ---
    classification_acc__df_dir = carto_output_processing__dir / "03_classification_result_stats_dfs"
    classification_acc__df_dir.mkdir(exist_ok=True, parents=True)
    classification_acc__df_path = classification_acc__df_dir / f"classification_results_stats__{hab_selection}_{train_split_attempt}.pkl"
    classification_acc__csv_path = classification_acc__df_dir / f"classification_results_stats__{hab_selection}_{train_split_attempt}.csv"

    # --- class map + extract tifs ---
    rstr_classes = read_carto_README_file(input_zip_folder)

    classification_tif_paths, tmp_extract_dir, carto_out_years = extract_carto_tifs_from_zip(
        zipped_folder_path=input_zip_folder,
    )

    # --- stack rasters (and cleanup tmp dir) ---
    try:
        _ = stack_rasters_func(
            raster_paths=classification_tif_paths,
            out_dir=stacked_rstrs_dir,
            out_name=stacked_rstr_name,
            band_names=[p.stem.split("_")[-1] for p in classification_tif_paths],
            tolerance=0.0,
            rstr_classes=rstr_classes,
            overwrite=False,
        )
    finally:
        shutil.rmtree(tmp_extract_dir, ignore_errors=True)

    # --- load & reproject validation gdf ---
    validation_gdf = import_gpkg_func(validation_data__gpkg_path)
    validation_gdf_UTM32631 = reproject_gpkg_func(gdf=validation_gdf, output_epsg="EPSG:32631")

    # --- nodata from stacked raster ---
    with rasterio.open(stacked_rstr_path) as src:
        nodata_dataset = src.nodata

    # --- metrics ---
    df_metrics = classification_accuracy_per_year(
        validation_gdf=validation_gdf_UTM32631,
        stacked_rstr_path=stacked_rstr_path,
        label_col=carto_validation_label_col,
        year_col=carto_validation_year_col,
        raster_band_years=carto_out_years,
        nodata_value=nodata_dataset,
        class_map=rstr_classes,
        assume_points=False,
        dropna_year=False,
    )

    df_metrics.to_pickle(classification_acc__df_path)
    df_metrics.to_csv(classification_acc__csv_path, index=False)
    return df_metrics