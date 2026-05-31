"""
Full workflow plotting function for habitat classification validation
"""

import pandas as pd
from functions.gpkg_funcs import import_gpkg_func
from notebooks_dir._05_output_validation._support._n04_funcs import (
    hex_to_rgb,
    filter_by_timeframe,
    remove_typologies_from_color_df,
    build_windows_gdf,
    aoi_overlays_to_gdf,
    all_pixels_gpkg_processing,
    training_validation_pixels_gdf,
    clip_gdf_to_window,
    clip_raster_to_window,
    add_svg_northarrow,
    get_default_north_arrow_kwargs,
    build_legend_df_from_raster,
    add_bw_scalebar,
    get_default_scalebar_kwargs,
    add_color_legend,
    get_default_legend_kwargs,
    add_consistency_legend,
    get_default_consistency_legend_kwargs,
    add_class_performance_badges,
    get_default_class_performance_kwargs,
    add_class_pixel_counts,
    get_default_class_pixel_counts_kwargs,
    resolve_background_path,
    extract_class_map,
    process_rgb_array,
    process_CLASS_MAP_background,
    process_background,
    get_band_mapping_for_raster,
    merge_illustration_settings,
    prepare_location_data,
    auto_detect_year_range,
    add_consistency_legend,
    plot_prepared_cell,
    plot_all_locations,
)
from functions.custom_common_nb_funcs import (
    extract_selection_colors,
)
from paths.OG_paths import(
    #######################################
    #          GENERAL POLYGONS           #
    #######################################
    # |        Veluwe polygons            |
    # +===================================+
    veluwe_polygon__aoi__UTM32631__gpkg_path,
    veluwe_polygon__subaoi__UTM32631__gpkg_path,
    veluwe_polygon__n2000__UTM32631__gpkg_path,
    veluwe_polygon__ruim__UTM32631__gpkg_path,
    veluwe_polygon__full_plotting_window__UTM32631__gpkg_path,

    #######################################
    #            00_common_dfs            #
    #######################################
    # |    n01_habitat_reference_dfs      |
    # +===================================+
    habitat_reference__WD__df_path,
    habitat_reference__s__df_path,

    #######################################
    #            MAP ADDITIONS            #
    #######################################
    # Dirs
    pdok_imgs_stacked__rstr_dir,
    north_arrow__path,
    # Plot windows
    plot_extent__deelenschev__UTM32631__gpkg_path,
    plot_extent__gerritsfles__UTM32631__gpkg_path,
    plot_extent__kootwijkerveen__UTM32631__gpkg_path,
    plot_extent__mosterdveen__UTM32631__gpkg_path,
    plot_extent__speulderveld__UTM32631__gpkg_path,

    #######################################
    #  03_training_validation_data_split  #
    #######################################
    # | n01_build_selected_pixels_idx_dfs |
    # +===================================+
    habitat_kart_plusOW__FC_p80_tmp2_selected__gelderland__RD__gpkg_path,

    # +===================================+
    # | n03_training_validation_split_ML  |
    # +===================================+    
    # --- training validation GPKGs ML-approach [UTM32631] --- #
    validation_ML__UTM32631__gpkgs_dir,
    training_ML__UTM32631__gpkgs_dir,

    #######################################
    #        04_ML_classification         #
    #######################################
    # |  n01_random_forest_s2_quarterly   |
    # +===================================+
    RF_out__UTM32631__rstrs_dir,

    #######################################
    #        05_output_validation         #
    #######################################
    # |    n01_checking_carto_results     |
    # +===================================+
    carto_output_stack__dir,

    # +===================================+
    # |   n02_carto_RF_results_comp_df    |
    # +===================================+
    carto_output_result_dfs__dir,

    # +===================================+
    # |     n03_building_entropy_rstrs    |
    # +===================================+
    pixel_stability_rstrs__dir,

    # +===================================+
    # |       n04_result_plotting         |
    # +===================================+
    result_plotting__dir,
)


def full_workflow_plotting_func(
    habitat_reference_df,
    train_years,
    raster_stack,
    hab_selection,
    train_split_attempt,
    vis_years_description,
    vis_years,
    project_epsg,
    project_UTM,
    locations,
    columns_to_vis,
    figsize=(4, 4),
    dpi=150,
    overwrite=False,
):
    """
    Full workflow for generating habitat classification validation plots.
    
    Parameters
    ----------
    habitat_reference_df : pd.DataFrame
        Reference dataframe containing habitat type information and colors
    train_years : str
        Training years range (e.g., "2017-2024")
    raster_stack : str
        Band selection identifier (e.g., "b2348")
    hab_selection : str
        Habitat selection code (e.g., "WD2")
    train_split_attempt : str
        Training split attempt identifier (e.g., "at1")
    vis_years_description : str
        Visualization years description (e.g., "2017-2024")
    vis_years : list
        List of years to visualize (e.g., [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024])
    project_epsg : str
        Project EPSG code (e.g., "EPSG:32631")
    project_UTM : str
        Project UTM zone (e.g., "UTM32631")
    background_config : dict
        Configuration for background layer
    gdf_overlay_config : dict
        Configuration for GDF overlays
    illustrations_config : dict
        Configuration for illustrations
    RF_row_stability_config : dict
        Configuration for RF row stability
    RF_row_stable_pixels_config : dict
        Configuration for RF stable pixels
    RF_row_unstable_pixels_config : dict
        Configuration for RF unstable pixels
    header_row_config : dict
        Configuration for header row
    meta_rows_config : dict
        Configuration for metadata rows
    locations : list
        List of location names to plot
    columns_to_vis : list
        List of column names to visualize
    figsize : tuple, optional
        Figure size, by default (4, 4)
    dpi : int, optional
        DPI for output figures, by default 150
    overwrite : bool, optional
        Whether to overwrite existing files, by default False
    
    Returns
    -------
    figures : dict
        Dictionary containing generated figures
    """
    
    # Convert vis_years to string tuple for overlay configs
    years_string_list = tuple(str(year) for year in vis_years)
    
    # ==================== PATH CONSTRUCTION ====================
    
    # --- Legend builder ---
    unstable_pixels_legend_path = str(
        pixel_stability_rstrs__dir / f"RF_row_stability__{hab_selection}__{train_split_attempt}__2017-2024__Q1234__2019__{raster_stack}/unstable_pixels.tif"
    )
    
    # --- Background rasters ---
    pdok_imgs_path = str(
        pdok_imgs_stacked__rstr_dir / "{location}_stack.tif"
    )
    
    pdok_full_veluwe_imgs = str(
        pdok_imgs_stacked__rstr_dir / "full_veluwe_stack.tif"
    )
    
    # --- Carto results ---
    carto_result_stats__path = str(
        carto_output_result_dfs__dir / f"classification_results_stats__{hab_selection}_{train_split_attempt}.csv"
    )
    
    carto_output_path = str(
        carto_output_stack__dir / f"{hab_selection}_{train_split_attempt}_gelderland_stacked_rstrs.tif"
    )
    
    carto_col_stability_path = str(
        pixel_stability_rstrs__dir / f"carto_col_stability__{hab_selection}__{train_split_attempt}__{vis_years_description}/decision_category.tif"
    )
    
    carto_modal_class_path = str(
        pixel_stability_rstrs__dir / f"carto_col_stability__{hab_selection}__{train_split_attempt}__{vis_years_description}/modal_class.tif"
    )
    
    carto_unstable_pixels_path = str(
        pixel_stability_rstrs__dir / f"carto_col_stability__{hab_selection}__{train_split_attempt}__{vis_years_description}/unstable_pixels.tif"
    )
    
    # --- RF results ---
    RF_results_stats__path = str(
        RF_out__UTM32631__rstrs_dir / f"{raster_stack}/{hab_selection}/performance_dfs/"
        f"RF_validation__{raster_stack}__{hab_selection}_{train_split_attempt}__train{train_years}__mdl1__UTM32631.csv"
    )
    
    RF_Q1_path = str(
        RF_out__UTM32631__rstrs_dir / f"{raster_stack}_stacked" / f"{hab_selection}" /
        f"stack_{raster_stack}__{hab_selection}_{train_split_attempt}_Q1__rstr.tif"
    )
    
    RF_Q2_path = str(
        RF_out__UTM32631__rstrs_dir / f"{raster_stack}_stacked" / f"{hab_selection}" /
        f"stack_{raster_stack}__{hab_selection}_{train_split_attempt}_Q2__rstr.tif"
    )
    
    RF_Q3_path = str(
        RF_out__UTM32631__rstrs_dir / f"{raster_stack}_stacked" / f"{hab_selection}" /
        f"stack_{raster_stack}__{hab_selection}_{train_split_attempt}_Q3__rstr.tif"
    )
    
    RF_Q4_path = str(
        RF_out__UTM32631__rstrs_dir / f"{raster_stack}_stacked" / f"{hab_selection}" /
        f"stack_{raster_stack}__{hab_selection}_{train_split_attempt}_Q4__rstr.tif"
    )
    
    RF_col_stability_Q1_path = str(
        pixel_stability_rstrs__dir / f"RF_col_stability__{hab_selection}__{train_split_attempt}__{vis_years_description}__Q1__{raster_stack}/decision_category.tif"
    )
    
    RF_col_stability_Q2_path = str(
        pixel_stability_rstrs__dir / f"RF_col_stability__{hab_selection}__{train_split_attempt}__{vis_years_description}__Q2__{raster_stack}/decision_category.tif"
    )
    
    RF_col_stability_Q3_path = str(
        pixel_stability_rstrs__dir / f"RF_col_stability__{hab_selection}__{train_split_attempt}__{vis_years_description}__Q3__{raster_stack}/decision_category.tif"
    )
    
    RF_col_stability_Q4_path = str(
        pixel_stability_rstrs__dir / f"RF_col_stability__{hab_selection}__{train_split_attempt}__{vis_years_description}__Q4__{raster_stack}/decision_category.tif"
    )
    
    RF_col_modal_class_Q1_path = str(
        pixel_stability_rstrs__dir / f"RF_col_stability__{hab_selection}__{train_split_attempt}__{vis_years_description}__Q1__{raster_stack}/modal_class.tif"
    )
    
    RF_col_modal_class_Q2_path = str(
        pixel_stability_rstrs__dir / f"RF_col_stability__{hab_selection}__{train_split_attempt}__{vis_years_description}__Q2__{raster_stack}/modal_class.tif"
    )
    
    RF_col_modal_class_Q3_path = str(
        pixel_stability_rstrs__dir / f"RF_col_stability__{hab_selection}__{train_split_attempt}__{vis_years_description}__Q3__{raster_stack}/modal_class.tif"
    )
    
    RF_col_modal_class_Q4_path = str(
        pixel_stability_rstrs__dir / f"RF_col_stability__{hab_selection}__{train_split_attempt}__{vis_years_description}__Q4__{raster_stack}/modal_class.tif"
    )
    
    RF_col_unstable_pixels_Q1_path = str(
        pixel_stability_rstrs__dir / f"RF_col_stability__{hab_selection}__{train_split_attempt}__{vis_years_description}__Q1__{raster_stack}/unstable_pixels.tif"
    )
    
    RF_col_unstable_pixels_Q2_path = str(
        pixel_stability_rstrs__dir / f"RF_col_stability__{hab_selection}__{train_split_attempt}__{vis_years_description}__Q2__{raster_stack}/unstable_pixels.tif"
    )
    
    RF_col_unstable_pixels_Q3_path = str(
        pixel_stability_rstrs__dir / f"RF_col_stability__{hab_selection}__{train_split_attempt}__{vis_years_description}__Q3__{raster_stack}/unstable_pixels.tif"
    )
    
    RF_col_unstable_pixels_Q4_path = str(
        pixel_stability_rstrs__dir / f"RF_col_stability__{hab_selection}__{train_split_attempt}__{vis_years_description}__Q4__{raster_stack}/unstable_pixels.tif"
    )
    
    RF_row_stability_Q1234_path = str(
        pixel_stability_rstrs__dir / f"RF_row_stability__{hab_selection}__{train_split_attempt}__2017-2024__Q1234__{{year}}__{raster_stack}/decision_category.tif"
    )
    
    RF_row_modal_class_Q1234_path = str(
        pixel_stability_rstrs__dir / f"RF_row_stability__{hab_selection}__{train_split_attempt}__2017-2024__Q1234__{{year}}__{raster_stack}/modal_class.tif"
    )
    
    RF_row_unstable_pixels_Q1234_path = str(
        pixel_stability_rstrs__dir / f"RF_row_stability__{hab_selection}__{train_split_attempt}__2017-2024__Q1234__{{year}}__{raster_stack}/unstable_pixels.tif"
    )
    
    # ==================== IMPORTING FILES ====================

    all_pixels_gdf = import_gpkg_func(
        habitat_kart_plusOW__FC_p80_tmp2_selected__gelderland__RD__gpkg_path,
        cols_to_keep=["index", "years", "habitatType1", "habitatnaam_1_disp", "bedekkingsPercentage1", "habitat_color", "type", "geometry"]
    )

    # ==================== BUILD WINDOWS GDF ====================
    
    window_info_list = [
        {"location": "Gerritsfles",
        "epsg": project_UTM,
        "plot_window": plot_extent__gerritsfles__UTM32631__gpkg_path},
        
        {"location": "Deelensche veld",
        "epsg": project_UTM,
        "plot_window": plot_extent__deelenschev__UTM32631__gpkg_path},
        
        {"location": "Kootwijkerveen",
        "epsg": project_UTM,
        "plot_window": plot_extent__kootwijkerveen__UTM32631__gpkg_path},
        
        {"location": "Mosterdveen",
        "epsg": project_UTM,
        "plot_window": plot_extent__mosterdveen__UTM32631__gpkg_path},
        
        {"location": "Speulderveld",
        "epsg": project_UTM,
        "plot_window": plot_extent__speulderveld__UTM32631__gpkg_path},

        {"location": "Veluwe overview",
        "epsg": project_UTM,
        "plot_window": veluwe_polygon__full_plotting_window__UTM32631__gpkg_path},
    ]
    
    windows_gdf = build_windows_gdf(window_info_list)
    
    # ==================== BUILD AOI OVERLAYS ====================
    
    sub_aoi_overlays = [
        {"description": "sub-aoi",
        "gpkg_path": veluwe_polygon__subaoi__UTM32631__gpkg_path,
        "epsg": project_UTM,
        "type": "Sub-region of interest",
        "facecolor": "none",
        "edgecolor": "red",
        "linewidth": 1.5,
        "linestyle": "solid",
        "years": years_string_list},
    ]
    sub_aois_overlay_gdf = aoi_overlays_to_gdf(sub_aoi_overlays, crs=project_epsg)

    aoi_overlays = [
        {"description": "aoi",
        "gpkg_path": veluwe_polygon__aoi__UTM32631__gpkg_path,
        "epsg": project_UTM,
        "type": "Full region of interest",
        "facecolor": "none",
        "edgecolor": "black",
        "linewidth": 1.5,
        "linestyle": ":",
        "years": years_string_list},
    ]

    aois_overlay_gdf = aoi_overlays_to_gdf(aoi_overlays, crs=project_epsg)

    veluwe_polygon__n2000 = [
        {"description": "n2000",
        "gpkg_path": veluwe_polygon__n2000__UTM32631__gpkg_path,
        "epsg": project_UTM,
        "type": "Full region of interest",
        "facecolor": "none",
        "edgecolor": "black",
        "linewidth": 0.5,
        "linestyle": "solid",
        "years": years_string_list},
    ]

    veluwe_polygon__n2000__UTM32631__gdf = aoi_overlays_to_gdf(veluwe_polygon__n2000)

    header_vis_plot = [
        {"description": "AOI",
        "gpkg_path": r"D:\Thesis\10.Thesis_Data\00_basismaps_raw\veluwe_polygons\Speulderveld_UTM32631.gpkg",
        "epsg": project_UTM,
        "type": "Full region of interest",
        "facecolor": "none",
        "edgecolor": "red",
        "linewidth": 0.5,
        "linestyle": "solid",
        "years": years_string_list},
    ]

    header_vis_plot_gdf = aoi_overlays_to_gdf(header_vis_plot, crs=project_epsg)
    
    # ==================== PROCESS PIXEL DATA ====================
    
    all_pixels_plotting_gdf = all_pixels_gpkg_processing(
        all_pixels_gdf=all_pixels_gdf,
        habitat_reference_df=habitat_reference_df,
        hab_selection=hab_selection,
        epsg=project_UTM,
    )
    
    colors_df = extract_selection_colors(habitat_reference_df, hab_selection=hab_selection)
    
    colors_df_no_remaining = remove_typologies_from_color_df(colors_df, typologies_to_remove=["Remaining"])
    
    # For validation pixels (default pattern)
    val_gdf = training_validation_pixels_gdf(
        pixels_dir=validation_ML__UTM32631__gpkgs_dir,
        hab_selection=hab_selection,
        train_split_attempt=train_split_attempt,
        epsg=project_UTM,
        type_color_df=colors_df,
        filename_pattern="validation_pixels__{hab_selection}_plusOW_p80_tmp2_ML_{train_split_attempt}__gelderland__{epsg}__gpkg.gpkg",
    )
    
    # For training pixels (custom pattern)
    train_gdf = training_validation_pixels_gdf(
        pixels_dir=training_ML__UTM32631__gpkgs_dir,
        hab_selection=hab_selection,
        train_split_attempt=train_split_attempt,
        epsg=project_UTM,
        type_color_df=colors_df,
        filename_pattern="training_pixels__{hab_selection}_plusOW_p80_tmp2_ML_{train_split_attempt}__gelderland__{epsg}__gpkg.gpkg",
    )
    
    # ==================== BUILD LEGEND DATA ====================

    RF_row_stability_color_df = pd.DataFrame({
        'value': [1, 2, 4, 5],
        'description': [
            '≥80%',
            '60-80%', 
            'Tie ≥40%',
            'Unstable'
        ],
        'color': [
            '#004D00',
            "#75BB2E",
            '#FFA500',
            '#FF0000',
        ],
    })

    # Building unstable pixels df
    unstable_pixels_df = build_legend_df_from_raster(
        raster_path=unstable_pixels_legend_path,
        label_col='description'
    )

    highly_unstable_color_df = unstable_pixels_df[unstable_pixels_df['pixel_value'] == 'highly_unstable']

    unstable_pixels_df = unstable_pixels_df[unstable_pixels_df['pixel_value'] != 'highly_unstable'].drop('pixel_value', axis=1)
    
    # ==================== LOAD STATS DATAFRAMES ====================
    
    # Loading stats dfs
    carto_stats_df = pd.read_csv(carto_result_stats__path)
    RF_stats_df = pd.read_csv(RF_results_stats__path)
    RF_Q1_stats_df = filter_by_timeframe(RF_stats_df, 'Q1')
    RF_Q2_stats_df = filter_by_timeframe(RF_stats_df, 'Q2')
    RF_Q3_stats_df = filter_by_timeframe(RF_stats_df, 'Q3')
    RF_Q4_stats_df = filter_by_timeframe(RF_stats_df, 'Q4')

    ################################################################################
    ## WD2 / WD3 / WD4 [STABILITY]
    ################################################################################

    # ==================== CLASSES OVERVIEW ====================
    # CLASSES OVERVIEW
    classes_overview = {
        "WD1": {
            0: "Open water",
            1: "Remaining",
            2: "Wet Nature",
        },
        "WD2": {
            0: "Open water",
            1: "Remaining",
            2: "Semi-Wet Nature",
            3: "Wet Nature",
        },
        "WD3": {
            0: "Open water",
            1: "Remaining",
            2: "Wet Nature",
        },
        "WD4": {
            0: "Open water",
            1: "Remaining",
            2: "Semi-Wet Nature",
            3: "Wet Nature",
        },
        "WD5": {
            0: "Open water",
            1: "Remaining",
            2: "Semi-Wet Nature",
        },
    }


    # ==================== BACKGROUNDS ====================
    # BACKGROUNDS 
    background_config = {
        "All Data": {
            "raster_path": pdok_imgs_path,
            "colorscheme": "rgb",
            "band_mapping": "auto",
            "background_alpha_override": 0.8,
        },
        "Training Data": {
            "raster_path": pdok_imgs_path,
            "colorscheme": "rgb",
            "band_mapping": "auto",
            "background_alpha_override": 0.7,
        },
        "Validation Data": {
            "raster_path": pdok_imgs_path,
            "colorscheme": "rgb",
            "band_mapping": "auto",
            "background_alpha_override": 0.6,
        },
        "Carto": {
            "raster_path": carto_output_path,
            "colorscheme": "CLASS-MAP",
            "band_mapping": "auto",
            "background_alpha_override": None,
            "color_df": colors_df,
        },
        "RF Q1": {
            "raster_path": RF_Q1_path,
            "colorscheme": "CLASS-MAP",
            "band_mapping": "auto",
            "background_alpha_override": None,
            "color_df": colors_df,
        },
        "RF Q2": {
            "raster_path": RF_Q2_path,
            "colorscheme": "CLASS-MAP",
            "band_mapping": "auto",
            "background_alpha_override": None,
            "color_df": colors_df,
        },
        "RF Q3": {
            "raster_path": RF_Q3_path,
            "colorscheme": "CLASS-MAP",
            "band_mapping": "auto",
            "background_alpha_override": None,
            "color_df": colors_df,
        },
        "RF Q4": {
            "raster_path": RF_Q4_path,
            "colorscheme": "CLASS-MAP",
            "band_mapping": "auto",
            "background_alpha_override": None,
            "color_df": colors_df,
        },
        "RF row stability": {
            "raster_path": pdok_imgs_path,
            "colorscheme": "rgb",
            "band_mapping": "auto",
            "background_alpha_override": 0.5,
        },
        "RF stable pixels": {
            "raster_path": pdok_imgs_path,
            "colorscheme": "rgb",
            "band_mapping": "auto",
            "background_alpha_override": 0.5,
        },
        "RF unstable pixels": {
            "raster_path": pdok_imgs_path,
            "colorscheme": "rgb",
            "band_mapping": "auto",
            "background_alpha_override": 0.5,
        },
    }

    # ==================== GDF OVERLAYS ====================
    # GDF OVERLAYS
    gdf_overlay_config = {
        "All Data": [
            {
                "gdf": sub_aois_overlay_gdf,
                "edgecolor": "red",
                "facecolor": "none",
                "linewidth": 1.0,
                "linestyle": "solid",
                "zorder": 4,
            },
            {
                "gdf": aois_overlay_gdf,
                "edgecolor": "black",
                "facecolor": "none",
                "linewidth": 1.0,
                "linestyle": "--",
                "zorder": 5,
            },
            {
                "gdf": all_pixels_plotting_gdf,
                "zorder": 4,
                "linewidth": 0.1,
                "alpha": 1,
            },
        ],
        "Training Data": [
            {
                "gdf": sub_aois_overlay_gdf,
                "edgecolor": "red",
                "facecolor": "none",
                "linewidth": 1.0,
                "linestyle": "solid",
                "zorder": 4,
            },
            {
                "gdf": aois_overlay_gdf,
                "edgecolor": "black",
                "facecolor": "none",
                "linewidth": 1.0,
                "linestyle": "--",
                "zorder": 5,
            },
            {
                "gdf": train_gdf,
                "zorder": 4,
                "linewidth": 0.1,
                "alpha": 1,
            },
        ],
        "Validation Data": [
            {
                "gdf": sub_aois_overlay_gdf,
                "edgecolor": "red",
                "facecolor": "none",
                "linewidth": 1.0,
                "linestyle": "solid",
                "zorder": 4,
            },
            {
                "gdf": aois_overlay_gdf,
                "edgecolor": "black",
                "facecolor": "none",
                "linewidth": 1.0,
                "linestyle": "--",
                "zorder": 5,
            },
            {
                "gdf": val_gdf,
                "zorder": 4,
                "linewidth": 0.1,
                "alpha": 1,
            },
        ],
        "Carto": [
            {
                "gdf": sub_aois_overlay_gdf,
                "edgecolor": "red",
                "facecolor": "none",
                "linewidth": 1.0,
                "linestyle": "solid",
                "zorder": 4,
            },
            {
                "gdf": aois_overlay_gdf,
                "edgecolor": "black",
                "facecolor": "none",
                "linewidth": 1.0,
                "linestyle": "--",
                "zorder": 5,
            },
        ],
        "RF Q1": [
            {
                "gdf": sub_aois_overlay_gdf,
                "edgecolor": "red",
                "facecolor": "none",
                "linewidth": 1.0,
                "linestyle": "solid",
                "zorder": 4,
            },
            {
                "gdf": aois_overlay_gdf,
                "edgecolor": "black",
                "facecolor": "none",
                "linewidth": 1.0,
                "linestyle": "--",
                "zorder": 5,
            },
        ],
        "RF Q2": [
            {
                "gdf": sub_aois_overlay_gdf,
                "edgecolor": "red",
                "facecolor": "none",
                "linewidth": 1.0,
                "linestyle": "solid",
                "zorder": 4,
            },
            {
                "gdf": aois_overlay_gdf,
                "edgecolor": "black",
                "facecolor": "none",
                "linewidth": 1.0,
                "linestyle": "--",
                "zorder": 5,
            },
        ],
        "RF Q3": [
            {
                "gdf": sub_aois_overlay_gdf,
                "edgecolor": "red",
                "facecolor": "none",
                "linewidth": 1.0,
                "linestyle": "solid",
                "zorder": 4,
            },
            {
                "gdf": aois_overlay_gdf,
                "edgecolor": "black",
                "facecolor": "none",
                "linewidth": 1.0,
                "linestyle": "--",
                "zorder": 5,
            },
        ],
        "RF Q4": [
            {
                "gdf": sub_aois_overlay_gdf,
                "edgecolor": "red",
                "facecolor": "none",
                "linewidth": 1.0,
                "linestyle": "solid",
                "zorder": 4,
            },
            {
                "gdf": aois_overlay_gdf,
                "edgecolor": "black",
                "facecolor": "none",
                "linewidth": 1.0,
                "linestyle": "--",
                "zorder": 5,
            },
        ],
        "RF row stability": [
            {
                "gdf": sub_aois_overlay_gdf,
                "edgecolor": "red",
                "facecolor": "none",
                "linewidth": 1.0,
                "linestyle": "solid",
                "zorder": 4,
            },
            {
                "gdf": aois_overlay_gdf,
                "edgecolor": "black",
                "facecolor": "none",
                "linewidth": 1.0,
                "linestyle": "--",
                "zorder": 5,
            },
        ],
        "RF stable pixels": [
            {
                "gdf": sub_aois_overlay_gdf,
                "edgecolor": "red",
                "facecolor": "none",
                "linewidth": 1.0,
                "linestyle": "solid",
                "zorder": 4,
            },
            {
                "gdf": aois_overlay_gdf,
                "edgecolor": "black",
                "facecolor": "none",
                "linewidth": 1.0,
                "linestyle": "--",
                "zorder": 5,
            },
        ],
        "RF unstable pixels": [
            {
                "gdf": sub_aois_overlay_gdf,
                "edgecolor": "red",
                "facecolor": "none",
                "linewidth": 1.0,
                "linestyle": "solid",
                "zorder": 4,
            },
            {
                "gdf": aois_overlay_gdf,
                "edgecolor": "black",
                "facecolor": "none",
                "linewidth": 1.0,
                "linestyle": "--",
                "zorder": 5,
            },
        ],
    }

    # ==================== ROW STABILITY ====================
    # ROW STABILITY (LAST COLS)
    RF_row_stability_config = {
        "RF row stability": [
            {
                "raster_path": RF_row_stability_Q1234_path,
                "color_df": RF_row_stability_color_df,
                "alpha_map": {
                    1: 1.0,    
                    2: 1.0,   
                    4: 1.0,
                    5: 1.0,
                }
            },
        ]
    }

    RF_row_stable_pixels_config = {
        "RF stable pixels": [
            {
                "stability_path": RF_row_stability_Q1234_path,
                "modal_class_path": RF_row_modal_class_Q1234_path,
                "color_df": colors_df_no_remaining,
                "stabilities_to_include": [1, 2],
                "alpha_map": {
                    1: 1.0,
                    2: 0.5,
                },
            }
        ]
    }

    RF_row_unstable_pixels_config = {
        "RF unstable pixels": [
            {
                "unstable_pixels_path": RF_row_unstable_pixels_Q1234_path,
                "alpha": 1.0,
                "zorder": 2,
                # "include_legend_keys": ["highly_unstable"],
                "exclude_legend_keys": ["highly_unstable"],
            }
        ]
    }


    # ==================== COL STABILITY ====================
    # COL STABILITY (BOTTOM ROWS)
    meta_rows_config = {
        "Stability": {
            "base_columns": ["Carto", "RF Q1", "RF Q2", "RF Q3", "RF Q4"],
            "background_config": {
                "Carto": {
                    "raster_path": pdok_imgs_path,
                    "colorscheme": "rgb",
                    "band_mapping": "auto",
                    "background_alpha_override": 0.5,
                },
                "RF Q1": {
                    "raster_path": pdok_imgs_path,
                    "colorscheme": "rgb",
                    "band_mapping": "auto",
                    "background_alpha_override": 0.5,
                },
                "RF Q2": {
                    "raster_path": pdok_imgs_path,
                    "colorscheme": "rgb",
                    "band_mapping": "auto",
                    "background_alpha_override": 0.5,
                },
                "RF Q3": {
                    "raster_path": pdok_imgs_path,
                    "colorscheme": "rgb",
                    "band_mapping": "auto",
                    "background_alpha_override": 0.5,
                },
                "RF Q4": {
                    "raster_path": pdok_imgs_path,
                    "colorscheme": "rgb",
                    "band_mapping": "auto",
                    "background_alpha_override": 0.5,
                },
            },
            "gdf_overlay_config": {
                "Carto": [
                    {
                        "gdf": sub_aois_overlay_gdf,
                        "edgecolor": "red",
                        "facecolor": "none",
                        "linewidth": 1.0,
                        "linestyle": "solid",
                        "zorder": 4,
                    },
                    {
                        "gdf": aois_overlay_gdf,
                        "edgecolor": "black",
                        "facecolor": "none",
                        "linewidth": 1.0,
                        "linestyle": "--",
                        "zorder": 5,
                    },
                ],
                "RF Q1": [
                    {
                        "gdf": sub_aois_overlay_gdf,
                        "edgecolor": "red",
                        "facecolor": "none",
                        "linewidth": 1.0,
                        "linestyle": "solid",
                        "zorder": 4,
                    },
                    {
                        "gdf": aois_overlay_gdf,
                        "edgecolor": "black",
                        "facecolor": "none",
                        "linewidth": 1.0,
                        "linestyle": "--",
                        "zorder": 5,
                    },
                ],
                "RF Q2": [
                    {
                        "gdf": sub_aois_overlay_gdf,
                        "edgecolor": "red",
                        "facecolor": "none",
                        "linewidth": 1.0,
                        "linestyle": "solid",
                        "zorder": 4,
                    },
                    {
                        "gdf": aois_overlay_gdf,
                        "edgecolor": "black",
                        "facecolor": "none",
                        "linewidth": 1.0,
                        "linestyle": "--",
                        "zorder": 5,
                    },
                ],
                "RF Q3": [
                    {
                        "gdf": sub_aois_overlay_gdf,
                        "edgecolor": "red",
                        "facecolor": "none",
                        "linewidth": 1.0,
                        "linestyle": "solid",
                        "zorder": 4,
                    },
                    {
                        "gdf": aois_overlay_gdf,
                        "edgecolor": "black",
                        "facecolor": "none",
                        "linewidth": 1.0,
                        "linestyle": "--",
                        "zorder": 5,
                    },
                ],
                "RF Q4": [
                    {
                        "gdf": sub_aois_overlay_gdf,
                        "edgecolor": "red",
                        "facecolor": "none",
                        "linewidth": 1.0,
                        "linestyle": "solid",
                        "zorder": 4,
                    },
                    {
                        "gdf": aois_overlay_gdf,
                        "edgecolor": "black",
                        "facecolor": "none",
                        "linewidth": 1.0,
                        "linestyle": "--",
                        "zorder": 5,
                    },
                ],
            },

            "RF_row_stability_config": {
                "Carto": [
                    {
                        "raster_path": carto_col_stability_path,
                        "color_df": RF_row_stability_color_df,
                        "alpha_map": {
                            1: 1.0,    
                            2: 1.0,   
                            3: 1.0,
                            4: 1.0,
                            5: 1.0,
                        },
                    }
                ],
                "RF Q1": [
                    {
                        "raster_path": RF_col_stability_Q1_path,
                        "color_df": RF_row_stability_color_df,
                        "alpha_map": {
                            1: 1.0,    
                            2: 1.0,   
                            3: 1.0,
                            4: 1.0,
                            5: 1.0,
                        },
                    }
                ],
                "RF Q2": [
                    {
                        "raster_path": RF_col_stability_Q2_path,
                        "color_df": RF_row_stability_color_df,
                        "alpha_map": {
                            1: 1.0,    
                            2: 1.0,   
                            3: 1.0,
                            4: 1.0,
                            5: 1.0,
                        },
                    }
                ],
                "RF Q3": [
                    {
                        "raster_path": RF_col_stability_Q3_path,
                        "color_df": RF_row_stability_color_df,
                        "alpha_map": {
                            1: 1.0,    
                            2: 1.0,   
                            3: 1.0,
                            4: 1.0,
                            5: 1.0,
                        },
                    }
                ],
                "RF Q4": [
                    {
                        "raster_path": RF_col_stability_Q4_path,
                        "color_df": RF_row_stability_color_df,
                        "alpha_map": {
                            1: 1.0,    
                            2: 1.0,   
                            3: 1.0,
                            4: 1.0,
                            5: 1.0,
                        },
                    }
                ],
            },
            "illustrations_config": {
                "Carto": {
                    "illustrations": ["stability_percentages", "north_arrow", "scalebar"],
                    "settings": {
                        "stability_percentages": {
                            "color_df": RF_row_stability_color_df,
                            "raster_path_template": carto_col_stability_path,
                        },
                    },
                },
                "RF Q1": {
                    "illustrations": ["stability_percentages", "north_arrow", "scalebar"],
                    "settings": {
                        "stability_percentages": {
                            "color_df": RF_row_stability_color_df,
                            "raster_path_template": RF_col_stability_Q1_path,
                        },
                    },
                },
                "RF Q2": {
                    "illustrations": ["stability_percentages", "north_arrow", "scalebar"],
                    "settings": {
                        "stability_percentages": {
                            "color_df": RF_row_stability_color_df,
                            "raster_path_template": RF_col_stability_Q2_path,
                        },
                    },
                },
                "RF Q3": {
                    "illustrations": ["stability_percentages", "north_arrow", "scalebar"],
                    "settings": {
                        "stability_percentages": {
                            "color_df": RF_row_stability_color_df,
                            "raster_path_template": RF_col_stability_Q3_path,
                        },
                    },
                },
                "RF Q4": {
                    "illustrations": ["stability_percentages", "north_arrow", "scalebar"],
                    "settings": {
                        "stability_percentages": {
                            "color_df": RF_row_stability_color_df,
                            "raster_path_template": RF_col_stability_Q4_path,
                        },
                    },
                },
            },
        },

        "Stable Pixels": {
            "base_columns": ["Carto", "RF Q1", "RF Q2", "RF Q3", "RF Q4"],
            "background_config": {
                "Carto": {
                    "raster_path": pdok_imgs_path,
                    "colorscheme": "rgb",
                    "band_mapping": "auto",
                    "background_alpha_override": 0.5,
                },
                "RF Q1": {
                    "raster_path": pdok_imgs_path,
                    "colorscheme": "rgb",
                    "band_mapping": "auto",
                    "background_alpha_override": 0.5,
                },
                "RF Q2": {
                    "raster_path": pdok_imgs_path,
                    "colorscheme": "rgb",
                    "band_mapping": "auto",
                    "background_alpha_override": 0.5,
                },
                "RF Q3": {
                    "raster_path": pdok_imgs_path,
                    "colorscheme": "rgb",
                    "band_mapping": "auto",
                    "background_alpha_override": 0.5,
                },
                "RF Q4": {
                    "raster_path": pdok_imgs_path,
                    "colorscheme": "rgb",
                    "band_mapping": "auto",
                    "background_alpha_override": 0.5,
                },
            },
            "gdf_overlay_config": {
                "Carto": [
                    {
                        "gdf": sub_aois_overlay_gdf,
                        "edgecolor": "red",
                        "facecolor": "none",
                        "linewidth": 1.0,
                        "linestyle": "solid",
                        "zorder": 4,
                    },
                    {
                        "gdf": aois_overlay_gdf,
                        "edgecolor": "black",
                        "facecolor": "none",
                        "linewidth": 1.0,
                        "linestyle": "--",
                        "zorder": 5,
                    },
                ],
                "RF Q1": [
                    {
                        "gdf": sub_aois_overlay_gdf,
                        "edgecolor": "red",
                        "facecolor": "none",
                        "linewidth": 1.0,
                        "linestyle": "solid",
                        "zorder": 4,
                    },
                    {
                        "gdf": aois_overlay_gdf,
                        "edgecolor": "black",
                        "facecolor": "none",
                        "linewidth": 1.0,
                        "linestyle": "--",
                        "zorder": 5,
                    },
                ],
                "RF Q2": [
                    {
                        "gdf": sub_aois_overlay_gdf,
                        "edgecolor": "red",
                        "facecolor": "none",
                        "linewidth": 1.0,
                        "linestyle": "solid",
                        "zorder": 4,
                    },
                    {
                        "gdf": aois_overlay_gdf,
                        "edgecolor": "black",
                        "facecolor": "none",
                        "linewidth": 1.0,
                        "linestyle": "--",
                        "zorder": 5,
                    },
                ],
                "RF Q3": [
                    {
                        "gdf": sub_aois_overlay_gdf,
                        "edgecolor": "red",
                        "facecolor": "none",
                        "linewidth": 1.0,
                        "linestyle": "solid",
                        "zorder": 4,
                    },
                    {
                        "gdf": aois_overlay_gdf,
                        "edgecolor": "black",
                        "facecolor": "none",
                        "linewidth": 1.0,
                        "linestyle": "--",
                        "zorder": 5,
                    },
                ],
                "RF Q4": [
                    {
                        "gdf": sub_aois_overlay_gdf,
                        "edgecolor": "red",
                        "facecolor": "none",
                        "linewidth": 1.0,
                        "linestyle": "solid",
                        "zorder": 4,
                    },
                    {
                        "gdf": aois_overlay_gdf,
                        "edgecolor": "black",
                        "facecolor": "none",
                        "linewidth": 1.0,
                        "linestyle": "--",
                        "zorder": 5,
                    },
                ],
            },
            "RF_row_stable_pixels_config": {
                "Carto": [
                    {
                        "stability_path": carto_col_stability_path,
                        "modal_class_path": carto_modal_class_path,
                        "color_df": colors_df_no_remaining,
                        "stabilities_to_include": [1, 2, 3],
                        "alpha": 1.0,
                    }
                ],
                "RF Q1": [
                    {
                        "stability_path": RF_col_stability_Q1_path,
                        "modal_class_path": RF_col_modal_class_Q1_path,
                        "color_df": colors_df_no_remaining,
                        "stabilities_to_include": [1, 2, 3],
                        "alpha": 1.0,
                    }
                ],
                "RF Q2": [
                    {
                        "stability_path": RF_col_stability_Q2_path,
                        "modal_class_path": RF_col_modal_class_Q2_path,
                        "color_df": colors_df_no_remaining,
                        "stabilities_to_include": [1, 2, 3],
                        "alpha": 1.0,
                    }
                ],
                "RF Q3": [
                    {
                        "stability_path": RF_col_stability_Q3_path,
                        "modal_class_path": RF_col_modal_class_Q3_path,
                        "color_df": colors_df_no_remaining,
                        "stabilities_to_include": [1, 2, 3],
                        "alpha": 1.0,
                    }
                ],
                "RF Q4": [
                    {
                        "stability_path": RF_col_stability_Q4_path,
                        "modal_class_path": RF_col_modal_class_Q4_path,
                        "color_df": colors_df_no_remaining,
                        "stabilities_to_include": [1, 2, 3],
                        "alpha": 1.0,
                    }
                ],
            },
            "illustrations_config": {     
                "Carto": {
                    "illustrations": ["stability_tiles_legend", "north_arrow", "scalebar"],
                    "settings": {
                        "stability_tiles_legend": {
                            "color_df": colors_df_no_remaining,
                            "alpha_map": {1: 1.0, 2: 0.5},
                            "stability_labels": {
                                1: ">80% stable",
                                2: ">70-<80%",
                            },
                            "class_name_overrides": {
                                "Semi-Wet Nature": "Semi-Wet",
                            },
                        },
                    },
                },
                "RF Q1": {
                    "illustrations": ["stability_tiles_legend", "north_arrow", "scalebar"],
                    "settings": {
                        "stability_tiles_legend": {
                            "color_df": colors_df_no_remaining,
                            "alpha_map": {1: 1.0, 2: 0.5},
                            "stability_labels": {
                                1: ">80% stable",
                                2: ">70-<80%",
                            },
                            "class_name_overrides": {
                                "Semi-Wet Nature": "Semi-Wet",
                            },
                        },
                    },
                },
                "RF Q2": {
                    "illustrations": ["stability_tiles_legend", "north_arrow", "scalebar"],
                    "settings": {
                        "stability_tiles_legend": {
                            "color_df": colors_df_no_remaining,
                            "alpha_map": {1: 1.0, 2: 0.5},
                            "stability_labels": {
                                1: ">80% stable",
                                2: ">70-<80%",
                            },
                            "class_name_overrides": {
                                "Semi-Wet Nature": "Semi-Wet",
                            },
                        },
                    },
                },
                "RF Q3": {
                    "illustrations": ["stability_tiles_legend", "north_arrow", "scalebar"],
                    "settings": {
                        "stability_tiles_legend": {
                            "color_df": colors_df_no_remaining,
                            "alpha_map": {1: 1.0, 2: 0.5},
                            "stability_labels": {
                                1: ">80% stable",
                                2: ">70-<80%",
                            },
                            "class_name_overrides": {
                                "Semi-Wet Nature": "Semi-Wet",
                            },
                        },
                    },
                },
                "RF Q4": {
                    "illustrations": ["stability_tiles_legend", "north_arrow", "scalebar"],
                    "settings": {
                        "stability_tiles_legend": {
                            "color_df": colors_df_no_remaining,
                            "alpha_map": {1: 1.0, 2: 0.5},
                            "stability_labels": {
                                1: ">80% stable",
                                2: ">70-<80%",
                            },
                            "class_name_overrides": {
                                "Semi-Wet Nature": "Semi-Wet",
                            },
                        },
                    },
                },
            },
        },
        
        "Unstable Pixels": {
            "base_columns": ["Carto", "RF Q1", "RF Q2", "RF Q3", "RF Q4"],
            "background_config": {
                "Carto": {
                    "raster_path": pdok_imgs_path,
                    "colorscheme": "rgb",
                    "band_mapping": "auto",
                    "background_alpha_override": 0.5,
                },
                "RF Q1": {
                    "raster_path": pdok_imgs_path,
                    "colorscheme": "rgb",
                    "band_mapping": "auto",
                    "background_alpha_override": 0.5,
                },
                "RF Q2": {
                    "raster_path": pdok_imgs_path,
                    "colorscheme": "rgb",
                    "band_mapping": "auto",
                    "background_alpha_override": 0.5,
                },
                "RF Q3": {
                    "raster_path": pdok_imgs_path,
                    "colorscheme": "rgb",
                    "band_mapping": "auto",
                    "background_alpha_override": 0.5,
                },
                "RF Q4": {
                    "raster_path": pdok_imgs_path,
                    "colorscheme": "rgb",
                    "band_mapping": "auto",
                    "background_alpha_override": 0.5,
                },
            },
            "gdf_overlay_config": {
                "Carto": [
                    {
                        "gdf": sub_aois_overlay_gdf,
                        "edgecolor": "red",
                        "facecolor": "none",
                        "linewidth": 1.0,
                        "linestyle": "solid",
                        "zorder": 4,
                    },
                    {
                        "gdf": aois_overlay_gdf,
                        "edgecolor": "black",
                        "facecolor": "none",
                        "linewidth": 1.0,
                        "linestyle": "--",
                        "zorder": 5,
                    },
                ],
                "RF Q1": [
                    {
                        "gdf": sub_aois_overlay_gdf,
                        "edgecolor": "red",
                        "facecolor": "none",
                        "linewidth": 1.0,
                        "linestyle": "solid",
                        "zorder": 4,
                    },
                    {
                        "gdf": aois_overlay_gdf,
                        "edgecolor": "black",
                        "facecolor": "none",
                        "linewidth": 1.0,
                        "linestyle": "--",
                        "zorder": 5,
                    },
                ],
                "RF Q2": [
                    {
                        "gdf": sub_aois_overlay_gdf,
                        "edgecolor": "red",
                        "facecolor": "none",
                        "linewidth": 1.0,
                        "linestyle": "solid",
                        "zorder": 4,
                    },
                    {
                        "gdf": aois_overlay_gdf,
                        "edgecolor": "black",
                        "facecolor": "none",
                        "linewidth": 1.0,
                        "linestyle": "--",
                        "zorder": 5,
                    },
                ],
                "RF Q3": [
                    {
                        "gdf": sub_aois_overlay_gdf,
                        "edgecolor": "red",
                        "facecolor": "none",
                        "linewidth": 1.0,
                        "linestyle": "solid",
                        "zorder": 4,
                    },
                    {
                        "gdf": aois_overlay_gdf,
                        "edgecolor": "black",
                        "facecolor": "none",
                        "linewidth": 1.0,
                        "linestyle": "--",
                        "zorder": 5,
                    },
                ],
                "RF Q4": [
                    {
                        "gdf": sub_aois_overlay_gdf,
                        "edgecolor": "red",
                        "facecolor": "none",
                        "linewidth": 1.0,
                        "linestyle": "solid",
                        "zorder": 4,
                    },
                    {
                        "gdf": aois_overlay_gdf,
                        "edgecolor": "black",
                        "facecolor": "none",
                        "linewidth": 1.0,
                        "linestyle": "--",
                        "zorder": 5,
                    },
                ],
            },
            "RF_row_unstable_pixels_config": {
                "Carto": [
                    {
                        "unstable_pixels_path": carto_unstable_pixels_path,
                        "alpha": 1.0,
                        "zorder": 3,
                        # "include_legend_keys": ["highly_unstable"],
                        "exclude_legend_keys": ["highly_unstable"],
                    }
                ],
                "RF Q1": [
                    {
                        "unstable_pixels_path": RF_col_unstable_pixels_Q1_path,
                        "alpha": 1.0,
                        "zorder": 3,
                        # "include_legend_keys": ["highly_unstable"],
                        "exclude_legend_keys": ["highly_unstable"],
                    }
                ],
                "RF Q2": [
                    {
                        "unstable_pixels_path": RF_col_unstable_pixels_Q2_path,
                        "alpha": 1.0,
                        "zorder": 3,
                        # "include_legend_keys": ["highly_unstable"],
                        "exclude_legend_keys": ["highly_unstable"],
                    }
                ],
                "RF Q3": [
                    {
                        "unstable_pixels_path": RF_col_unstable_pixels_Q3_path,
                        "alpha": 1.0,
                        "zorder": 3,
                        # "include_legend_keys": ["highly_unstable"],
                        "exclude_legend_keys": ["highly_unstable"],
                    }
                ],
                "RF Q4": [
                    {
                        "unstable_pixels_path": RF_col_unstable_pixels_Q4_path,
                        "alpha": 1.0,
                        "zorder": 3,
                        # "include_legend_keys": ["highly_unstable"],
                        "exclude_legend_keys": ["highly_unstable"],
                    }
                ],
            },
            "illustrations_config": {
                "Carto": {
                    "illustrations": ["unstable_pixels_legend", "north_arrow", "scalebar"],
                    "settings": {
                        "unstable_pixels_legend": {
                            "color_df": unstable_pixels_df,
                        },
                    },
                },
                "RF Q1": {
                    "illustrations": ["unstable_pixels_legend", "north_arrow", "scalebar"],
                    "settings": {
                        "unstable_pixels_legend": {
                            "color_df": unstable_pixels_df,
                        },
                    },
                },
                "RF Q2": {
                    "illustrations": ["unstable_pixels_legend", "north_arrow", "scalebar"],
                    "settings": {
                        "unstable_pixels_legend": {
                            "color_df": unstable_pixels_df,
                        },
                    },
                },
                "RF Q3": {
                    "illustrations": ["unstable_pixels_legend", "north_arrow", "scalebar"],
                    "settings": {
                        "unstable_pixels_legend": {
                            "color_df": unstable_pixels_df,
                        },
                    },
                },
                "RF Q4": {
                    "illustrations": ["unstable_pixels_legend", "north_arrow", "scalebar"],
                    "settings": {
                        "unstable_pixels_legend": {
                            "color_df": unstable_pixels_df,
                        },
                    },
                },
            },
        },
    }


    # ==================== HEADER ROW ====================
    # HEADER ROW
    all_pixels_plotting_gdf['years'] = all_pixels_plotting_gdf['years'].astype(int)
    train_gdf['years'] = train_gdf['years'].astype(int)
    val_gdf['years'] = val_gdf['years'].astype(int)

    header_row_config = {
        "window_gdf": windows_gdf[windows_gdf["location"] == "Veluwe overview"],
        "row_label": "Overview",
        "background_config": {
            "All Data": {
                "raster_path": pdok_full_veluwe_imgs,
                "colorscheme": "rgb",
                "band_mapping": "auto",
                "alpha_override": 0.5,
            },
            "Training Data": {
                "raster_path": pdok_full_veluwe_imgs,
                "colorscheme": "rgb",
                "band_mapping": "auto",
                "alpha_override": 0.5,
            },
            "Validation Data": {
                "raster_path": pdok_full_veluwe_imgs,
                "colorscheme": "rgb",
                "band_mapping": "auto",
                "alpha_override": 0.5,
            },        
            "Carto": {
                "raster_path": pdok_full_veluwe_imgs,
                "colorscheme": "rgb",
                "band_mapping": "auto",
                "alpha_override": 0.5,
            },
            "RF Q1": {
                "raster_path": pdok_full_veluwe_imgs,
                "colorscheme": "rgb",
                "band_mapping": "auto",
                "alpha_override": 0.5,
            },
            "RF Q2": {
                "raster_path": pdok_full_veluwe_imgs,
                "colorscheme": "rgb",
                "band_mapping": "auto",
                "alpha_override": 0.5,
            },
            "RF Q3": {
                "raster_path": pdok_full_veluwe_imgs,
                "colorscheme": "rgb",
                "band_mapping": "auto",
                "alpha_override": 0.5,
            },
            "RF Q4": {
                "raster_path": pdok_full_veluwe_imgs,
                "colorscheme": "rgb",
                "band_mapping": "auto",
                "alpha_override": 0.5,
            },
            "RF row stability": {
                "raster_path": pdok_full_veluwe_imgs,
                "colorscheme": "rgb",
                "band_mapping": "auto",
                "alpha_override": 0.2,
            },
            "RF stable pixels": {
                "raster_path": pdok_full_veluwe_imgs,
                "colorscheme": "rgb",
                "band_mapping": "auto",
                "alpha_override": 0.2,
            },
            "RF unstable pixels": {
                "raster_path": pdok_full_veluwe_imgs,
                "colorscheme": "rgb",
                "band_mapping": "auto",
                "alpha_override": 0.2,
            },
        },
        "gdf_overlay_config": {
            "All Data": [
                    {
                        "gdf": veluwe_polygon__n2000__UTM32631__gdf,
                    },
                    {
                        "gdf": header_vis_plot_gdf,
                        "edgecolor": "red",
                        "facecolor": "none",
                        "linewidth": 0.7,
                    },
                    {
                        "gdf": all_pixels_plotting_gdf,
                        "zorder": 4,
                        "linewidth": 0.1,
                        "alpha": 1,
                    },

            ],
            "Training Data": [
                    {
                        "gdf": veluwe_polygon__n2000__UTM32631__gdf,
                    },
                    {
                        "gdf": header_vis_plot_gdf,
                        "edgecolor": "red",
                        "facecolor": "none",
                        "linewidth": 0.7,
                    },
                    {
                        "gdf": train_gdf,
                        "zorder": 4,
                        "linewidth": 0.1,
                        "alpha": 1,
                    },
            ],
            "Validation Data": [
                    {
                        "gdf": veluwe_polygon__n2000__UTM32631__gdf,
                    },
                    {
                        "gdf": header_vis_plot_gdf,
                        "edgecolor": "red",
                        "facecolor": "none",
                        "linewidth": 0.7,
                    },
                    {
                        "gdf": val_gdf,
                        "zorder": 4,
                        "linewidth": 0.1,
                        "alpha": 1,
                    },
            ],
            "Carto": [
                    {
                        "gdf": veluwe_polygon__n2000__UTM32631__gdf,
                    },
                    {
                        "gdf": header_vis_plot_gdf,
                        "edgecolor": "red",
                        "facecolor": "none",
                        "linewidth": 0.7,
                    },
            ],
            "RF Q1": [
                    {
                        "gdf": veluwe_polygon__n2000__UTM32631__gdf,
                    },
                    {
                        "gdf": header_vis_plot_gdf,
                        "edgecolor": "red",
                        "facecolor": "none",
                        "linewidth": 0.7,
                    },
            ],
            "RF Q2": [
                    {
                        "gdf": veluwe_polygon__n2000__UTM32631__gdf,
                    },
                    {
                        "gdf": header_vis_plot_gdf,
                        "edgecolor": "red",
                        "facecolor": "none",
                        "linewidth": 0.7,
                    },
            ],
            "RF Q3": [
                    {
                        "gdf": veluwe_polygon__n2000__UTM32631__gdf,
                    },
                    {
                        "gdf": header_vis_plot_gdf,
                        "edgecolor": "red",
                        "facecolor": "none",
                        "linewidth": 0.7,
                    },
            ],
            "RF Q4": [
                    {
                        "gdf": veluwe_polygon__n2000__UTM32631__gdf,
                    },
                    {
                        "gdf": header_vis_plot_gdf,
                        "edgecolor": "red",
                        "facecolor": "none",
                        "linewidth": 0.7,
                    },
            ],
            "RF row stability": [
                    {
                        "gdf": veluwe_polygon__n2000__UTM32631__gdf,
                    },
                    {
                        "gdf": header_vis_plot_gdf,
                        "edgecolor": "red",
                        "facecolor": "none",
                        "linewidth": 0.7,
                    },
            ],
            "RF stable pixels": [
                    {
                        "gdf": veluwe_polygon__n2000__UTM32631__gdf,
                        "alpha": 1,
                        "zorder": 4,
                        "edgecolor": "black",
                        "facecolor": "none",
                        "linewidth": 0.7,
                    },
                    {
                        "gdf": header_vis_plot_gdf,
                        "edgecolor": "red",
                        "facecolor": "none",
                        "linewidth": 0.7,
                    },
            ],
            "RF unstable pixels": [
                    {
                        "gdf": veluwe_polygon__n2000__UTM32631__gdf,
                        "alpha": 1,
                        "zorder": 4,
                        "edgecolor": "black",
                        "facecolor": "none",
                        "linewidth": 0.7,
                    },
                    {
                        "gdf": header_vis_plot_gdf,
                        "edgecolor": "red",
                        "facecolor": "none",
                        "linewidth": 0.7,
                    },
            ],
        },
        "illustrations_config": {
            "All Data": {
                "illustrations": ["north_arrow"],
            },
            "Training Data": {
                "illustrations": ["north_arrow"],
            },
            "Validation Data": {
                "illustrations": ["north_arrow"],
            },
            "Carto": {
                "illustrations": ["north_arrow"],
            },
            "RF Q1": {
                "illustrations": ["north_arrow"],
            },
            "RF Q2": {
                "illustrations": ["north_arrow"],
            },
            "RF Q3": {
                "illustrations": ["north_arrow"],
            },
            "RF Q4": {
                "illustrations": ["north_arrow"],
            },
            "RF row stability": {
                "illustrations": ["north_arrow"],
                "settings": {
                    "consistency_legend": {
                        "color_df": RF_row_stability_color_df,
                    },
                },
            },
            "RF stable pixels": {
                "illustrations": ["north_arrow"],
                "settings": {
                    "legend": {
                        "color_df": colors_df_no_remaining,
                    },
                },
            },
            "RF unstable pixels": {
                "illustrations": ["north_arrow"],
            },
        },
    }


    # ==================== ILLUSTRATIONS ====================
    # ILLUSTRATIONS CONFIG
    illustrations_config = {
        # Global settings
        "_global": {
            "north_arrow": {
                "svg_path": north_arrow__path,
            },
            "scalebar": {},
            "legend": {
                "color_df": colors_df,
            },
            "consistency_legend": {
                "loc": "lower right",
                "title": "Agreement",
                "title_fontsize": 9,
                "fontsize": 8,
            },
            "class_performance": {
                "color_df": colors_df,
                "font_size": 8,
                "show_class_names": True, 
                "class_name_font_size": 8,
                "tile_size": 0.025,
                "tile_spacing": 0.01,
                "row_spacing_pts": 8.0,
                "column_spacing_pts": 64.0, 
                "class_name_overrides": {
                    "Semi-Wet Nature": "Semi-Wet",
                },
            },
            "class_pixel_counts": {
                "color_df": colors_df,
                "class_column": "type",
                "font_size": 8,
                "show_percentage": True,
                "show_class_names": True,
                "class_name_font_size": 8, 
                "tile_size": 0.025, 
                "tile_spacing": 0.01,
                "row_spacing_pts": 8.0,  
                "column_spacing_pts": 62.0,
                "horizontal_offset_pts": -2.0,
                "class_name_overrides": {
                    "Semi-Wet Nature": "Semi-Wet",
                },
            },
            "stability_percentages": {
                "font_size": 8,
                "tile_spacing": 0.01,
                "row_spacing_pts": 8.0,
                "column_spacing_pts": 55.0,
                "horizontal_offset_pts": -8.0,
                "percentage_horizontal_offset_pts": 4.0,
            },
            "stability_tiles_legend": {
                "font_size": 8,
                "show_class_names": True,
                "class_name_font_size": 8,
                "tile_size": 0.025,
                "tile_spacing": 0.01,
                "column_spacing": 0.33,
                "row_spacing_pts": 8.0,
                "horizontal_offset_pts": -2.0,
                "class_name_overrides": {
                    "Semi-Wet Nature": "Semi-Wet",
                },
            },
            "unstable_pixels_legend": {
                "font_size": 8,
                "tile_size": 0.025,
                "tile_spacing": 0.01,
                "row_spacing_pts": 8.0,  
                "left_margin": 0.001,      
                "tiles_per_row": 2,        
                "label_overrides": {
                    "Open water / Semi-Wet Nature": "Open water / Semi-Wet",
                    "Remaining / Semi-Wet Nature": "Remaining / Semi-Wet",
                    "Semi-Wet Nature / Wet Nature": "Semi-Wet / Wet",
                },
                "col1_horizontal_offset_pts": -1.0,
                "col2_horizontal_offset_pts": 2.0,
            },
        },
        
        # Column-specific configurations
        "All Data": {
            "illustrations": ["class_pixel_counts", "north_arrow", "scalebar"],
            "settings": {
                "class_pixel_counts": {
                    "gdf": all_pixels_plotting_gdf,
                    "year_column": "years",
                },
            },
        },
        
        "Training Data": {
            "illustrations": ["class_pixel_counts", "north_arrow", "scalebar"],
            "settings": {
                "class_pixel_counts": {
                    "gdf": train_gdf,
                    "year_column": "years",
                },
            },
        },
        
        "Validation Data": {
            "illustrations": ["class_pixel_counts", "north_arrow", "scalebar"],
            "settings": {
                "class_pixel_counts": {
                    "gdf": val_gdf,
                    "year_column": "years",
                },
            },
        },
        
        "Carto": {
            "illustrations": ["class_performance", "north_arrow", "scalebar"],
            "settings": {
                "class_performance": {
                    "metrics_df": RF_Q1_stats_df,
                },
            },
        },
        
        "RF Q1": {
            "illustrations": ["class_performance", "north_arrow", "scalebar"],
            "settings": {
                "class_performance": {
                    "metrics_df": RF_Q1_stats_df,
                },
            },
        },

        "RF Q2": {
            "illustrations": ["class_performance", "north_arrow", "scalebar"],
            "settings": {
                "class_performance": {
                    "metrics_df": RF_Q2_stats_df,
                },
            },
        },

        "RF Q3": {
            "illustrations": ["class_performance", "north_arrow", "scalebar"],
            "settings": {
                "class_performance": {
                    "metrics_df": RF_Q3_stats_df,
                },
            },
        },

        "RF Q4": {
            "illustrations": ["class_performance", "north_arrow", "scalebar"],
            "settings": {
                "class_performance": {
                    "metrics_df": RF_Q4_stats_df,
                },
            },
        },
        
        "RF row stability": {
            "illustrations": ["north_arrow", "scalebar", "stability_percentages"],
            "settings": {
                "stability_percentages": {
                    "color_df": RF_row_stability_color_df,
                    "raster_path_template": RF_row_stability_Q1234_path,
                },
            },
        },

        "RF stable pixels": {
            "illustrations": ["stability_tiles_legend", "north_arrow", "scalebar"],
            "settings": {
                "stability_tiles_legend": {
                    "color_df": colors_df_no_remaining,
                    "alpha_map": {1: 1.0, 2: 0.5},
                    "stability_labels": {
                        1: ">80% stable",
                        2: ">70-80%",
                    },
                },
            },
        },

        "RF unstable pixels": {
            "illustrations": ["north_arrow", "unstable_pixels_legend"],
            "settings": {
                "unstable_pixels_legend": {
                    "color_df": unstable_pixels_df,
                },
            },
        },
    }

    # ==================== GENERATE PLOTS ====================
    
    figures = plot_all_locations(
        locations_gdf=windows_gdf,
        background_config=background_config,
        gdf_overlay_config=gdf_overlay_config,
        illustrations_config=illustrations_config,
        RF_row_stability_config=RF_row_stability_config,
        RF_row_stable_pixels_config=RF_row_stable_pixels_config,
        RF_row_unstable_pixels_config=RF_row_unstable_pixels_config,
        header_row_config=header_row_config,
        meta_rows_config=meta_rows_config,
        locations=locations,
        vis_years=vis_years_description,
        years=vis_years,
        columns_to_vis=columns_to_vis,
        figsize=figsize,
        dpi=dpi,
        output_dir=result_plotting__dir,
        hab_selection=hab_selection,
        train_split_attempt=train_split_attempt,
        band_selection=raster_stack,
        overwrite=overwrite,
    )
    
    return figures