from pathlib import Path

#################################################################################
#                             FINDING EXTERNAL DRIVE                            #
#################################################################################
data_dir = Path(r'D:\Thesis\10.Thesis_Data')


################################################################################
#                              GENERAL POLYGONS                                #
################################################################################
# +============================================================================+
# |                             Veluwe polygons                                |
# +============================================================================+
# DIR
veluwe_polygon__gpkg_dir = data_dir / "00_basismaps_raw/veluwe_polygons"
# Veluwe n2000
veluwe_polygon__n2000__RD__gpkg_path = veluwe_polygon__gpkg_dir / "veluwe_polygon__n2000__RD__gpkg.gpkg"
veluwe_polygon__n2000__WGS84__gpkg_path = veluwe_polygon__gpkg_dir / "veluwe_polygon__n2000__WGS84__gpkg.gpkg"
veluwe_polygon__n2000__UTM32631__gpkg_path = veluwe_polygon__gpkg_dir / "veluwe_polygon__n2000__UTM32631__gpkg.gpkg"
# Veluwe ruim
veluwe_polygon__ruim__RD__gpkg_path = veluwe_polygon__gpkg_dir / "veluwe_polygon__ruim__RD__gpkg.gpkg"
veluwe_polygon__ruim__WGS84__gpkg_path = veluwe_polygon__gpkg_dir / "veluwe_polygon__ruim__WGS84__gpkg.gpkg"
veluwe_polygon__ruim__UTM32631__gpkg_path = veluwe_polygon__gpkg_dir / "veluwe_polygon__ruim__UTM32631__gpkg.gpkg"
# Veluwe aoi
veluwe_polygon__aoi__RD__gpkg_path = veluwe_polygon__gpkg_dir / "veluwe_polygon__aoi__RD__gpkg.gpkg"
veluwe_polygon__aoi__WGS84__gpkg_path = veluwe_polygon__gpkg_dir / "veluwe_polygon__aoi__WGS84__gpkg.gpkg"
veluwe_polygon__aoi__UTM32631__gpkg_path = veluwe_polygon__gpkg_dir / "veluwe_polygon__aoi__UTM32631__gpkg.gpkg"
# Veluwe subaoi
veluwe_polygon__subaoi__RD__gpkg_path = veluwe_polygon__gpkg_dir / "veluwe_polygon__subaoi__RD__gpkg.gpkg"
veluwe_polygon__subaoi__WGS84__gpkg_path = veluwe_polygon__gpkg_dir / "veluwe_polygon__subaoi__WGS84__gpkg.gpkg"
veluwe_polygon__subaoi__UTM32631__gpkg_path = veluwe_polygon__gpkg_dir / "veluwe_polygon__subaoi__UTM32631__gpkg.gpkg"
# Full plotting window
veluwe_polygon__full_plotting_window__UTM32631__gpkg_path = veluwe_polygon__gpkg_dir / "veluwe_polygon__full_plotting_window__UTM32631__gpkg.gpkg"
veluwe_plotting_windows__pkl_file_path = veluwe_polygon__gpkg_dir / "veluwe_plot_windows_gdf.pkl"


################################################################################
#                          HABITAT / VEGETATION DATA                           #
################################################################################
# +============================================================================+
# |                        T0 Habitatkart Veluwe [RAW]                         |
# +============================================================================+
# DIR
habitat_kart__raw__gelderland__gpkg_dir = data_dir / "00_basismaps_raw/habitatkartering_veluwe/T0_data_Gelderland"
habitat_kart__raw__website__gpkg_dir = data_dir / "00_basismaps_raw/habitatkartering_veluwe/T0_data_Website"
# GPKGs
habitat_kart__raw__gelderland__RD__gpkg_path = habitat_kart__raw__gelderland__gpkg_dir / "Gelderland_gpkg.gpkg"
habitat_kart__raw__website__RD__gpkg_path = habitat_kart__raw__website__gpkg_dir / "website_gpkg.gpkg"


# +============================================================================+
# |              T1 Vegetatiekartering Veluwe GPKG files [RAW]                 |
# +============================================================================+
# DIR
habitat_kart_T1__raw__all__gpkg_dir = data_dir / "00_basismaps_raw/habitatkartering_veluwe/T1_data_Gelderland/_00_T1_gpkgs"
# T1 GPKGs
habitat_kart_T1__raw__bosgr_2020__RD__gpkg_path = habitat_kart_T1__raw__all__gpkg_dir / "habitat_kart_T1__raw__bosgr_2020__RD__gpkg.gpkg"
habitat_kart_T1__raw__bosgr_2021__RD__gpkg_path = habitat_kart_T1__raw__all__gpkg_dir / "habitat_kart_T1__raw__bosgr_2021__RD__gpkg.gpkg"
habitat_kart_T1__raw__bosgr_2022__RD__gpkg_path = habitat_kart_T1__raw__all__gpkg_dir / "habitat_kart_T1__raw__bosgr_2022__RD__gpkg.gpkg"
habitat_kart_T1__raw__bosgr_2023__RD__gpkg_path = habitat_kart_T1__raw__all__gpkg_dir / "habitat_kart_T1__raw__bosgr_2023__RD__gpkg.gpkg"
habitat_kart_T1__raw__bosgr_2024__RD__gpkg_path = habitat_kart_T1__raw__all__gpkg_dir / "habitat_kart_T1__raw__bosgr_2024__RD__gpkg.gpkg"
habitat_kart_T1__raw__glk_2020__RD__gpkg_path = habitat_kart_T1__raw__all__gpkg_dir / "habitat_kart_T1__raw__glk_2020__RD__gpkg.gpkg"
habitat_kart_T1__raw__glk_2021__RD__gpkg_path = habitat_kart_T1__raw__all__gpkg_dir / "habitat_kart_T1__raw__glk_2021__RD__gpkg.gpkg"
habitat_kart_T1__raw__nm_2020__RD__gpkg_path = habitat_kart_T1__raw__all__gpkg_dir / "habitat_kart_T1__raw__nm_2020__RD__gpkg.gpkg"
habitat_kart_T1__raw__nm_2021__RD__gpkg_path = habitat_kart_T1__raw__all__gpkg_dir / "habitat_kart_T1__raw__nm_2021__RD__gpkg.gpkg"
habitat_kart_T1__raw__nphv_2023__RD__gpkg_path = habitat_kart_T1__raw__all__gpkg_dir / "habitat_kart_T1__raw__nphv_2023__RD__gpkg.gpkg"
habitat_kart_T1__raw__nphv_2024__RD__gpkg_path = habitat_kart_T1__raw__all__gpkg_dir / "habitat_kart_T1__raw__nphv_2024__RD__gpkg.gpkg"
habitat_kart_T1__raw__ah_2022__RD__gpkg_path = habitat_kart_T1__raw__all__gpkg_dir / "habitat_kart_T1__raw__ah_2022__RD__gpkg.gpkg"
habitat_kart_T1__raw__eg_2024__RD__gpkg_path = habitat_kart_T1__raw__all__gpkg_dir / "habitat_kart_T1__raw__eg_2024__RD__gpkg.gpkg"
habitat_kart_T1__raw__isk_2024__RD__gpkg_path = habitat_kart_T1__raw__all__gpkg_dir / "habitat_kart_T1__raw__isk_2024__RD__gpkg.gpkg"
habitat_kart_T1__raw__vegpla_2020__RD__gpkg_path = habitat_kart_T1__raw__all__gpkg_dir / "habitat_kart_T1__raw__vegpla_2020__RD__gpkg.gpkg"
habitat_kart_T1__raw__vegpla_2021__RD__gpkg_path = habitat_kart_T1__raw__all__gpkg_dir / "habitat_kart_T1__raw__vegpla_2021__RD__gpkg.gpkg"
habitat_kart_T1__raw__vegpla_2021_2__RD__gpkg_path = habitat_kart_T1__raw__all__gpkg_dir / "habitat_kart_T1__raw__vegpla_2021_2__RD__gpkg.gpkg"
habitat_kart_T1__raw__vegpla_2022__RD__gpkg_path = habitat_kart_T1__raw__all__gpkg_dir / "habitat_kart_T1__raw__vegpla_2022__RD__gpkg.gpkg"
habitat_kart_T1__raw__vegpla_2023__RD__gpkg_path = habitat_kart_T1__raw__all__gpkg_dir / "habitat_kart_T1__raw__vegpla_2023__RD__gpkg.gpkg"
habitat_kart_T1__raw__vegpla_2023_2__RD__gpkg_path = habitat_kart_T1__raw__all__gpkg_dir / "habitat_kart_T1__raw__vegpla_2023_2__RD__gpkg.gpkg"
habitat_kart_T1__raw__vegpla_2024__RD__gpkg_path = habitat_kart_T1__raw__all__gpkg_dir / "habitat_kart_T1__raw__vegpla_2024__RD__gpkg.gpkg"



################################################################################
#                                 LGN RAW Rasters                              #
################################################################################
# DIR
lgn__nl__raw__rstr_dir = data_dir / "00_basismaps_raw/LGN"
# RSTRs
lgn2018__nl__raw__RD__rstr_path = lgn__nl__raw__rstr_dir / "LGN_2018/LGN_2018.tif"
lgn2019__nl__raw__RD__rstr_path = lgn__nl__raw__rstr_dir / "LGN_2019/LGN_2019.tif"
lgn2020__nl__raw__RD__rstr_path = lgn__nl__raw__rstr_dir / "LGN_2020/LGN_2020.tif"
lgn2021__nl__raw__RD__rstr_path = lgn__nl__raw__rstr_dir / "LGN_2021/LGN_2021.tif"
lgn2022__nl__raw__RD__rstr_path = lgn__nl__raw__rstr_dir / "LGN_2022/LGN_2022.tif"
lgn2023__nl__raw__RD__rstr_path = lgn__nl__raw__rstr_dir / "LGN_2023/LGN_2023.tif"
lgn2024__nl__raw__RD__rstr_path = lgn__nl__raw__rstr_dir / "LGN_2024/LGN_2024.tif"



################################################################################
#                                  MAP ADDITIONS                               #
################################################################################
# DIRs
pdok_imgs_raw__rstr_dir = data_dir / "00_basismaps_raw/pdok_imgs"
pdok_imgs_stacked__rstr_dir = data_dir / "01_basismaps_processed/pdok_imgs"
north_arrow__path = data_dir / "00_basismaps_raw/N_arrow.svg"
# Plot windows
plot_extent__deelenschev__UTM32631__gpkg_path = veluwe_polygon__gpkg_dir / "plot_extent__deelenschev__UTM32631__gpkg.gpkg"
plot_extent__gerritsfles__UTM32631__gpkg_path = veluwe_polygon__gpkg_dir / "plot_extent__gerritsfles__UTM32631__gpkg.gpkg"
plot_extent__kootwijkerveen__UTM32631__gpkg_path = veluwe_polygon__gpkg_dir / "plot_extent__kootwijkerveen__UTM32631__gpkg.gpkg"
plot_extent__mosterdveen__UTM32631__gpkg_path = veluwe_polygon__gpkg_dir / "plot_extent__mosterdveen__UTM32631__gpkg.gpkg"
plot_extent__speulderveld__UTM32631__gpkg_path = veluwe_polygon__gpkg_dir / "plot_extent__speulderveld__UTM32631__gpkg.gpkg"



################################################################################
#                                 00_common_dfs                                #
################################################################################
# +============================================================================+
# |                       n01_habitat_reference_dfs                            |
# +============================================================================+
# DIR
habitat_reference__df_dir = data_dir / "00_explanation_dfs"
# DFs
habitat_reference__WD__df_path = habitat_reference__df_dir / "habitat_reference__WD__df.pkl"
habitat_reference__s__df_path = habitat_reference__df_dir / "habitat_reference__s__df.pkl"
habitat_reference__UC__df_path = habitat_reference__df_dir / "habitat_reference__UC__df.pkl"
habitat_reference__WDNF__df_path = habitat_reference__df_dir / "habitat_reference__WDNF__df.pkl"


# +============================================================================+
# |                       n02_habitat_reference_dfs                            |
# +============================================================================+
# DIR
explanation__df_dir = data_dir / "00_explanation_dfs"
# DFs
explanation__hab_group_codes__df_path = explanation__df_dir / "explanation__hab_group_codes__df.pkl"
explanation__tmp_codes__df_path = explanation__df_dir / "explanation__tmp_codes__df.pkl"
explanation__names_concention__df_path = explanation__df_dir / "explanation__names_concention__df.pkl"


# +============================================================================+
# |                           n03_LGN_legend_dfs                               |
# +============================================================================+
# DIR
lgn_reference__df_dir = data_dir / "00_basismaps_raw/LGN/lgn_exp_dfs"
# DFs
lgn2018_reference__df_path = lgn_reference__df_dir / "lgn2018_reference__df.pkl"
lgn2019_reference__df_path = lgn_reference__df_dir / "lgn2019_reference__df.pkl"
lgn2020_reference__df_path = lgn_reference__df_dir / "lgn2020_reference__df.pkl"
lgn2021_reference__df_path = lgn_reference__df_dir / "lgn2021_reference__df.pkl"
lgn2022_reference__df_path = lgn_reference__df_dir / "lgn2022_reference__df.pkl"
lgn2023_reference__df_path = lgn_reference__df_dir / "lgn2023_reference__df.pkl"
lgn2024_reference__df_path = lgn_reference__df_dir / "lgn2024_reference__df.pkl"



################################################################################
#                            01_pre_processing_hab_kart                        #
################################################################################
# +============================================================================+
# |                         n01_T0_habitat_gpkg_processing                     |
# +============================================================================+
# --- Habitat kart Indexed GPKGs (saved as xlsx) --- #
# DIR
habitat_kart__indexed__xlsx_dir = data_dir / "01_basismaps_processed/habitatkartering_veluwe/00_habitatkartering_veluwe_indexed"
# XLSXs
habitat_kart__indexed__gelderland__RD__xlsx_path = habitat_kart__indexed__xlsx_dir / "habitat_kart_gelderland/habitat_kart__indexed__gelderland__RD__xlsx.xlsx"
habitat_kart__indexed__gelderland_minHR__RD__xlsx_path = habitat_kart__indexed__xlsx_dir / "habitat_kart_gelderland_minHR/habitat_kart__indexed__gelderland_minHR__RD__xlsx.xlsx"
habitat_kart__indexed__website__RD__xlsx_path = habitat_kart__indexed__xlsx_dir / "habitat_kart_website/habitat_kart__indexed__website__RD__xlsx.xlsx"
habitat_kart__indexed__website_plusHR__RD__xlsx_path = habitat_kart__indexed__xlsx_dir / "habitat_kart_website_plusHR/habitat_kart__indexed__website_plusHR__RD__xlsx.xlsx"

# --- Habitat kart Processed GPKGs --- #
# DIRs
habitat_kart__processed__gelderland__gpkg_dir = data_dir / "01_basismaps_processed/habitatkartering_veluwe/01_habitatkartering_veluwe_processed/habitat_kart_gelderland"
habitat_kart__processed__gelderland_minHR__gpkg_dir = data_dir / "01_basismaps_processed/habitatkartering_veluwe/01_habitatkartering_veluwe_processed/habitat_kart_gelderland_minHR"
habitat_kart__processed__website__gpkg_dir = data_dir / "01_basismaps_processed/habitatkartering_veluwe/01_habitatkartering_veluwe_processed/habitat_kart_website"
habitat_kart__processed__website_plusHR__gpkg_dir = data_dir / "01_basismaps_processed/habitatkartering_veluwe/01_habitatkartering_veluwe_processed/habitat_kart_website_plusHR"
# GPKGs RD
habitat_kart__processed__gelderland__RD__gpkg_path = habitat_kart__processed__gelderland__gpkg_dir / "habitat_kart__processed__gelderland__RD__gpkg.gpkg"
habitat_kart__processed__gelderland_minHR__RD__gpkg_path = habitat_kart__processed__gelderland_minHR__gpkg_dir / "habitat_kart__processed__gelderland_minHR__RD__gpkg.gpkg"
habitat_kart__processed__website__RD__gpkg_path = habitat_kart__processed__website__gpkg_dir / "habitat_kart__processed__website__RD__gpkg.gpkg"
habitat_kart__processed__website_plusHR__RD__gpkg_path = habitat_kart__processed__website_plusHR__gpkg_dir / "habitat_kart__processed__website_plusHR__RD__gpkg.gpkg"
# GPKGs WGS84
habitat_kart__processed__gelderland__WGS84__gpkg_path = habitat_kart__processed__gelderland__gpkg_dir / "habitat_kart__processed__gelderland__WGS84__gpkg.gpkg"
habitat_kart__processed__gelderland_minHR__WGS84__gpkg_path = habitat_kart__processed__gelderland_minHR__gpkg_dir / "habitat_kart__processed__gelderland_minHR__WGS84__gpkg.gpkg"
habitat_kart__processed__website__WGS84__gpkg_path = habitat_kart__processed__website__gpkg_dir / "habitat_kart__processed__website__WGS84__gpkg.gpkg"
habitat_kart__processed__website_plusHR__WGS84__gpkg_path = habitat_kart__processed__website_plusHR__gpkg_dir / "habitat_kart__processed__website_plusHR__WGS84__gpkg.gpkg"
# GPKGs UTM32631
habitat_kart__processed__gelderland__UTM32631__gpkg_path = habitat_kart__processed__gelderland__gpkg_dir / "habitat_kart__processed__gelderland__UTM32631__gpkg.gpkg"
habitat_kart__processed__gelderland_minHR__UTM32631__gpkg_path = habitat_kart__processed__gelderland_minHR__gpkg_dir / "habitat_kart__processed__gelderland_minHR__UTM32631__gpkg.gpkg"
habitat_kart__processed__website__UTM32631__gpkg_path = habitat_kart__processed__website__gpkg_dir / "habitat_kart__processed__website__UTM32631__gpkg.gpkg"
habitat_kart__processed__website_plusHR__UTM32631__gpkg_path = habitat_kart__processed__website_plusHR__gpkg_dir / "habitat_kart__processed__website_plusHR__UTM32631__gpkg.gpkg"


# +============================================================================+
# |                             n03_extract_OW_from_LGN                        |
# +============================================================================+
# --- LGN Veluwe clip ruim Maps --- #
# DIR
lgn__ruim__clip__rstr_dir = data_dir / "01_basismaps_processed/lgn/lgn__ruim__clip__rstrs"
# RSTRs RD
lgn2018__ruim__clip__RD__rstr_path = lgn__ruim__clip__rstr_dir / "lgn2018__ruim__clip__RD__rstr.tif"
lgn2019__ruim__clip__RD__rstr_path = lgn__ruim__clip__rstr_dir / "lgn2019__ruim__clip__RD__rstr.tif"
lgn2020__ruim__clip__RD__rstr_path = lgn__ruim__clip__rstr_dir / "lgn2020__ruim__clip__RD__rstr.tif"
lgn2021__ruim__clip__RD__rstr_path = lgn__ruim__clip__rstr_dir / "lgn2021__ruim__clip__RD__rstr.tif"
lgn2022__ruim__clip__RD__rstr_path = lgn__ruim__clip__rstr_dir / "lgn2022__ruim__clip__RD__rstr.tif"
lgn2023__ruim__clip__RD__rstr_path = lgn__ruim__clip__rstr_dir / "lgn2023__ruim__clip__RD__rstr.tif"
lgn2024__ruim__clip__RD__rstr_path = lgn__ruim__clip__rstr_dir / "lgn2024__ruim__clip__RD__rstr.tif"
# RSTRs WGS84
lgn2018__ruim__clip__WGS84__rstr_path = lgn__ruim__clip__rstr_dir / "lgn2018__ruim__clip__WGS84__rstr.tif"
lgn2019__ruim__clip__WGS84__rstr_path = lgn__ruim__clip__rstr_dir / "lgn2019__ruim__clip__WGS84__rstr.tif"
lgn2020__ruim__clip__WGS84__rstr_path = lgn__ruim__clip__rstr_dir / "lgn2020__ruim__clip__WGS84__rstr.tif"
lgn2021__ruim__clip__WGS84__rstr_path = lgn__ruim__clip__rstr_dir / "lgn2021__ruim__clip__WGS84__rstr.tif"
lgn2022__ruim__clip__WGS84__rstr_path = lgn__ruim__clip__rstr_dir / "lgn2022__ruim__clip__WGS84__rstr.tif"
lgn2023__ruim__clip__WGS84__rstr_path = lgn__ruim__clip__rstr_dir / "lgn2023__ruim__clip__WGS84__rstr.tif"
lgn2024__ruim__clip__WGS84__rstr_path = lgn__ruim__clip__rstr_dir / "lgn2024__ruim__clip__WGS84__rstr.tif"
# RSTRs UTM32631
lgn2018__ruim__clip__UTM32631__rstr_path = lgn__ruim__clip__rstr_dir / "lgn2018__ruim__clip__UTM32631__rstr.tif"
lgn2019__ruim__clip__UTM32631__rstr_path = lgn__ruim__clip__rstr_dir / "lgn2019__ruim__clip__UTM32631__rstr.tif"
lgn2020__ruim__clip__UTM32631__rstr_path = lgn__ruim__clip__rstr_dir / "lgn2020__ruim__clip__UTM32631__rstr.tif"
lgn2021__ruim__clip__UTM32631__rstr_path = lgn__ruim__clip__rstr_dir / "lgn2021__ruim__clip__UTM32631__rstr.tif"
lgn2022__ruim__clip__UTM32631__rstr_path = lgn__ruim__clip__rstr_dir / "lgn2022__ruim__clip__UTM32631__rstr.tif"
lgn2023__ruim__clip__UTM32631__rstr_path = lgn__ruim__clip__rstr_dir / "lgn2023__ruim__clip__UTM32631__rstr.tif"
lgn2024__ruim__clip__UTM32631__rstr_path = lgn__ruim__clip__rstr_dir / "lgn2024__ruim__clip__UTM32631__rstr.tif"


# --- LGN Veluwe clip N2000 Maps --- #
# DIR
lgn__n2000__clip__rstr_dir = data_dir / "01_basismaps_processed/lgn/lgn__n2000__clip__rstrs"
# RSTRs RD
lgn2018__n2000__clip__RD__rstr_path = lgn__n2000__clip__rstr_dir / "lgn2018__n2000__clip__RD__rstr.tif"
lgn2019__n2000__clip__RD__rstr_path = lgn__n2000__clip__rstr_dir / "lgn2019__n2000__clip__RD__rstr.tif"
lgn2020__n2000__clip__RD__rstr_path = lgn__n2000__clip__rstr_dir / "lgn2020__n2000__clip__RD__rstr.tif"
lgn2021__n2000__clip__RD__rstr_path = lgn__n2000__clip__rstr_dir / "lgn2021__n2000__clip__RD__rstr.tif"
lgn2022__n2000__clip__RD__rstr_path = lgn__n2000__clip__rstr_dir / "lgn2022__n2000__clip__RD__rstr.tif"
lgn2023__n2000__clip__RD__rstr_path = lgn__n2000__clip__rstr_dir / "lgn2023__n2000__clip__RD__rstr.tif"
lgn2024__n2000__clip__RD__rstr_path = lgn__n2000__clip__rstr_dir / "lgn2024__n2000__clip__RD__rstr.tif"
# RSTRs WGS84
lgn2018__n2000__clip__WGS84__rstr_path = lgn__n2000__clip__rstr_dir / "lgn2018__n2000__clip__WGS84__rstr.tif"
lgn2019__n2000__clip__WGS84__rstr_path = lgn__n2000__clip__rstr_dir / "lgn2019__n2000__clip__WGS84__rstr.tif"
lgn2020__n2000__clip__WGS84__rstr_path = lgn__n2000__clip__rstr_dir / "lgn2020__n2000__clip__WGS84__rstr.tif"
lgn2021__n2000__clip__WGS84__rstr_path = lgn__n2000__clip__rstr_dir / "lgn2021__n2000__clip__WGS84__rstr.tif"
lgn2022__n2000__clip__WGS84__rstr_path = lgn__n2000__clip__rstr_dir / "lgn2022__n2000__clip__WGS84__rstr.tif"
lgn2023__n2000__clip__WGS84__rstr_path = lgn__n2000__clip__rstr_dir / "lgn2023__n2000__clip__WGS84__rstr.tif"
lgn2024__n2000__clip__WGS84__rstr_path = lgn__n2000__clip__rstr_dir / "lgn2024__n2000__clip__WGS84__rstr.tif"


# --- LGN_OW n2000 clip rstr --- #
# DIR
lgn_plusOW__n2000__clip__rstr_dir = data_dir / "01_basismaps_processed/lgn/lgn_plusOW__n2000__clip__rstrs"
# RSTRs RD
lgn2018_plusOW__n2000__clip__RD__rstr_path = lgn_plusOW__n2000__clip__rstr_dir / "lgn2018_plusOW__n2000__clip__RD__rstr.tif"
lgn2019_plusOW__n2000__clip__RD__rstr_path = lgn_plusOW__n2000__clip__rstr_dir / "lgn2019_plusOW__n2000__clip__RD__rstr.tif"
lgn2020_plusOW__n2000__clip__RD__rstr_path = lgn_plusOW__n2000__clip__rstr_dir / "lgn2020_plusOW__n2000__clip__RD__rstr.tif"
lgn2021_plusOW__n2000__clip__RD__rstr_path = lgn_plusOW__n2000__clip__rstr_dir / "lgn2021_plusOW__n2000__clip__RD__rstr.tif"
lgn2022_plusOW__n2000__clip__RD__rstr_path = lgn_plusOW__n2000__clip__rstr_dir / "lgn2022_plusOW__n2000__clip__RD__rstr.tif"
lgn2023_plusOW__n2000__clip__RD__rstr_path = lgn_plusOW__n2000__clip__rstr_dir / "lgn2023_plusOW__n2000__clip__RD__rstr.tif"
lgn2024_plusOW__n2000__clip__RD__rstr_path = lgn_plusOW__n2000__clip__rstr_dir / "lgn2024_plusOW__n2000__clip__RD__rstr.tif"
# RSTRs WGS84
lgn2018_plusOW__n2000__clip__WGS84__rstr_path = lgn_plusOW__n2000__clip__rstr_dir / "lgn2018_plusOW__n2000__clip__WGS84__rstr.tif"
lgn2019_plusOW__n2000__clip__WGS84__rstr_path = lgn_plusOW__n2000__clip__rstr_dir / "lgn2019_plusOW__n2000__clip__WGS84__rstr.tif"
lgn2020_plusOW__n2000__clip__WGS84__rstr_path = lgn_plusOW__n2000__clip__rstr_dir / "lgn2020_plusOW__n2000__clip__WGS84__rstr.tif"
lgn2021_plusOW__n2000__clip__WGS84__rstr_path = lgn_plusOW__n2000__clip__rstr_dir / "lgn2021_plusOW__n2000__clip__WGS84__rstr.tif"
lgn2022_plusOW__n2000__clip__WGS84__rstr_path = lgn_plusOW__n2000__clip__rstr_dir / "lgn2022_plusOW__n2000__clip__WGS84__rstr.tif"
lgn2023_plusOW__n2000__clip__WGS84__rstr_path = lgn_plusOW__n2000__clip__rstr_dir / "lgn2023_plusOW__n2000__clip__WGS84__rstr.tif"
lgn2024_plusOW__n2000__clip__WGS84__rstr_path = lgn_plusOW__n2000__clip__rstr_dir / "lgn2024_plusOW__n2000__clip__WGS84__rstr.tif"



################################################################################
#                            01_pre_processing_sat_obs                         #
################################################################################
# +============================================================================+
# |                              n01_exp_sat_raster                            |
# +============================================================================+
# --- sentinel 2 test rasters ---
# DIRs
S2_rndm_img_test_raster_paths_dir = data_dir / "00_basismaps_raw/sentinel_2_rasters/rndm_dates_images"
S2_mosaic_img_test_raster_paths_dir = data_dir / "00_basismaps_raw/sentinel_2_rasters"
# S2 Test1 rasters 
S2_test1_0_UTM32631_raster_path = S2_rndm_img_test_raster_paths_dir / "S2_rndm_date_1_images/S2C_MSIL2A_20250203T105311_N0511_R051_T31UFU_20250205T104011.SAFE.zip"
S2_test1_1_UTM32631_raster_path = S2_rndm_img_test_raster_paths_dir / "S2_rndm_date_1_images/S2C_MSIL2A_20250203T105311_N0511_R051_T31UGU_20250205T104011.SAFE.zip"
S2_test1_2_UTM32631_raster_path = S2_rndm_img_test_raster_paths_dir / "S2_rndm_date_1_images/S2C_MSIL2A_20250203T105311_N0511_R051_T31UFT_20250205T104011.SAFE.zip"
S2_test1_3_UTM32631_raster_path = S2_rndm_img_test_raster_paths_dir / "S2_rndm_date_1_images/S2C_MSIL2A_20250203T105311_N0511_R051_T31UGT_20250205T104011.SAFE.zip"
# S2 Test2 rasters
S2_test2_0_UTM32631_raster_path = S2_rndm_img_test_raster_paths_dir / "S2_rndm_date_2_images/S2A_MSIL2A_20220719T105041_N0510_R051_T31UFU_20240705T161711.zip"
S2_test2_1_UTM32631_raster_path = S2_rndm_img_test_raster_paths_dir / "S2_rndm_date_2_images/S2A_MSIL2A_20220719T105041_N0510_R051_T31UGU_20240705T161711.SAFE.zip"
S2_test2_2_UTM32631_raster_path = S2_rndm_img_test_raster_paths_dir / "S2_rndm_date_2_images/S2A_MSIL2A_20220719T105041_N0510_R051_T31UFT_20240705T161711.SAFE.zip"
S2_test2_3_UTM32631_raster_path = S2_rndm_img_test_raster_paths_dir / "S2_rndm_date_2_images/S2A_MSIL2A_20220719T105041_N0510_R051_T31UGT_20240705T161711.SAFE.zip"
# S2 mosaic Test1 raster
S2M_test1_0_UTM32631_raster_path = S2_rndm_img_test_raster_paths_dir / "S2M_rndm_date_1_images/Sentinel-2_mosaic_2022_Q1_31UFU_0_0.zip"
S2M_test1_1_UTM32631_raster_path = S2_rndm_img_test_raster_paths_dir / "S2M_rndm_date_1_images/Sentinel-2_mosaic_2022_Q1_31UGU_0_0.zip"
S2M_test1_2_UTM32631_raster_path = S2_rndm_img_test_raster_paths_dir / "S2M_rndm_date_1_images/Sentinel-2_mosaic_2022_Q1_31UFT_0_0.zip"
S2M_test1_3_UTM32631_raster_path = S2_rndm_img_test_raster_paths_dir / "S2M_rndm_date_1_images/Sentinel-2_mosaic_2022_Q1_31UGT_0_0.zip"
# S2 mosaic Test2 raster (10m)
S2M_test2_0_UTM32631_raster_path = S2_rndm_img_test_raster_paths_dir / "S2M_rndm_date_2_images/Sentinel-2_mosaic_2025_Q3_31UFU_0_0.zip"
S2M_test2_1_UTM32631_raster_path = S2_rndm_img_test_raster_paths_dir / "S2M_rndm_date_2_images/Sentinel-2_mosaic_2025_Q3_31UGU_0_0.zip"
S2M_test2_2_UTM32631_raster_path = S2_rndm_img_test_raster_paths_dir / "S2M_rndm_date_2_images/Sentinel-2_mosaic_2025_Q3_31UFT_0_0.zip"
S2M_test2_3_UTM32631_raster_path = S2_rndm_img_test_raster_paths_dir / "S2M_rndm_date_2_images/Sentinel-2_mosaic_2025_Q3_31UGT_0_0.zip"

# --- sentinel 2 Full mosaics ---
S2_test1_10m_UTM32631_mosaic_path = S2_mosaic_img_test_raster_paths_dir / "s2_test1/full_mosaic/S2_test1_10m_mosaic.tif"
S2_test1_20m_UTM32631_mosaic_path = S2_mosaic_img_test_raster_paths_dir / "s2_test1/full_mosaic/S2_test1_20m_mosaic.tif"
S2_test2_10m_UTM32631_mosaic_path = S2_mosaic_img_test_raster_paths_dir / "s2_test2/full_mosaic/S2_test2_10m_mosaic.tif"
S2_test2_20m_UTM32631_mosaic_path = S2_mosaic_img_test_raster_paths_dir / "s2_test2/full_mosaic/S2_test2_20m_mosaic.tif"
S2M_test1_UTM32631_mosaic_path = S2_mosaic_img_test_raster_paths_dir / "s2m_test1/full_mosaic/S2M_test1_mosaic.tif"
S2M_test2_UTM32631_mosaic_path = S2_mosaic_img_test_raster_paths_dir / "s2m_test2/full_mosaic/S2M_test2_mosaic.tif"

# --- sentinel 2 veluwe clip ruim raster ---
# DIRs
S2_test1_10m_veluwe_clip_ruim_UTM32631_rstr_dir = S2_mosaic_img_test_raster_paths_dir / "s2_test1/veluwe_clip_ruim"
S2_test1_20m_veluwe_clip_ruim_UTM32631_rstr_dir = S2_mosaic_img_test_raster_paths_dir / "s2_test1/veluwe_clip_ruim"
S2_test2_10m_veluwe_clip_ruim_UTM32631_rstr_dir = S2_mosaic_img_test_raster_paths_dir / "s2_test2/veluwe_clip_ruim"
S2_test2_20m_veluwe_clip_ruim_UTM32631_rstr_dir = S2_mosaic_img_test_raster_paths_dir / "s2_test2/veluwe_clip_ruim"
S2M_test1_veluwe_clip_ruim_UTM32631_rstr_dir = S2_mosaic_img_test_raster_paths_dir / "s2m_test1/veluwe_clip_ruim"
S2M_test2_veluwe_clip_ruim_UTM32631_rstr_dir = S2_mosaic_img_test_raster_paths_dir / "s2m_test2/veluwe_clip_ruim"
# RSTRs
S2_test1_10m_veluwe_clip_ruim_UTM32631_rstr_path = S2_test1_10m_veluwe_clip_ruim_UTM32631_rstr_dir / "S2_test1_10m_veluwe_clip_ruim_UTM32631_rstr.tif"
S2_test1_20m_veluwe_clip_ruim_UTM32631_rstr_path = S2_test1_20m_veluwe_clip_ruim_UTM32631_rstr_dir / "S2_test1_20m_veluwe_clip_ruim_UTM32631_rstr.tif"
S2_test2_10m_veluwe_clip_ruim_UTM32631_rstr_path = S2_test2_10m_veluwe_clip_ruim_UTM32631_rstr_dir / "S2_test2_10m_veluwe_clip_ruim_UTM32631_rstr.tif"
S2_test2_20m_veluwe_clip_ruim_UTM32631_rstr_path = S2_test2_20m_veluwe_clip_ruim_UTM32631_rstr_dir / "S2_test2_20m_veluwe_clip_ruim_UTM32631_rstr.tif"
S2M_test1_veluwe_clip_ruim_UTM32631_rstr_path = S2M_test1_veluwe_clip_ruim_UTM32631_rstr_dir / "S2M_test1_veluwe_clip_ruim_UTM32631_rstr.tif"
S2M_test2_veluwe_clip_ruim_UTM32631_rstr_path = S2M_test2_veluwe_clip_ruim_UTM32631_rstr_dir / "S2M_test2_veluwe_clip_ruim_UTM32631_rstr.tif"

# --- sentinel 2 Basisraster ---
# DIR
s2_basisraster__dir = data_dir / "01_basismaps_processed/sentinel_2_basisraster"
# RSTRs
s2_basisraster__10m__UTM32631__rstr_path = s2_basisraster__dir / "s2_basisraster__10m__UTM32631__rstr.tif"
s2_basisraster__20m__UTM32631__rstr_path = s2_basisraster__dir / "s2_basisraster__20m__UTM32631__rstr.tif"
s2_basisraster__10m__RD__rstr_path = s2_basisraster__dir / "s2_basisraster__10m__RD__rstr.tif"
s2_basisraster__20m__RD__rstr_path = s2_basisraster__dir / "s2_basisraster__20m__RD__rstr.tif"
s2_basisraster__10m__WGS84__rstr_path = s2_basisraster__dir / "s2_basisraster__10m__WGS84__rstr.tif"
s2_basisraster__20m__WGS84__rstr_path = s2_basisraster__dir / "s2_basisraster__20m__WGS84__rstr.tif"
# GPKGs
s2_basisraster__10m__UTM32631__gpkg_path = s2_basisraster__dir / "s2_basisraster__10m__UTM32631__gpkg.gpkg"
s2_basisraster__20m__UTM32631__gpkg_path = s2_basisraster__dir / "s2_basisraster__20m__UTM32631__gpkg.gpkg"
s2_basisraster__10m__RD__gpkg_path = s2_basisraster__dir / "s2_basisraster__10m__RD__gpkg.gpkg"
s2_basisraster__20m__RD__gpkg_path = s2_basisraster__dir / "s2_basisraster__20m__RD__gpkg.gpkg"
s2_basisraster__10m__WGS84__gpkg_path = s2_basisraster__dir / "s2_basisraster__10m__WGS84__gpkg.gpkg"
s2_basisraster__20m__WGS84__gpkg_path = s2_basisraster__dir / "s2_basisraster__20m__WGS84__gpkg.gpkg"


# +============================================================================+
# |                           n02_stitch_sat_obs_rstrs                         |
# +============================================================================+
# --- sentinel mosaics raw --- #
s1_mosaics__raw__dir = data_dir / "00_remote_sensed_raw/s1_mosaic"
s2_mosaics__raw__dir = data_dir / "00_remote_sensed_raw/s2_mosaic"

# --- sentinel mosaics stiched --- #
s1_mosaics__stitched__dir = data_dir / "01_remote_sensed_processed/s1_mosaic/01_s1_mosaic_stitched"
s2_mosaics__stitched__dir = data_dir / "01_remote_sensed_processed/s2_mosaic/01_s2_mosaic_stitched"

# --- s2 observations to uint32 --- #
s2_mosaics__to_unt16__dir = data_dir / "01_remote_sensed_processed/s2_mosaic/02_s2_to_int16_conversion"

# --- sentinel mosaics clip --- #
s1_mosaics__n2000__clip__dir = data_dir / "01_remote_sensed_processed/s1_mosaic/02_s1_mosaic_n2000_clip"
s2_mosaics__n2000__clip__dir = data_dir / "01_remote_sensed_processed/s2_mosaic/03_s2_mosaic_n2000_clip"


# +============================================================================+
# |                          n03_build_ndvi_ndwi_rstrs                         |
# +============================================================================+
s2_ndvwi_calc_dir = data_dir / "01_remote_sensed_processed/s2_mosaic/04_ndvwi_rstrs"


# +============================================================================+
# |                          n04_s1_res_to_10m_and_Q                           |
# +============================================================================+
s1_mosaic_10m_res__dir = data_dir / "01_remote_sensed_processed/s1_mosaic/03_s1_mosaic_10m_res"
s1_Q_mosaic__dir = data_dir / "01_remote_sensed_processed/s1_mosaic/04_s1_mosaic_Q"
s1_Q_mosaic_dB__dir = data_dir / "01_remote_sensed_processed/s1_mosaic/05_s1_mosaic_Q_dB"

# +============================================================================+
# |                             n09_stack_sat_rstrs                            |
# +============================================================================+
s2_stack_b2348__dir = data_dir / "01_remote_sensed_processed/raster_stacks/b2b3b4b8_stack"
s2_stack_b28ndvwi__dir = data_dir / "01_remote_sensed_processed/raster_stacks/b28ndvwi_stack"
s2_stack_b2348s1__dir = data_dir / "01_remote_sensed_processed/raster_stacks/b2348s1_stack"
s2_stack_b28ndvwis1__dir = data_dir / "01_remote_sensed_processed/raster_stacks/b28ndvwis1_stack"



################################################################################
#                      02_hab_kart_selection_and_division                      #
################################################################################
# +============================================================================+
# |                           n01_first_data_division                          |
# +============================================================================+
# DIRs
idx__division__gelderland__df_dir = data_dir / "01_basismaps_processed/habitatkartering_veluwe/02_habitatkartering_veluwe_idx_dfs/habitat_kart_gelderland"
idx__division__gelderland_minHR__df_dir = data_dir / "01_basismaps_processed/habitatkartering_veluwe/02_habitatkartering_veluwe_idx_dfs/habitat_kart_gelderland_minHR"
idx__division__website__df_dir = data_dir / "01_basismaps_processed/habitatkartering_veluwe/02_habitatkartering_veluwe_idx_dfs/habitat_kart_website"
idx__division__website_plusHR__df_dir = data_dir / "01_basismaps_processed/habitatkartering_veluwe/02_habitatkartering_veluwe_idx_dfs/habitat_kart_website_plusHR"
# AO tmp0
idx__AO__division__gelderland__tmp0__df_path = idx__division__gelderland__df_dir / "idx__AO__division__gelderland__tmp0__df.pkl"
idx__AO__division__gelderland_minHR__tmp0__df_path = idx__division__gelderland_minHR__df_dir / "idx__AO__division__gelderland_minHR__tmp0__df.pkl"
idx__AO__division__website__tmp0__df_path = idx__division__website__df_dir / "idx__AO__division__website__tmp0__df.pkl"
idx__AO__division__website_plusHR__tmp0__df_path = idx__division__website_plusHR__df_dir / "idx__AO__division__website_plusHR__tmp0__df.pkl"
# AO tmp1
idx__AO__division__gelderland__tmp1__df_path = idx__division__gelderland__df_dir / "idx__AO__division__gelderland__tmp1__df.pkl"
idx__AO__division__gelderland_minHR__tmp1__df_path = idx__division__gelderland_minHR__df_dir / "idx__AO__division__gelderland_minHR__tmp1__df.pkl"
idx__AO__division__website__tmp1__df_path = idx__division__website__df_dir / "idx__AO__division__website__tmp1__df.pkl"
idx__AO__division__website_plusHR__tmp1__df_path = idx__division__website_plusHR__df_dir / "idx__AO__division__website_plusHR__tmp1__df.pkl"
# AO tmp2
idx__AO__division__gelderland__tmp2__df_path = idx__division__gelderland__df_dir / "idx__AO__division__gelderland__tmp2__df.pkl"
idx__AO__division__website_plusHR__tmp2__df_path = idx__division__website_plusHR__df_dir / "idx__AO__division__website_plusHR__tmp2__df.pkl"
# HB tmp1
idx__HB__division__gelderland__tmp1__df_path = idx__division__gelderland__df_dir / "idx__HB__division__gelderland__tmp1__df.pkl"
idx__HB__division__gelderland_minHR__tmp1__df_path = idx__division__gelderland_minHR__df_dir / "idx__HB__division__gelderland_minHR__tmp1__df.pkl"
idx__HB__division__website__tmp1__df_path = idx__division__website__df_dir / "idx__HB__division__website__tmp1__df.pkl"
idx__HB__division__website_plusHR__tmp1__df_path = idx__division__website_plusHR__df_dir / "idx__HB__division__website_plusHR__tmp1__df.pkl"
# HB tmp2
idx__HB__division__gelderland__tmp2__df_path = idx__division__gelderland__df_dir / "idx__HB__division__gelderland__tmp2__df.pkl"
idx__HB__division__website_plusHR__tmp2__df_path = idx__division__website_plusHR__df_dir / "idx__HB__division__website_plusHR__tmp2__df.pkl"
# FC tmp1
idx__FC__division__gelderland__tmp1__df_path = idx__division__gelderland__df_dir / "idx__FC__division__gelderland__tmp1__df.pkl"
idx__FC__division__gelderland_minHR__tmp1__df_path = idx__division__gelderland_minHR__df_dir / "idx__FC__division__gelderland_minHR__tmp1__df.pkl"
idx__FC__division__website__tmp1__df_path = idx__division__website__df_dir / "idx__FC__division__website__tmp1__df.pkl"
idx__FC__division__website_plusHR__tmp1__df_path = idx__division__website_plusHR__df_dir / "idx__FC__division__website_plusHR__tmp1__df.pkl"
# FC_p80 tmp2
idx__FC_p80__division__gelderland__tmp2__df_path = idx__division__gelderland__df_dir / "idx__FC_p80__division__gelderland__tmp2__df.pkl"
idx__FC_p80__division__website_plusHR__tmp2__df_path = idx__division__website_plusHR__df_dir / "idx__FC_p80__division__website_plusHR__tmp2__df.pkl"
# WD tmp1
idx__WD__division__gelderland__tmp1__df_path = idx__division__gelderland__df_dir / "idx__WD__division__gelderland__tmp1__df.pkl"
idx__WD__division__gelderland_minHR__tmp1__df_path = idx__division__gelderland_minHR__df_dir / "idx__WD__division__gelderland_minHR__tmp1__df.pkl"
idx__WD__division__website__tmp1__df_path = idx__division__website__df_dir / "idx__WD__division__website__tmp1__df.pkl"
idx__WD__division__website_plusHR__tmp1__df_path = idx__division__website_plusHR__df_dir / "idx__WD__division__website_plusHR__tmp1__df.pkl"
# s1 tmp1
idx__s1__division__gelderland__tmp1__df_path = idx__division__gelderland__df_dir / "idx__s1__division__gelderland__tmp1__df.pkl"
idx__s1__division__gelderland_minHR__tmp1__df_path = idx__division__gelderland_minHR__df_dir / "idx__s1__division__gelderland_minHR__tmp1__df.pkl"
idx__s1__division__website__tmp1__df_path = idx__division__website__df_dir / "idx__s1__division__website__tmp1__df.pkl"
idx__s1__division__website_plusHR__tmp1__df_path = idx__division__website_plusHR__df_dir / "idx__s1__division__website_plusHR__tmp1__df.pkl"


# +============================================================================+
# |                        n03_add_LGN_OW_to_hab_kart                          |
# +============================================================================+
# --- lgn OW RD gpkg --- #
# DIR
lgn_plusOW__n2000__clip__gdf_dir = data_dir / "01_basismaps_processed/lgn/lgn_plusOW__n2000__clip__gdfs"
# GPKGs RD
lgn2018_plusOW__n2000__clip__RD__gpkg_path = lgn_plusOW__n2000__clip__gdf_dir / "lgn2018_plusOW__n2000__clip__RD__gpkg.gpkg"
lgn2019_plusOW__n2000__clip__RD__gpkg_path = lgn_plusOW__n2000__clip__gdf_dir / "lgn2019_plusOW__n2000__clip__RD__gpkg.gpkg"
lgn2020_plusOW__n2000__clip__RD__gpkg_path = lgn_plusOW__n2000__clip__gdf_dir / "lgn2020_plusOW__n2000__clip__RD__gpkg.gpkg"
lgn2021_plusOW__n2000__clip__RD__gpkg_path = lgn_plusOW__n2000__clip__gdf_dir / "lgn2021_plusOW__n2000__clip__RD__gpkg.gpkg"
lgn2022_plusOW__n2000__clip__RD__gpkg_path = lgn_plusOW__n2000__clip__gdf_dir / "lgn2022_plusOW__n2000__clip__RD__gpkg.gpkg"
lgn2023_plusOW__n2000__clip__RD__gpkg_path = lgn_plusOW__n2000__clip__gdf_dir / "lgn2023_plusOW__n2000__clip__RD__gpkg.gpkg"
lgn2024_plusOW__n2000__clip__RD__gpkg_path = lgn_plusOW__n2000__clip__gdf_dir / "lgn2024_plusOW__n2000__clip__RD__gpkg.gpkg"

# --- lgn OW processed RD gpkgs --- #
# DIRs
lgn_plusOW__n2000__processed__gelderland__gpkg_dir = data_dir / "01_basismaps_processed/lgn/lgn_plusOW__n2000__processed__gdfs/gelderland"
lgn_plusOW__n2000__processed__gelderland_minHR__gpkg_dir = data_dir / "01_basismaps_processed/lgn/lgn_plusOW__n2000__processed__gdfs/gelderland_minHR"
lgn_plusOW__n2000__processed__website__gpkg_dir = data_dir / "01_basismaps_processed/lgn/lgn_plusOW__n2000__processed__gdfs/website"
lgn_plusOW__n2000__processed__website_plusHR__gpkg_dir = data_dir / "01_basismaps_processed/lgn/lgn_plusOW__n2000__processed__gdfs/website_plusHR"
# Gelderland
lgn2018_plusOW__n2000__processed__gelderland__RD__gpkg_path = lgn_plusOW__n2000__processed__gelderland__gpkg_dir / "lgn2018_plusOW__n2000__processed__gelderland__RD__gpkg.gpkg"
lgn2019_plusOW__n2000__processed__gelderland__RD__gpkg_path = lgn_plusOW__n2000__processed__gelderland__gpkg_dir / "lgn2019_plusOW__n2000__processed__gelderland__RD__gpkg.gpkg"
lgn2020_plusOW__n2000__processed__gelderland__RD__gpkg_path = lgn_plusOW__n2000__processed__gelderland__gpkg_dir / "lgn2020_plusOW__n2000__processed__gelderland__RD__gpkg.gpkg"
lgn2021_plusOW__n2000__processed__gelderland__RD__gpkg_path = lgn_plusOW__n2000__processed__gelderland__gpkg_dir / "lgn2021_plusOW__n2000__processed__gelderland__RD__gpkg.gpkg"
lgn2022_plusOW__n2000__processed__gelderland__RD__gpkg_path = lgn_plusOW__n2000__processed__gelderland__gpkg_dir / "lgn2022_plusOW__n2000__processed__gelderland__RD__gpkg.gpkg"
lgn2023_plusOW__n2000__processed__gelderland__RD__gpkg_path = lgn_plusOW__n2000__processed__gelderland__gpkg_dir / "lgn2023_plusOW__n2000__processed__gelderland__RD__gpkg.gpkg"
lgn2024_plusOW__n2000__processed__gelderland__RD__gpkg_path = lgn_plusOW__n2000__processed__gelderland__gpkg_dir / "lgn2024_plusOW__n2000__processed__gelderland__RD__gpkg.gpkg"
# Gelderland_minHR
lgn2018_plusOW__n2000__processed__gelderland_minHR__RD__gpkg_path = lgn_plusOW__n2000__processed__gelderland_minHR__gpkg_dir / "lgn2018_plusOW__n2000__processed__gelderland_minHR__RD__gpkg.gpkg"
lgn2019_plusOW__n2000__processed__gelderland_minHR__RD__gpkg_path = lgn_plusOW__n2000__processed__gelderland_minHR__gpkg_dir / "lgn2019_plusOW__n2000__processed__gelderland_minHR__RD__gpkg.gpkg"
lgn2020_plusOW__n2000__processed__gelderland_minHR__RD__gpkg_path = lgn_plusOW__n2000__processed__gelderland_minHR__gpkg_dir / "lgn2020_plusOW__n2000__processed__gelderland_minHR__RD__gpkg.gpkg"
lgn2021_plusOW__n2000__processed__gelderland_minHR__RD__gpkg_path = lgn_plusOW__n2000__processed__gelderland_minHR__gpkg_dir / "lgn2021_plusOW__n2000__processed__gelderland_minHR__RD__gpkg.gpkg"
lgn2022_plusOW__n2000__processed__gelderland_minHR__RD__gpkg_path = lgn_plusOW__n2000__processed__gelderland_minHR__gpkg_dir / "lgn2022_plusOW__n2000__processed__gelderland_minHR__RD__gpkg.gpkg"
lgn2023_plusOW__n2000__processed__gelderland_minHR__RD__gpkg_path = lgn_plusOW__n2000__processed__gelderland_minHR__gpkg_dir / "lgn2023_plusOW__n2000__processed__gelderland_minHR__RD__gpkg.gpkg"
lgn2024_plusOW__n2000__processed__gelderland_minHR__RD__gpkg_path = lgn_plusOW__n2000__processed__gelderland_minHR__gpkg_dir / "lgn2024_plusOW__n2000__processed__gelderland_minHR__RD__gpkg.gpkg"
# website
lgn2018_plusOW__n2000__processed__website__RD__gpkg_path = lgn_plusOW__n2000__processed__website__gpkg_dir / "lgn2018_plusOW__n2000__processed__website__RD__gpkg.gpkg"
lgn2019_plusOW__n2000__processed__website__RD__gpkg_path = lgn_plusOW__n2000__processed__website__gpkg_dir / "lgn2019_plusOW__n2000__processed__website__RD__gpkg.gpkg"
lgn2020_plusOW__n2000__processed__website__RD__gpkg_path = lgn_plusOW__n2000__processed__website__gpkg_dir / "lgn2020_plusOW__n2000__processed__website__RD__gpkg.gpkg"
lgn2021_plusOW__n2000__processed__website__RD__gpkg_path = lgn_plusOW__n2000__processed__website__gpkg_dir / "lgn2021_plusOW__n2000__processed__website__RD__gpkg.gpkg"
lgn2022_plusOW__n2000__processed__website__RD__gpkg_path = lgn_plusOW__n2000__processed__website__gpkg_dir / "lgn2022_plusOW__n2000__processed__website__RD__gpkg.gpkg"
lgn2023_plusOW__n2000__processed__website__RD__gpkg_path = lgn_plusOW__n2000__processed__website__gpkg_dir / "lgn2023_plusOW__n2000__processed__website__RD__gpkg.gpkg"
lgn2024_plusOW__n2000__processed__website__RD__gpkg_path = lgn_plusOW__n2000__processed__website__gpkg_dir / "lgn2024_plusOW__n2000__processed__website__RD__gpkg.gpkg"
# Website_plusHR
lgn2018_plusOW__n2000__processed__website_plusHR__RD__gpkg_path = lgn_plusOW__n2000__processed__website_plusHR__gpkg_dir / "lgn2018_plusOW__n2000__processed__website_plusHR__RD__gpkg.gpkg"
lgn2019_plusOW__n2000__processed__website_plusHR__RD__gpkg_path = lgn_plusOW__n2000__processed__website_plusHR__gpkg_dir / "lgn2019_plusOW__n2000__processed__website_plusHR__RD__gpkg.gpkg"
lgn2020_plusOW__n2000__processed__website_plusHR__RD__gpkg_path = lgn_plusOW__n2000__processed__website_plusHR__gpkg_dir / "lgn2020_plusOW__n2000__processed__website_plusHR__RD__gpkg.gpkg"
lgn2021_plusOW__n2000__processed__website_plusHR__RD__gpkg_path = lgn_plusOW__n2000__processed__website_plusHR__gpkg_dir / "lgn2021_plusOW__n2000__processed__website_plusHR__RD__gpkg.gpkg"
lgn2022_plusOW__n2000__processed__website_plusHR__RD__gpkg_path = lgn_plusOW__n2000__processed__website_plusHR__gpkg_dir / "lgn2022_plusOW__n2000__processed__website_plusHR__RD__gpkg.gpkg"
lgn2023_plusOW__n2000__processed__website_plusHR__RD__gpkg_path = lgn_plusOW__n2000__processed__website_plusHR__gpkg_dir / "lgn2023_plusOW__n2000__processed__website_plusHR__RD__gpkg.gpkg"
lgn2024_plusOW__n2000__processed__website_plusHR__RD__gpkg_path = lgn_plusOW__n2000__processed__website_plusHR__gpkg_dir / "lgn2024_plusOW__n2000__processed__website_plusHR__RD__gpkg.gpkg"

# --- lgn OW stacked GPKG --- #
# DIR
lgn_plusOW__n2000__stacked__gpkg_dir = data_dir / "01_basismaps_processed/lgn/lgn_plusOW__n2000__stacked__gdfs"
# GPKGs RD
lgn_plusOW__n2000__stacked__gelderland__RD__gpkg_path = lgn_plusOW__n2000__stacked__gpkg_dir / "lgn_plusOW__n2000__stacked__gelderland__RD__gpkg.gpkg"
lgn_plusOW__n2000__stacked__gelderland_minHR__RD__gpkg_path = lgn_plusOW__n2000__stacked__gpkg_dir / "lgn_plusOW__n2000__stacked__gelderland_minHR__RD__gpkg.gpkg"
lgn_plusOW__n2000__stacked__website__RD__gpkg_path = lgn_plusOW__n2000__stacked__gpkg_dir / "lgn_plusOW__n2000__stacked__website__RD__gpkg.gpkg"
lgn_plusOW__n2000__stacked__website_plusHR__RD__gpkg_path = lgn_plusOW__n2000__stacked__gpkg_dir / "lgn_plusOW__n2000__stacked__website_plusHR__RD__gpkg.gpkg"

# --- hab kart plusOw merged GPKG --- #
habitat_kart_plusOW__merged__gelderland__RD__gpkg_path = habitat_kart__processed__gelderland__gpkg_dir / "habitat_kart_plusOW__merged__gelderland__RD__gpkg.gpkg"
habitat_kart_plusOW__merged__gelderland_minHR__RD__gpkg_path = habitat_kart__processed__gelderland_minHR__gpkg_dir / "habitat_kart_plusOW__merged__gelderland_minHR__RD__gpkg.gpkg"
habitat_kart_plusOW__merged__website__RD__gpkg_path = habitat_kart__processed__website__gpkg_dir / "habitat_kart_plusOW__merged__website__RD__gpkg.gpkg"
habitat_kart_plusOW__merged__website_plusHR__RD__gpkg_path = habitat_kart__processed__website_plusHR__gpkg_dir / "habitat_kart_plusOW__merged__website_plusHR__RD__gpkg.gpkg"


# +============================================================================+
# |                     n04_visualizing_plusOW_pixel_count                     |
# +============================================================================+
# AO tmp1 plusOW
idx__AO_plusOW__division__gelderland__tmp1__df_path = idx__division__gelderland__df_dir / "idx__AO_plusOW__division__gelderland__tmp1__df.pkl"
idx__AO_plusOW__division__gelderland_minHR__tmp1__df_path = idx__division__gelderland_minHR__df_dir / "idx__AO_plusOW__division__gelderland_minHR__tmp1__df.pkl"
idx__AO_plusOW__division__website__tmp1__df_path = idx__division__website__df_dir / "idx__AO_plusOW__division__website__tmp1__df.pkl"
idx__AO_plusOW__division__website_plusHR__tmp1__df_path = idx__division__website_plusHR__df_dir / "idx__AO_plusOW__division__website_plusHR__tmp1__df.pkl"
# WD tmp1 plusOW
idx__WD_plusOW__division__gelderland__tmp1__df_path = idx__division__gelderland__df_dir / "idx__WD_plusOW__division__gelderland__tmp1__df.pkl"
idx__WD_plusOW__division__gelderland_minHR__tmp1__df_path = idx__division__gelderland_minHR__df_dir / "idx__WD_plusOW__division__gelderland_minHR__tmp1__df.pkl"
idx__WD_plusOW__division__website__tmp1__df_path = idx__division__website__df_dir / "idx__WD_plusOW__division__website__tmp1__df.pkl"
idx__WD_plusOW__division__website_plusHR__tmp1__df_path = idx__division__website_plusHR__df_dir / "idx__WD_plusOW__division__website_plusHR__tmp1__df.pkl"
# s1 plusOW tmp1
idx__s1_plusOW__division__gelderland__tmp1__df_path = idx__division__gelderland__df_dir / "idx__s1_plusOW__division__gelderland__tmp1__df.pkl"
idx__s1_plusOW__division__gelderland_minHR__tmp1__df_path = idx__division__gelderland_minHR__df_dir / "idx__s1_plusOW__division__gelderland_minHR__tmp1__df.pkl"
idx__s1_plusOW__division__website__tmp1__df_path = idx__division__website__df_dir / "idx__s1_plusOW__division__website__tmp1__df.pkl"
idx__s1_plusOW__division__website_plusHR__tmp1__df_path = idx__division__website_plusHR__df_dir / "idx__s1_plusOW__division__website_plusHR__tmp1__df.pkl"


# +============================================================================+
# |                           n05_pixelize_hab_kart                            |
# +============================================================================+
# --- FC p80 tmp2 GPKGs --- #
# DIRs
habitat_kart__polys__gelderland__gpkg_dir = data_dir / "01_basismaps_processed/habitatkartering_veluwe/03_idx_selected_polys_GPKG/Gelderland"
habitat_kart__polys__website_plusHR__gpkg_dir = data_dir / "01_basismaps_processed/habitatkartering_veluwe/03_idx_selected_polys_GPKG/Website_plusHR"
# poly GPKGs RD
habitat_kart__FC_p80_tmp2_polys__gelderland__RD__gpkg_path = habitat_kart__polys__gelderland__gpkg_dir / "habitat_kart__FC_p80_tmp2_polys__gelderland__RD__gpkg.gpkg"
habitat_kart__FC_p80_tmp2_polys__website_plusHR__RD__gpkg_path = habitat_kart__polys__website_plusHR__gpkg_dir / "habitat_kart__FC_p80_tmp2_polys__website_plusHR__RD__gpkg.gpkg"

# --- Habitat kart pixeled for inspection GPKGs --- #
# DIR
habitat_kart__pixeled__gpkg_dir = data_dir / "01_basismaps_processed/habitatkartering_veluwe/04_pixeled_for_inspection_gpkgs"
lgn_OW__pixeled__gpkg_dir = data_dir / "01_basismaps_processed/lgn/06_lgn_pixeled_for_inspection"
# pixeled GPKGs
habitat_kart__FC_p80_tmp2_inspection__gelderland__RD__gpkg_path = habitat_kart__pixeled__gpkg_dir / "habitat_kart__FC_p80_tmp2_inspection__gelderland__RD__gpkg.gpkg"
habitat_kart__FC_p80_tmp2_inspection__website_plusHR__RD__gpkg_path = habitat_kart__pixeled__gpkg_dir / "habitat_kart__FC_p80_tmp2_inspection__website_plusHR__RD__gpkg.gpkg"
OW_all_years__inspection__lgn__RD__gpkg_path = lgn_OW__pixeled__gpkg_dir / "OW_all_years__inspection__lgn__RD__gpkg.gpkg"


################################################################################
#                       03_training_validation_data_split                      #
################################################################################
# +============================================================================+
# |                     n01_build_selected_pixels_idx_dfs                      |
# +============================================================================+
# --- manual inspected pixeled gpkg --- # 
# DIR
habitat_kart__selected__gelderland__gpkg_dir = data_dir / "03_training_validation_data/00_manual_inspected_pixeled_gpkgs"
# GPKG
habitat_kart_plusOW__FC_p80_tmp2_selected__gelderland__RD__gpkg_path = habitat_kart__selected__gelderland__gpkg_dir / "habitat_kart_plusOW__FC_p80_tmp2_selected__gelderland__RD__gpkg.gpkg"

# --- selected pixels IDX dfs --- # 
# DIR
idx__selected__gelderland__df_dir = data_dir / "03_training_validation_data/01_selected_pixels_idx_dfs/habitat_kart_gelderland"
# PKL
idx__FC_plusOW_p80__selected__gelderland__tmp2__df_path = idx__selected__gelderland__df_dir / "idx__FC_plusOW_p80__selected__gelderland__tmp2__df.pkl"
idx__WD1_plusOW_p80__selected__gelderland__tmp2__df_path = idx__selected__gelderland__df_dir / "idx__WD1_plusOW_p80__selected__gelderland__tmp2__df.pkl"
idx__WD2_plusOW_p80__selected__gelderland__tmp2__df_path = idx__selected__gelderland__df_dir / "idx__WD2_plusOW_p80__selected__gelderland__tmp2__df.pkl"
idx__WD3_plusOW_p80__selected__gelderland__tmp2__df_path = idx__selected__gelderland__df_dir / "idx__WD3_plusOW_p80__selected__gelderland__tmp2__df.pkl"
idx__WD4_plusOW_p80__selected__gelderland__tmp2__df_path = idx__selected__gelderland__df_dir / "idx__WD4_plusOW_p80__selected__gelderland__tmp2__df.pkl"
idx__WD5_plusOW_p80__selected__gelderland__tmp2__df_path = idx__selected__gelderland__df_dir / "idx__WD5_plusOW_p80__selected__gelderland__tmp2__df.pkl"
idx__WD6_plusOW_p80__selected__gelderland__tmp2__df_path = idx__selected__gelderland__df_dir / "idx__WD6_plusOW_p80__selected__gelderland__tmp2__df.pkl"
idx__s1_plusOW_p80__selected__gelderland__tmp2__df_path = idx__selected__gelderland__df_dir / "idx__s1_plusOW_p80__selected__gelderland__tmp2__df.pkl"
idx__s2_plusOW_p80__selected__gelderland__tmp2__df_path = idx__selected__gelderland__df_dir / "idx__s2_plusOW_p80__selected__gelderland__tmp2__df.pkl"
idx__s3_plusOW_p80__selected__gelderland__tmp2__df_path = idx__selected__gelderland__df_dir / "idx__s3_plusOW_p80__selected__gelderland__tmp2__df.pkl"
idx__WDNF1_plusOW_p80__selected__gelderland__tmp2__df_path = idx__selected__gelderland__df_dir / "idx__WDNF1_plusOW_p80__selected__gelderland__tmp2__df.pkl"
idx__WDNF2_plusOW_p80__selected__gelderland__tmp2__df_path = idx__selected__gelderland__df_dir / "idx__WDNF2_plusOW_p80__selected__gelderland__tmp2__df.pkl"
idx__WDNF3_plusOW_p80__selected__gelderland__tmp2__df_path = idx__selected__gelderland__df_dir / "idx__WDNF3_plusOW_p80__selected__gelderland__tmp2__df.pkl"
idx__WDNF4_plusOW_p80__selected__gelderland__tmp2__df_path = idx__selected__gelderland__df_dir / "idx__WDNF4_plusOW_p80__selected__gelderland__tmp2__df.pkl"
idx__WDNF5_plusOW_p80__selected__gelderland__tmp2__df_path = idx__selected__gelderland__df_dir / "idx__WDNF5_plusOW_p80__selected__gelderland__tmp2__df.pkl"
idx__WDNF6_plusOW_p80__selected__gelderland__tmp2__df_path = idx__selected__gelderland__df_dir / "idx__WDNF6_plusOW_p80__selected__gelderland__tmp2__df.pkl"


# +============================================================================+
# |                    n03_training_validation_split_carto                     |
# +============================================================================+
# --- idx training validation dfs --- #
# DIR
training_validation__dir = data_dir / "03_training_validation_data"
idx__pixeled_training_validation_carto__dir = data_dir / "03_training_validation_data/02_training_validation_idx_dfs/Carto"
# WD1
idx__WD1_plusOW_p80_tmp2_training_cart_at1__gelderland__df_path = idx__pixeled_training_validation_carto__dir / "idx__WD1_plusOW_p80_tmp2_training_cart_at1__gelderland__df.pkl"
idx__WD1_plusOW_p80_tmp2_validation_cart_at1__gelderland__df_path = idx__pixeled_training_validation_carto__dir / "idx__WD1_plusOW_p80_tmp2_validation_cart_at1__gelderland__df.pkl"
# WD2
idx__WD2_plusOW_p80_tmp2_training_cart_at1__gelderland__df_path = idx__pixeled_training_validation_carto__dir / "idx__WD2_plusOW_p80_tmp2_training_cart_at1__gelderland__df.pkl"
idx__WD2_plusOW_p80_tmp2_validation_cart_at1__gelderland__df_path = idx__pixeled_training_validation_carto__dir / "idx__WD2_plusOW_p80_tmp2_validation_cart_at1__gelderland__df.pkl"
# WD3
idx__WD3_plusOW_p80_tmp2_training_cart_at1__gelderland__df_path = idx__pixeled_training_validation_carto__dir / "idx__WD3_plusOW_p80_tmp2_training_at1_cart__gelderland__df.pkl"
idx__WD3_plusOW_p80_tmp2_validation_cart_at1__gelderland__df_path = idx__pixeled_training_validation_carto__dir / "idx__WD3_plusOW_p80_tmp2_validation_cart_at1__gelderland__df.pkl"
# WD4
idx__WD4_plusOW_p80_tmp2_training_cart_at1__gelderland__df_path = idx__pixeled_training_validation_carto__dir / "idx__WD4_plusOW_p80_tmp2_training_cart_at1__gelderland__df.pkl"
idx__WD4_plusOW_p80_tmp2_validation_cart_at1__gelderland__df_path = idx__pixeled_training_validation_carto__dir / "idx__WD4_plusOW_p80_tmp2_validation_cart_at1__gelderland__df.pkl"
# WD5
idx__WD5_plusOW_p80_tmp2_training_cart_at1__gelderland__df_path = idx__pixeled_training_validation_carto__dir / "idx__WD5_plusOW_p80_tmp2_training_cart_at1__gelderland__df.pkl"
idx__WD5_plusOW_p80_tmp2_validation_cart_at1__gelderland__df_path = idx__pixeled_training_validation_carto__dir / "idx__WD5_plusOW_p80_tmp2_validation_cart_at1__gelderland__df.pkl"
# WD6
idx__WD6_plusOW_p80_tmp2_training_cart_at1__gelderland__df_path = idx__pixeled_training_validation_carto__dir / "idx__WD6_plusOW_p80_tmp2_training_cart_at1__gelderland__df.pkl"
idx__WD6_plusOW_p80_tmp2_validation_cart_at1__gelderland__df_path = idx__pixeled_training_validation_carto__dir / "idx__WD6_plusOW_p80_tmp2_validation_cart_at1__gelderland__df.pkl"
# S1
idx__s1_plusOW_p80_tmp2_training_cart_at1__gelderland__df_path = idx__pixeled_training_validation_carto__dir / "idx__s1_plusOW_p80_tmp2_training_cart_at1__gelderland__df.pkl"
idx__s1_plusOW_p80_tmp2_validation_cart_at1__gelderland__df_path = idx__pixeled_training_validation_carto__dir / "idx__s1_plusOW_p80_tmp2_validation_cart_at1__gelderland__df.pkl"
# S2
idx__s2_plusOW_p80_tmp2_training_cart_at1__gelderland__df_path = idx__pixeled_training_validation_carto__dir / "idx__s2_plusOW_p80_tmp2_training_cart_at1__gelderland__df.pkl"
idx__s2_plusOW_p80_tmp2_validation_cart_at1__gelderland__df_path = idx__pixeled_training_validation_carto__dir / "idx__s2_plusOW_p80_tmp2_validation_cart_at1__gelderland__df.pkl"
# WDNF1
idx__WDNF1_plusOW_p80_tmp2_training_cart_at1__gelderland__df_path = idx__pixeled_training_validation_carto__dir / "idx__WDNF1_plusOW_p80_tmp2_training_cart_at1__gelderland__df.pkl"
idx__WDNF1_plusOW_p80_tmp2_validation_cart_at1__gelderland__df_path = idx__pixeled_training_validation_carto__dir / "idx__WDNF1_plusOW_p80_tmp2_validation_cart_at1__gelderland__df.pkl"
# WDNF2
idx__WDNF2_plusOW_p80_tmp2_training_cart_at1__gelderland__df_path = idx__pixeled_training_validation_carto__dir / "idx__WDNF2_plusOW_p80_tmp2_training_cart_at1__gelderland__df.pkl"
idx__WDNF2_plusOW_p80_tmp2_validation_cart_at1__gelderland__df_path = idx__pixeled_training_validation_carto__dir / "idx__WDNF2_plusOW_p80_tmp2_validation_cart_at1__gelderland__df.pkl"
# WDNF3
idx__WDNF3_plusOW_p80_tmp2_training_cart_at1__gelderland__df_path = idx__pixeled_training_validation_carto__dir / "idx__WDNF3_plusOW_p80_tmp2_training_at1_cart__gelderland__df.pkl"
idx__WDNF3_plusOW_p80_tmp2_validation_cart_at1__gelderland__df_path = idx__pixeled_training_validation_carto__dir / "idx__WDNF3_plusOW_p80_tmp2_validation_cart_at1__gelderland__df.pkl"
# WDNF4
idx__WDNF4_plusOW_p80_tmp2_training_cart_at1__gelderland__df_path = idx__pixeled_training_validation_carto__dir / "idx__WDNF4_plusOW_p80_tmp2_training_cart_at1__gelderland__df.pkl"
idx__WDNF4_plusOW_p80_tmp2_validation_cart_at1__gelderland__df_path = idx__pixeled_training_validation_carto__dir / "idx__WDNF4_plusOW_p80_tmp2_validation_cart_at1__gelderland__df.pkl"
# WDNF5
idx__WDNF5_plusOW_p80_tmp2_training_cart_at1__gelderland__df_path = idx__pixeled_training_validation_carto__dir / "idx__WDNF5_plusOW_p80_tmp2_training_cart_at1__gelderland__df.pkl"
idx__WDNF5_plusOW_p80_tmp2_validation_cart_at1__gelderland__df_path = idx__pixeled_training_validation_carto__dir / "idx__WDNF5_plusOW_p80_tmp2_validation_cart_at1__gelderland__df.pkl"
# WDNF6
idx__WDNF6_plusOW_p80_tmp2_training_cart_at1__gelderland__df_path = idx__pixeled_training_validation_carto__dir / "idx__WDNF6_plusOW_p80_tmp2_training_cart_at1__gelderland__df.pkl"
idx__WDNF6_plusOW_p80_tmp2_validation_cart_at1__gelderland__df_path = idx__pixeled_training_validation_carto__dir / "idx__WDNF6_plusOW_p80_tmp2_validation_cart_at1__gelderland__df.pkl"

# --- training validation GPKGs Carto [RD] --- #
# DIRs
training_cart__RD__gpkgs_dir = data_dir / "03_training_validation_data/03_training_data_gpkgs/Carto [RD]"
validation_cart__RD__gpkgs_dir = data_dir / "03_training_validation_data/03_validation_data_gpkgs/Carto [RD]"
# WD1
training_polys__WD1_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg_path = training_cart__RD__gpkgs_dir / "training_polys__WD1_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg.gpkg"
validation_pixels__WD1_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg_path = validation_cart__RD__gpkgs_dir / "validation_pixels__WD1_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg.gpkg"
# WD2
training_polys__WD2_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg_path = training_cart__RD__gpkgs_dir / "training_polys__WD2_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg.gpkg"
validation_pixels__WD2_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg_path = validation_cart__RD__gpkgs_dir / "validation_pixels__WD2_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg.gpkg"
# WD3
training_polys__WD3_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg_path = training_cart__RD__gpkgs_dir / "training_polys__WD3_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg.gpkg"
validation_pixels__WD3_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg_path = validation_cart__RD__gpkgs_dir / "validation_pixels__WD3_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg.gpkg"
# WD4
training_polys__WD4_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg_path = training_cart__RD__gpkgs_dir / "training_polys__WD4_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg.gpkg"
validation_pixels__WD4_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg_path = validation_cart__RD__gpkgs_dir / "validation_pixels__WD4_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg.gpkg"
# WD5
training_polys__WD5_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg_path = training_cart__RD__gpkgs_dir / "training_polys__WD5_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg.gpkg"
validation_pixels__WD5_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg_path = validation_cart__RD__gpkgs_dir / "validation_pixels__WD5_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg.gpkg"
# WD6
training_polys__WD6_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg_path = training_cart__RD__gpkgs_dir / "training_polys__WD6_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg.gpkg"
validation_pixels__WD6_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg_path = validation_cart__RD__gpkgs_dir / "validation_pixels__WD6_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg.gpkg"
# S1
training_polys__s1_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg_path = training_cart__RD__gpkgs_dir / "training_polys__s1_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg.gpkg"
validation_pixels__s1_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg_path = validation_cart__RD__gpkgs_dir / "validation_pixels__s1_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg.gpkg"
# S2
training_polys__s2_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg_path = training_cart__RD__gpkgs_dir / "training_polys__s2_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg.gpkg"
validation_pixels__s2_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg_path = validation_cart__RD__gpkgs_dir / "validation_pixels__s2_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg.gpkg"
# WDNF1
training_polys__WDNF1_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg_path = training_cart__RD__gpkgs_dir / "training_polys__WDNF1_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg.gpkg"
validation_pixels__WDNF1_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg_path = validation_cart__RD__gpkgs_dir / "validation_pixels__WDNF1_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg.gpkg"
# WDNF2
training_polys__WDNF2_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg_path = training_cart__RD__gpkgs_dir / "training_polys__WDNF2_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg.gpkg"
validation_pixels__WDNF2_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg_path = validation_cart__RD__gpkgs_dir / "validation_pixels__WDNF2_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg.gpkg"
# WDNF3
training_polys__WDNF3_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg_path = training_cart__RD__gpkgs_dir / "training_polys__WDNF3_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg.gpkg"
validation_pixels__WDNF3_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg_path = validation_cart__RD__gpkgs_dir / "validation_pixels__WDNF3_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg.gpkg"
# WDNF4
training_polys__WDNF4_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg_path = training_cart__RD__gpkgs_dir / "training_polys__WDNF4_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg.gpkg"
validation_pixels__WDNF4_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg_path = validation_cart__RD__gpkgs_dir / "validation_pixels__WDNF4_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg.gpkg"
# WDNF5
training_polys__WDNF5_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg_path = training_cart__RD__gpkgs_dir / "training_polys__WDNF5_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg.gpkg"
validation_pixels__WDNF5_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg_path = validation_cart__RD__gpkgs_dir / "validation_pixels__WDNF5_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg.gpkg"
# WDNF6
training_polys__WDNF6_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg_path = training_cart__RD__gpkgs_dir / "training_polys__WDNF6_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg.gpkg"
validation_pixels__WDNF6_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg_path = validation_cart__RD__gpkgs_dir / "validation_pixels__WDNF6_plusOW_p80_tmp2_cart_at1__gelderland__RD__gpkg.gpkg"


# +============================================================================+
# |                     n03_training_validation_split_ML                       |
# +============================================================================+
# --- idx training validation dfs --- #
# DIR
idx__pixeled_training_validation_ML__dir = data_dir / "03_training_validation_data/02_training_validation_idx_dfs/ML"
#WD1
idx__WD1_plusOW_p80_tmp2_training_ML_at1__gelderland__df_path = idx__pixeled_training_validation_ML__dir / "idx__WD1_plusOW_p80_tmp2_training_ML_at1__gelderland__df.pkl"
idx__WD1_plusOW_p80_tmp2_validation_ML_at1__gelderland__df_path = idx__pixeled_training_validation_ML__dir / "idx__WD1_plusOW_p80_tmp2_validation_ML_at1__gelderland__df.pkl"
#WD2
idx__WD2_plusOW_p80_tmp2_training_ML_at1__gelderland__df_path = idx__pixeled_training_validation_ML__dir / "idx__WD2_plusOW_p80_tmp2_training_ML_at1__gelderland__df.pkl"
idx__WD2_plusOW_p80_tmp2_validation_ML_at1__gelderland__df_path = idx__pixeled_training_validation_ML__dir / "idx__WD2_plusOW_p80_tmp2_validation_ML_at1__gelderland__df.pkl"
#WD3
idx__WD3_plusOW_p80_tmp2_training_ML_at1__gelderland__df_path = idx__pixeled_training_validation_ML__dir / "idx__WD3_plusOW_p80_tmp2_training_ML_at1__gelderland__df.pkl"
idx__WD3_plusOW_p80_tmp2_validation_ML_at1__gelderland__df_path = idx__pixeled_training_validation_ML__dir / "idx__WD3_plusOW_p80_tmp2_validation_ML_at1__gelderland__df.pkl"
#WD4
idx__WD4_plusOW_p80_tmp2_training_ML_at1__gelderland__df_path = idx__pixeled_training_validation_ML__dir / "idx__WD4_plusOW_p80_tmp2_training_ML_at1__gelderland__df.pkl"
idx__WD4_plusOW_p80_tmp2_validation_ML_at1__gelderland__df_path = idx__pixeled_training_validation_ML__dir / "idx__WD4_plusOW_p80_tmp2_validation_ML_at1__gelderland__df.pkl"
#WD5
idx__WD5_plusOW_p80_tmp2_training_ML_at1__gelderland__df_path = idx__pixeled_training_validation_ML__dir / "idx__WD5_plusOW_p80_tmp2_training_ML_at1__gelderland__df.pkl"
idx__WD5_plusOW_p80_tmp2_validation_ML_at1__gelderland__df_path = idx__pixeled_training_validation_ML__dir / "idx__WD5_plusOW_p80_tmp2_validation_ML_at1__gelderland__df.pkl"
#WD6
idx__WD6_plusOW_p80_tmp2_training_ML_at1__gelderland__df_path = idx__pixeled_training_validation_ML__dir / "idx__WD6_plusOW_p80_tmp2_training_ML_at1__gelderland__df.pkl"
idx__WD6_plusOW_p80_tmp2_validation_ML_at1__gelderland__df_path = idx__pixeled_training_validation_ML__dir / "idx__WD6_plusOW_p80_tmp2_validation_ML_at1__gelderland__df.pkl"
#s1
idx__s1_plusOW_p80_tmp2_training_ML_at1__gelderland__df_path = idx__pixeled_training_validation_ML__dir / "idx__WD5_plusOW_p80_tmp2_training_ML_at1__gelderland__df.pkl"
idx__s1_plusOW_p80_tmp2_validation_ML_at1__gelderland__df_path = idx__pixeled_training_validation_ML__dir / "idx__WD5_plusOW_p80_tmp2_validation_ML_at1__gelderland__df.pkl"
#s2
idx__s2_plusOW_p80_tmp2_training_ML_at1__gelderland__df_path = idx__pixeled_training_validation_ML__dir / "idx__WD5_plusOW_p80_tmp2_training_ML_at1__gelderland__df.pkl"
idx__s2_plusOW_p80_tmp2_validation_ML_at1__gelderland__df_path = idx__pixeled_training_validation_ML__dir / "idx__WD5_plusOW_p80_tmp2_validation_ML_at1__gelderland__df.pkl"
#WDNF1
idx__WDNF1_plusOW_p80_tmp2_training_ML_at1__gelderland__df_path = idx__pixeled_training_validation_ML__dir / "idx__WDNF1_plusOW_p80_tmp2_training_ML_at1__gelderland__df.pkl"
idx__WDNF1_plusOW_p80_tmp2_validation_ML_at1__gelderland__df_path = idx__pixeled_training_validation_ML__dir / "idx__WDNF1_plusOW_p80_tmp2_validation_ML_at1__gelderland__df.pkl"
#WDNF2
idx__WDNF2_plusOW_p80_tmp2_training_ML_at1__gelderland__df_path = idx__pixeled_training_validation_ML__dir / "idx__WDNF2_plusOW_p80_tmp2_training_ML_at1__gelderland__df.pkl"
idx__WDNF2_plusOW_p80_tmp2_validation_ML_at1__gelderland__df_path = idx__pixeled_training_validation_ML__dir / "idx__WDNF2_plusOW_p80_tmp2_validation_ML_at1__gelderland__df.pkl"
#WDNF3
idx__WDNF3_plusOW_p80_tmp2_training_ML_at1__gelderland__df_path = idx__pixeled_training_validation_ML__dir / "idx__WDNF3_plusOW_p80_tmp2_training_ML_at1__gelderland__df.pkl"
idx__WDNF3_plusOW_p80_tmp2_validation_ML_at1__gelderland__df_path = idx__pixeled_training_validation_ML__dir / "idx__WDNF3_plusOW_p80_tmp2_validation_ML_at1__gelderland__df.pkl"
#WDNF4
idx__WDNF4_plusOW_p80_tmp2_training_ML_at1__gelderland__df_path = idx__pixeled_training_validation_ML__dir / "idx__WDNF4_plusOW_p80_tmp2_training_ML_at1__gelderland__df.pkl"
idx__WDNF4_plusOW_p80_tmp2_validation_ML_at1__gelderland__df_path = idx__pixeled_training_validation_ML__dir / "idx__WDNF4_plusOW_p80_tmp2_validation_ML_at1__gelderland__df.pkl"
#WDNF5
idx__WDNF5_plusOW_p80_tmp2_training_ML_at1__gelderland__df_path = idx__pixeled_training_validation_ML__dir / "idx__WDNF5_plusOW_p80_tmp2_training_ML_at1__gelderland__df.pkl"
idx__WDNF5_plusOW_p80_tmp2_validation_ML_at1__gelderland__df_path = idx__pixeled_training_validation_ML__dir / "idx__WDNF5_plusOW_p80_tmp2_validation_ML_at1__gelderland__df.pkl"
#WDNF6
idx__WDNF6_plusOW_p80_tmp2_training_ML_at1__gelderland__df_path = idx__pixeled_training_validation_ML__dir / "idx__WDNF6_plusOW_p80_tmp2_training_ML_at1__gelderland__df.pkl"
idx__WDNF6_plusOW_p80_tmp2_validation_ML_at1__gelderland__df_path = idx__pixeled_training_validation_ML__dir / "idx__WDNF6_plusOW_p80_tmp2_validation_ML_at1__gelderland__df.pkl"

# --- training validation GPKGs ML-approach [UTM32631] --- #
# DIRs
training_ML__UTM32631__gpkgs_dir = data_dir / "03_training_validation_data/03_training_data_gpkgs/ML_approach [UTM32631]"
validation_ML__UTM32631__gpkgs_dir = data_dir / "03_training_validation_data/03_validation_data_gpkgs/ML_approach [UTM32631]"
# WD1
training_pixels__WD1_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg_path = training_ML__UTM32631__gpkgs_dir / "training_pixels__WD1_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg.gpkg"
validation_pixels__WD1_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg_path = validation_ML__UTM32631__gpkgs_dir / "validation_pixels__WD1_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg.gpkg"
# WD2
training_pixels__WD2_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg_path = training_ML__UTM32631__gpkgs_dir / "training_pixels__WD2_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg.gpkg"
validation_pixels__WD2_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg_path = validation_ML__UTM32631__gpkgs_dir / "validation_pixels__WD2_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg.gpkg"
# WD3
training_pixels__WD3_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg_path = training_ML__UTM32631__gpkgs_dir / "training_pixels__WD3_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg.gpkg"
validation_pixels__WD3_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg_path = validation_ML__UTM32631__gpkgs_dir / "validation_pixels__WD3_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg.gpkg"
# WD4
training_pixels__WD4_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg_path = training_ML__UTM32631__gpkgs_dir / "training_pixels__WD4_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg.gpkg"
validation_pixels__WD4_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg_path = validation_ML__UTM32631__gpkgs_dir / "validation_pixels__WD4_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg.gpkg"
# WD5
training_pixels__WD5_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg_path = training_ML__UTM32631__gpkgs_dir / "training_pixels__WD5_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg.gpkg"
validation_pixels__WD5_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg_path = validation_ML__UTM32631__gpkgs_dir / "validation_pixels__WD5_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg.gpkg"
# WD6
training_pixels__WD6_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg_path = training_ML__UTM32631__gpkgs_dir / "training_pixels__WD6_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg.gpkg"
validation_pixels__WD6_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg_path = validation_ML__UTM32631__gpkgs_dir / "validation_pixels__WD6_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg.gpkg"
# S1
training_pixels__s1_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg_path = training_ML__UTM32631__gpkgs_dir / "training_pixels__s1_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg.gpkg"
validation_pixels__s1_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg_path = validation_ML__UTM32631__gpkgs_dir / "validation_pixels__s1_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg.gpkg"
# S2
training_pixels__s2_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg_path = training_ML__UTM32631__gpkgs_dir / "training_pixels__s2_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg.gpkg"
validation_pixels__s2_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg_path = validation_ML__UTM32631__gpkgs_dir / "validation_pixels__s2_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg.gpkg"
# S3
training_pixels__s3_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg_path = training_ML__UTM32631__gpkgs_dir / "training_pixels__s3_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg.gpkg"
validation_pixels__s3_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg_path = validation_ML__UTM32631__gpkgs_dir / "validation_pixels__s3_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg.gpkg"
# WDNF1
training_pixels__WDNF1_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg_path = training_ML__UTM32631__gpkgs_dir / "training_pixels__WDNF1_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg.gpkg"
validation_pixels__WDNF1_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg_path = validation_ML__UTM32631__gpkgs_dir / "validation_pixels__WDNF1_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg.gpkg"
# WDNF2
training_pixels__WDNF2_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg_path = training_ML__UTM32631__gpkgs_dir / "training_pixels__WDNF2_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg.gpkg"
validation_pixels__WDNF2_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg_path = validation_ML__UTM32631__gpkgs_dir / "validation_pixels__WDNF2_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg.gpkg"
# WDNF3
training_pixels__WDNF3_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg_path = training_ML__UTM32631__gpkgs_dir / "training_pixels__WDNF3_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg.gpkg"
validation_pixels__WDNF3_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg_path = validation_ML__UTM32631__gpkgs_dir / "validation_pixels__WDNF3_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg.gpkg"
# WDNF4
training_pixels__WDNF4_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg_path = training_ML__UTM32631__gpkgs_dir / "training_pixels__WDNF4_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg.gpkg"
validation_pixels__WDNF4_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg_path = validation_ML__UTM32631__gpkgs_dir / "validation_pixels__WDNF4_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg.gpkg"
# WDNF5
training_pixels__WDNF5_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg_path = training_ML__UTM32631__gpkgs_dir / "training_pixels__WDNF5_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg.gpkg"
validation_pixels__WDNF5_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg_path = validation_ML__UTM32631__gpkgs_dir / "validation_pixels__WDNF5_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg.gpkg"
# WDNF6
training_pixels__WDNF6_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg_path = training_ML__UTM32631__gpkgs_dir / "training_pixels__WDNF6_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg.gpkg"
validation_pixels__WDNF6_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg_path = validation_ML__UTM32631__gpkgs_dir / "validation_pixels__WDNF6_plusOW_p80_tmp2_ML_at1__gelderland__UTM32631__gpkg.gpkg"



################################################################################
#                             04_ML_classification                             #
################################################################################
# +============================================================================+
# |                            n01_RF_s2_quarterly                             |
# +============================================================================+
# --- s2 2017 quarterly input --- #
# s2 stacks
s2_stack_b2348__veluwe__2017_q1__UTM32631__rstr_path = s2_stack_b2348__dir / "2017_Q1/S2_b2348_stack.tif"
s2_stack_b2348__veluwe__2017_q2__UTM32631__rstr_path = s2_stack_b2348__dir / "2017_Q2/S2_b2348_stack.tif"
s2_stack_b2348__veluwe__2017_q3__UTM32631__rstr_path = s2_stack_b2348__dir / "2017_Q3/S2_b2348_stack.tif"
s2_stack_b2348__veluwe__2017_q4__UTM32631__rstr_path = s2_stack_b2348__dir / "2017_Q4/S2_b2348_stack.tif"

# --- RF model output --- #
RF_out__UTM32631__rstrs_dir = data_dir / "04_RF_output"


################################################################################
#                             05_output_validation                             #
################################################################################
# +============================================================================+
# |                        n01_checking_carto_results                          |
# +============================================================================+
# DIRs
carto_output_main__dir = data_dir / "04_carto_output"
carto_output_stack__dir = carto_output_main__dir / "02_stacked_rstrs"


# +============================================================================+
# |                                n02_heatmaps                                |
# +============================================================================+
# DIRs
carto_output_result_dfs__dir = data_dir / "04_carto_output/03_classification_result_stats_dfs"
carto_heatmaps__dir = data_dir / "05_performance_evaluation/01_carto_heatmaps"
RF_heatmaps__dir = data_dir / "05_performance_evaluation/01_RF_heatmaps"


# +============================================================================+
# |                        n03_building_entropy_rstrs                          |
# +============================================================================+
# DIR
pixel_stability_rstrs__dir = data_dir / "05_performance_evaluation/02_entropy_rasters"


# +============================================================================+
# |                            n04_result_plotting                             |
# +============================================================================+
result_plotting__dir = data_dir / "05_performance_evaluation/03_result_plotting"


# +============================================================================+
# |                            n05_area_over_time                             |
# +============================================================================+
area_over_time__dir = data_dir / "05_performance_evaluation/04_area_over_time"