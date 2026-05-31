################################################################################
##  CLEAN INSPECTED GPKG
################################################################################
import numpy as np
import geopandas as gpd

def clean_and_update_inspected_GPKG(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Drops selected columns, recalculates Shape_Area from geometry,
    and updates bedekkingsOppervlakte1/2 using percentages.
    Rounds all computed areas to integers.
    """
    gdf = gdf.copy()

    # 1) Drop columns if they exist
    gdf = gdf.drop(columns=["month", "veldSituatieDatum", "OBJECTID"], errors="ignore")

    # 2) Recalculate Shape_Area (units depend on CRS; RD typically meters -> m²)
    gdf["Shape_Area"] = gdf.geometry.area.round(0).astype("Int64")

    # Helper: compute covered area from percentage
    def _covered_area(shape_area, pct):
        pct_num = pd.to_numeric(pct, errors="coerce")
        return (shape_area * (pct_num / 100.0)).round(0).astype("Int64")

    import pandas as pd  # local import so the function is self-contained

    # 3) Update bedekkingsOppervlakte1 and 2 (only if columns exist)
    if "bedekkingsPercentage1" in gdf.columns:
        gdf["bedekkingsOppervlakte1"] = _covered_area(gdf["Shape_Area"], gdf["bedekkingsPercentage1"])

    if "bedekkingsPercentage2" in gdf.columns:
        gdf["bedekkingsOppervlakte2"] = _covered_area(gdf["Shape_Area"], gdf["bedekkingsPercentage2"])

    return gdf