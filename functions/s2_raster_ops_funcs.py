################################################################################
##  CUT GDF TO RASTER PIXELS
################################################################################
from __future__ import annotations

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
from rasterio.features import rasterize
from shapely.geometry import box, MultiPolygon
from shapely.validation import make_valid


def _force_multipolygon(g):
    """Return a MultiPolygon (or None) keeping only polygonal parts."""
    if g is None or g.is_empty:
        return None

    gt = g.geom_type
    if gt == "Polygon":
        return MultiPolygon([g])
    if gt == "MultiPolygon":
        return g
    if gt == "GeometryCollection":
        polys = []
        for gg in g.geoms:
            if gg.geom_type == "Polygon":
                polys.append(gg)
            elif gg.geom_type == "MultiPolygon":
                polys.extend(list(gg.geoms))
        return MultiPolygon(polys) if polys else None

    # Lines/points etc. -> nothing to keep for area polygons
    return None

def cut_gdf_to_raster_pixels(
    gdf: gpd.GeoDataFrame,
    raster_path: str,
    all_touched: bool = True,
    area_col: str = "bedekkingsOppervlakte1",
    make_geoms_valid: bool = True,
    force_multipolygon: bool = True,
    progress: bool = True,
    progress_every: int = 200,
) -> gpd.GeoDataFrame:
    """
    Per-row -> per-selected-pixel output with new index suffixes: GI_1_1, GI_1_2, ...

    - Copies all non-geometry columns to outputs
    - Recalculates `area_col` from the new geometry
    - Updates geometry and forces MultiPolygon if requested
    - all_touched=True: partial pixel intersections (clipped to raster extent)
      all_touched=False: only fully covered pixels (can extend beyond raster extent)
    """
    if gdf.empty:
        return gdf.copy()

    gdf_in = gdf.copy()

    if make_geoms_valid:
        gdf_in["geometry"] = gdf_in.geometry.apply(lambda geom: make_valid(geom) if geom is not None else geom)

    iterator = gdf_in.iterrows()
    use_tqdm = False
    if progress:
        try:
            from tqdm.auto import tqdm  # type: ignore

            iterator = tqdm(iterator, total=len(gdf_in), desc="Clipping features", unit="feat")
            use_tqdm = True
        except Exception:
            use_tqdm = False

    rows_out = []

    with rasterio.open(raster_path) as ds:
        if gdf_in.crs is None:
            raise ValueError("Input GeoDataFrame has no CRS set.")
        if ds.crs is None:
            raise ValueError("Raster has no CRS set.")
        if gdf_in.crs.to_string() != ds.crs.to_string():
            raise ValueError(f"CRS mismatch: gdf={gdf_in.crs}, raster={ds.crs}")

        raster_bounds_poly = box(ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top)

        for i, (orig_idx, row) in enumerate(iterator, start=1):
            if progress and (not use_tqdm) and (i % progress_every == 0):
                print(f"[{i}/{len(gdf_in)}] processed; output rows so far: {len(rows_out)}")

            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            if all_touched:
                geom2 = geom.intersection(raster_bounds_poly)
                if geom2.is_empty:
                    continue
            else:
                geom2 = geom

            minx, miny, maxx, maxy = geom2.bounds
            w = from_bounds(minx, miny, maxx, maxy, transform=ds.transform)
            w = w.round_offsets().round_lengths()
            if w.width <= 0 or w.height <= 0:
                continue

            window_transform = ds.window_transform(w)
            out_shape = (int(w.height), int(w.width))

            mask = rasterize(
                [(geom2, 1)],
                out_shape=out_shape,
                transform=window_transform,
                fill=0,
                dtype="uint8",
                all_touched=all_touched,
            )

            hit_rc = np.argwhere(mask == 1)
            if hit_rc.size == 0:
                continue

            counter = 0
            for r, c in hit_rc:
                x1, y1 = window_transform * (int(c), int(r))
                x2, y2 = window_transform * (int(c) + 1, int(r) + 1)
                cell = box(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

                if all_touched:
                    part = geom2.intersection(cell)
                else:
                    if not geom2.covers(cell):
                        continue
                    part = cell

                if part.is_empty:
                    continue

                if force_multipolygon:
                    part = _force_multipolygon(part)
                    if part is None or part.is_empty:
                        continue

                counter += 1
                new_id = f"{str(orig_idx)}_{counter}"

                out_row = row.drop(labels=["geometry"]).to_dict()
                out_row["geometry"] = part
                if area_col in out_row:
                    out_row[area_col] = float(part.area)  # assumes projected CRS (m²)
                else:
                    out_row[area_col] = float(part.area)

                out_row["new_index"] = new_id
                rows_out.append(out_row)

    out = gpd.GeoDataFrame(rows_out, crs=gdf_in.crs)
    if not out.empty:
        out = out.set_index("new_index", drop=True)
    return out