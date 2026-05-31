from __future__ import annotations

################################################################################
## QUICK RASTER PLOT
################################################################################
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from rasterio.plot import plotting_extent
import matplotlib.pyplot as plt

def quick_rstr_plot_func(
    raster_path,
    band=1,
    mask_band="auto",   # "auto" | None | int (1-based)
    cmap="viridis",
    figsize=(10, 8),
    add_colorbar=True,
):
    """
    Plot a single raster band, using an available mask (alpha/mask/nodata) to improve plotting.

    Parameters
    ----------
    raster_path : str | Path
    band : int
        Band to plot (1-based).
    mask_band : "auto" | None | int
        - "auto": try alpha band; else dataset/band mask; else nodata
        - None: no masking
        - int: explicitly use this band as mask/alpha band (0 => masked)
    """
    raster_path = Path(raster_path)

    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError(f"Raster has no CRS: {raster_path}")

        if not (1 <= band <= src.count):
            raise ValueError(f"Requested band={band}, but raster has {src.count} band(s).")

        data = src.read(band)

        # --- Determine mask (True = masked) ---
        mask = None

        if mask_band is None:
            mask = None

        elif isinstance(mask_band, int):
            if not (1 <= mask_band <= src.count):
                raise ValueError(f"mask_band={mask_band} out of range (1..{src.count}).")
            m = src.read(mask_band)
            mask = (m == 0)

        elif mask_band == "auto":
            # 1) Prefer an explicit alpha band if present
            alpha_idx = None
            try:
                cis = list(src.colorinterp)
                if ColorInterp.alpha in cis:
                    alpha_idx = cis.index(ColorInterp.alpha) + 1  # to 1-based
            except Exception:
                alpha_idx = None

            if alpha_idx is not None:
                alpha = src.read(alpha_idx)
                mask = (alpha == 0)
            else:
                # 2) Use internal rasterio mask (0 = invalid)
                m = src.read_masks(band)
                if m is not None:
                    mask = (m == 0)

                # 3) Fallback to nodata if set
                if (mask is None or not mask.any()) and (src.nodata is not None):
                    mask = (data == src.nodata)
        else:
            raise ValueError('mask_band must be "auto", None, or an int (1-based).')

        arr = np.ma.array(data, mask=mask) if mask is not None else data

        extent = plotting_extent(src)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(arr, cmap=cmap, extent=extent)
    ax.set_title(f"{raster_path.name} | band {band}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if add_colorbar:
        plt.colorbar(im, ax=ax, label="value")

    plt.tight_layout()
    plt.show()