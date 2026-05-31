###############################################################################
## STACK RASTERS TREE - MULTI-LOCATION VERSION (with auto-alignment)
###############################################################################
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


def stack_rasters_tree_multi(
    in_roots: List[str | Path],
    out_root: str | Path,
    filenames: List[str],
    *,
    out_name: str = "stack.tif",
    overwrite: bool = False,
    skip_if_up_to_date: bool = False,
    tol: float = 0.0,
    auto_align: bool = False,
    resampling_method: str = "bilinear",
    include_single_mask_band: bool = True,
    mask_band_index: Optional[int] = None,
    mask_out_as_alpha: bool = False,
    require_same_dtype: bool = False,
    dst_dtype: Optional[str] = None,
    nodata: Optional[float | int] = None,
    blocksize: int = 512,
    compress: str = "deflate",
) -> List[Path]:
    """
    Stack rasters from multiple directory trees with matching folder structures.
    
    Takes rasters from corresponding folders across multiple input roots and
    stacks them into multi-band GeoTIFFs. All input trees must have the same
    folder structure. For each subfolder found in the first input root, the
    function looks for the same relative path in all other input roots.
    
    Parameters
    ----------
    in_roots : List[str | Path]
        List of root directories to search for input rasters. All must have
        the same folder structure.
    out_root : str | Path
        Root directory where stacked outputs will be written, preserving the
        folder structure from the first input root.
    filenames : List[str]
        List of filenames to stack from each folder. For each folder, the
        function will look for these exact filenames (or the first matching
        if it contains wildcards like '*').
        Examples: ["VV_Q_mean.tif", "VH_Q_mean.tif", "ratio_Q_mean.tif"]
    out_name : str, default="stack.tif"
        Filename for each output stacked raster.
    overwrite : bool, default=False
        If True, overwrite existing output files. If False, skip existing files.
    skip_if_up_to_date : bool, default=False
        If True, skip processing if output exists and is newer than all inputs.
    tol : float, default=0.0
        Tolerance for alignment checks. If 0.0, requires exact match of CRS,
        transform, shape, resolution, and bounds. If > 0.0, allows numeric
        differences within `tol` for transform, resolution, and bounds.
    auto_align : bool, default=False
        If True, automatically reproject/resample all inputs to match the first
        raster's grid. If False, raises ValueError on alignment mismatch.
        Useful when stacking data from different sensors (e.g., S1 + S2).
    resampling_method : str, default="bilinear"
        Resampling method when auto_align=True. Options: "nearest", "bilinear",
        "cubic", "cubic_spline", "lanczos", "average", "mode".
        Ignored when auto_align=False.
    include_single_mask_band : bool, default=True
        If True, include a single shared mask band (from the first multi-band
        raster) as the last band in the output.
    mask_band_index : int | None, default=None
        Index (1-based) of the mask band in multi-band inputs. If None, assumes
        band 2 for 2-band rasters. Ignored for single-band rasters.
    mask_out_as_alpha : bool, default=False
        If True, set the mask band's color interpretation to alpha (requires
        `include_single_mask_band=True`).
    require_same_dtype : bool, default=False
        If True, raises an error if input data bands have different dtypes.
    dst_dtype : str | None, default=None
        Output data type (e.g., 'uint8', 'float32'). If None, uses the dtype
        of the first data band. If `require_same_dtype=True`, validates all
        inputs share this dtype.
    nodata : float | int | None, default=None
        NoData value for the output. If None, uses the nodata value from the
        first input raster.
    blocksize : int, default=512
        Tile block size for the output GeoTIFF (blockxsize and blockysize).
    compress : str, default="deflate"
        Compression method for the output GeoTIFF (e.g., 'deflate', 'lzw').
    
    Returns
    -------
    List[Path]
        List of paths to all created (or existing) stacked raster files.
    
    Raises
    ------
    FileNotFoundError
        If any of the input roots do not exist.
    ValueError
        If no matching rasters are found, if alignment checks fail (when
        auto_align=False), if dtypes differ when `require_same_dtype=True`,
        or if folder structures don't match.
    
    Examples
    --------
    # Basic usage (old way - strict alignment required)
    >>> outputs = stack_rasters_tree_multi(
    ...     in_roots=["data/s1", "data/s1", "data/s1"],
    ...     out_root="data/s1_stacked",
    ...     filenames=["VV_Q_mean.tif", "VH_Q_mean.tif", "ratio_Q_mean.tif"],
    ...     out_name="VV_VH_ratio_stack.tif"
    ... )
    
    # With auto-alignment for multi-sensor stacking (S2 + S1)
    >>> outputs = stack_rasters_tree_multi(
    ...     in_roots=["data/s2", "data/s2", "data/s1", "data/s1"],
    ...     out_root="data/combined_stack",
    ...     filenames=["B2.tif", "B3.tif", "VV_mean.tif", "VH_mean.tif"],
    ...     out_name="S2_S1_stack.tif",
    ...     auto_align=True,
    ...     resampling_method="bilinear"
    ... )
    """
    
    # Resampling method mapping
    resampling_map = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "cubic_spline": Resampling.cubic_spline,
        "lanczos": Resampling.lanczos,
        "average": Resampling.average,
        "mode": Resampling.mode,
    }
    
    if resampling_method not in resampling_map:
        raise ValueError(
            f"Invalid resampling_method: {resampling_method}. "
            f"Choose from: {list(resampling_map.keys())}"
        )
    
    resampling_enum = resampling_map[resampling_method]
    
    # Convert to Path objects
    in_roots = [Path(p) for p in in_roots]
    out_root = Path(out_root)
    
    # Validate inputs
    if not in_roots:
        raise ValueError("in_roots cannot be empty")
    
    if not filenames:
        raise ValueError("filenames cannot be empty")
    
    if len(in_roots) != len(filenames):
        raise ValueError(
            f"Number of in_roots ({len(in_roots)}) must match number of filenames ({len(filenames)})"
        )
    
    # Check all roots exist
    for root in in_roots:
        if not root.exists():
            raise FileNotFoundError(f"in_root not found: {root}")
    
    out_root.mkdir(parents=True, exist_ok=True)
    
    # Get all subfolders from the first root (reference structure)
    ref_root = in_roots[0]
    ref_folders = sorted([p for p in ref_root.rglob("*") if p.is_dir()])
    
    # Include the root itself if it directly contains files
    ref_folders.insert(0, ref_root)
    
    # Remove duplicates and sort
    ref_folders = sorted(set(ref_folders))
    
    def aligns_all(paths: Sequence[Path], tol: float) -> bool:
        """Check that all rasters align (CRS/transform/shape/res/bounds)."""
        if len(paths) < 2:
            return True
        
        def get(ds):
            return (ds.crs, ds.transform, ds.width, ds.height, ds.res, ds.bounds)
        
        with rasterio.open(paths[0]) as ref:
            ref_info = get(ref)
        
        for p in paths[1:]:
            with rasterio.open(p) as ds:
                info = get(ds)
                if tol == 0.0:
                    if info != ref_info:
                        return False
                else:
                    if info[0] != ref_info[0]:
                        return False
                    if (info[2], info[3]) != (ref_info[2], ref_info[3]):
                        return False
                    if not np.allclose(np.array(info[1]), np.array(ref_info[1]), atol=tol, rtol=0):
                        return False
                    if not np.allclose(np.array(info[4], float), np.array(ref_info[4], float), atol=tol, rtol=0):
                        return False
                    if not np.allclose(np.array(info[5], float), np.array(ref_info[5], float), atol=tol, rtol=0):
                        return False
        return True
    
    def _rasters_aligned(src, ref_crs, ref_transform, ref_wh, tol):
        """Check if source raster matches reference grid."""
        if src.crs != ref_crs:
            return False
        if (src.width, src.height) != ref_wh:
            return False
        if tol == 0.0:
            return src.transform == ref_transform
        else:
            return np.allclose(
                np.array(src.transform), 
                np.array(ref_transform), 
                atol=tol, 
                rtol=0
            )
    
    outputs: List[Path] = []
    
    # Process each folder
    for ref_folder in ref_folders:
        # Get relative path from reference root
        try:
            rel_path = ref_folder.relative_to(ref_root)
        except ValueError:
            continue
        
        label = rel_path.as_posix() if rel_path.as_posix() != "." else ref_root.name
        
        # Collect matching files from all roots
        raster_paths: List[Path] = []
        
        for i, (root, filename) in enumerate(zip(in_roots, filenames)):
            # Construct the corresponding folder in this root
            current_folder = root / rel_path
            
            if not current_folder.exists():
                # Skip this folder if it doesn't exist in this root
                # (allows for sparse structures)
                continue
            
            # Look for the file
            if "*" in filename or "?" in filename:
                # Handle wildcards
                matches = list(current_folder.glob(filename))
                if matches:
                    raster_paths.append(matches[0])  # Take first match
            else:
                # Exact filename
                file_path = current_folder / filename
                if file_path.exists():
                    raster_paths.append(file_path)
        
        # Skip if no files found in this folder
        if not raster_paths:
            continue
        
        # Skip if we don't have all expected files
        if len(raster_paths) != len(filenames):
            print(f"\nWARNING: Skipping {label} - found {len(raster_paths)}/{len(filenames)} files")
            continue
        
        print(f"\nProcessing: {label} ({len(raster_paths)} rasters)")
        
        # Create output directory
        out_dir = out_root / rel_path
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / out_name
        
        # Check if we should skip
        if skip_if_up_to_date and out_path.exists() and not overwrite:
            newest_in = max(p.stat().st_mtime for p in raster_paths)
            if out_path.stat().st_mtime >= newest_in:
                print(f"  - stack: exists (up-to-date) -> {out_path}")
                outputs.append(out_path)
                continue
        
        if out_path.exists() and not overwrite:
            print(f"  - stack: exists -> {out_path}")
            outputs.append(out_path)
            continue
        
        # Alignment check (only strict if auto_align=False)
        if not auto_align:
            if not aligns_all(raster_paths, tol=tol):
                raise ValueError(
                    f"Alignment check failed for group '{label}'.\n"
                    f"Consider setting auto_align=True to reproject misaligned rasters.\n"
                    f"Files:\n- " + "\n- ".join(str(p) for p in raster_paths)
                )
        
        # Reference profile/template (from FIRST raster)
        with rasterio.open(raster_paths[0]) as ref:
            ref_profile = ref.profile.copy()
            ref_crs = ref.crs
            ref_transform = ref.transform
            ref_wh = (ref.width, ref.height)
        
        data_bands: List[tuple[Path, int]] = []
        mask_source: Optional[Path] = None
        mask_src_band: Optional[int] = None
        
        # Determine which band is data and which is mask per input
        for p in raster_paths:
            with rasterio.open(p) as ds:
                if ds.count == 1:
                    data_bands.append((p, 1))
                else:
                    mb = mask_band_index
                    if mb is None:
                        mb = 2 if ds.count == 2 else None
                    
                    if mb is None or mb < 1 or mb > ds.count:
                        raise ValueError(
                            f"Cannot determine mask band for {p} (count={ds.count}). "
                            f"Set mask_band_index explicitly."
                        )
                    
                    data_band = 1 if mb != 1 else 2
                    data_bands.append((p, data_band))
                    
                    if include_single_mask_band and mask_source is None:
                        mask_source = p
                        mask_src_band = mb
        
        if include_single_mask_band and mask_source is None:
            include_single_mask_band = False
        
        # dtype handling
        if require_same_dtype and dst_dtype is None:
            dtypes = set()
            for p, bidx in data_bands:
                with rasterio.open(p) as ds:
                    dtypes.add(ds.dtypes[bidx - 1])
            if len(dtypes) != 1:
                raise ValueError(f"Input data dtypes differ in group '{label}': {sorted(dtypes)}")
            dst_dtype = next(iter(dtypes))
        
        if dst_dtype is None:
            with rasterio.open(data_bands[0][0]) as ds:
                dst_dtype = ds.dtypes[data_bands[0][1] - 1]
        
        if nodata is None:
            with rasterio.open(data_bands[0][0]) as ds:
                nodata = ds.nodata
        
        out_count = len(data_bands) + (1 if include_single_mask_band else 0)
        
        profile = ref_profile
        profile.update(
            driver="GTiff",
            count=out_count,
            dtype=dst_dtype,
            nodata=nodata,
            tiled=True,
            blockxsize=blocksize,
            blockysize=blocksize,
            compress=compress,
            crs=ref_crs,
            transform=ref_transform,
            width=ref_wh[0],
            height=ref_wh[1],
        )
        
        sources_str = "|".join(str(p.resolve()) for p in raster_paths)
        
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.update_tags(
                STACK_SOURCES=sources_str,
                STACK_FILENAMES=",".join(filenames),
                STACK_MASK_INCLUDED=str(include_single_mask_band),
                STACK_AUTO_ALIGNED=str(auto_align),
            )
            
            # Write data bands (with optional reprojection)
            for out_bi, (p, bi) in enumerate(data_bands, start=1):
                with rasterio.open(p) as src:
                    
                    # Check if reprojection is needed
                    needs_reproject = auto_align and not _rasters_aligned(
                        src, ref_crs, ref_transform, ref_wh, tol
                    )
                    
                    if needs_reproject:
                        # Reproject this raster to match reference
                        arr = np.empty((ref_wh[1], ref_wh[0]), dtype=src.dtypes[bi - 1])
                        
                        reproject(
                            source=rasterio.band(src, bi),
                            destination=arr,
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=ref_transform,
                            dst_crs=ref_crs,
                            resampling=resampling_enum,
                            src_nodata=src.nodata,
                            dst_nodata=nodata,
                        )
                        
                        print(f"    └─ Reprojected {p.name}:band{bi} using {resampling_method}")
                    else:
                        # Already aligned, just read
                        arr = src.read(bi)
                    
                    # Handle nodata conversion
                    if src.nodata is not None and nodata is not None and src.nodata != nodata:
                        arr = np.where(arr == src.nodata, nodata, arr)
                    
                    # Convert dtype if needed
                    if arr.dtype != np.dtype(dst_dtype):
                        arr = arr.astype(dst_dtype, copy=False)
                    
                    dst.write(arr, out_bi)
                    
                    # Set band description
                    if src.count == 1:
                        desc = p.stem
                    elif src.count == 2 and mask_band_index in (1, 2, None):
                        mb_eff = 2 if mask_band_index is None else mask_band_index
                        data_band_eff = 1 if mb_eff != 1 else 2
                        desc = p.stem if bi == data_band_eff else f"{p.stem}:band{bi}"
                    else:
                        desc = f"{p.stem}:band{bi}"
                    
                    dst.set_band_description(out_bi, desc)
            
            # Write single mask band (from first raster that had a mask)
            if include_single_mask_band:
                assert mask_source is not None and mask_src_band is not None
                
                with rasterio.open(mask_source) as src:
                    
                    needs_reproject = auto_align and not _rasters_aligned(
                        src, ref_crs, ref_transform, ref_wh, tol
                    )
                    
                    if needs_reproject:
                        mask = np.empty((ref_wh[1], ref_wh[0]), dtype='uint8')
                        
                        reproject(
                            source=rasterio.band(src, mask_src_band),
                            destination=mask,
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=ref_transform,
                            dst_crs=ref_crs,
                            resampling=Resampling.nearest,  # Always use nearest for masks
                            src_nodata=src.nodata,
                        )
                        
                        print(f"    └─ Reprojected mask from {mask_source.name} using nearest")
                    else:
                        mask = src.read(mask_src_band)
                
                mask = (mask > 0).astype("uint8")
                
                mask_out_index = len(data_bands) + 1
                dst.write(mask.astype(dst_dtype), mask_out_index)
                dst.set_band_description(mask_out_index, "VALID_MASK")
                
                if mask_out_as_alpha:
                    try:
                        from rasterio.enums import ColorInterp
                        
                        cis = [ColorInterp.gray] * out_count
                        cis[mask_out_index - 1] = ColorInterp.alpha
                        dst.colorinterp = cis
                    except Exception:
                        pass
        
        print(f"  - stack: wrote -> {out_path}")
        outputs.append(out_path)
    
    print(f"\n{'='*60}")
    print(f"Completed: {len(outputs)} stacked images created")
    print(f"{'='*60}")
    
    return outputs