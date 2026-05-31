###############################################################################
## FIND S1 FILES
###############################################################################
from pathlib import Path
from typing import List, Tuple, Callable
import rasterio


def process_tree(
    in_root: str | Path,
    *,
    pattern: str = "*.tif",
    recursive: bool = True,
    transform_fn: Callable[[Path, Path], bool],
    out_root: str | Path | None = None,
    preserve_structure: bool = True,
) -> Tuple[List[Path], List[Path]]:
    """
    Find files in a directory tree and apply a transformation function to each.

    Parameters
    ----------
    in_root : path
        Root folder to search
    pattern : str
        Glob pattern for files (e.g. "*.tif")
    recursive : bool
        If True, use rglob (recursive), else glob
    transform_fn : callable
        Function with signature: (in_path: Path, out_path: Path) -> bool
        Should return True if processed, False if skipped
    out_root : path, optional
        Output root folder (will mirror input structure).
        If None, out_path will be same as in_path (in-place processing)
    preserve_structure : bool
        Whether to preserve directory structure in output (default: True)
        Only used when out_root is not None

    Returns
    -------
    processed : List[Path]
        List of input paths that were processed
    skipped : List[Path]
        List of input paths that were skipped
    """
    in_root = Path(in_root)
    
    if not in_root.exists():
        raise FileNotFoundError(f"in_root not found: {in_root}")
    
    if out_root is not None:
        out_root = Path(out_root)
        out_root.mkdir(parents=True, exist_ok=True)

    # Find files
    paths = (in_root.rglob(pattern) if recursive else in_root.glob(pattern))
    paths = sorted(p for p in paths if p.is_file())

    if not paths:
        print(f"WARNING: No files matching '{pattern}' found in {in_root}")
        return [], []
    
    print(f"\nFound {len(paths)} files matching '{pattern}'")

    processed: List[Path] = []
    skipped: List[Path] = []

    for in_path in paths:
        # Determine output path
        if out_root is not None:
            if preserve_structure:
                # Preserve directory structure (default behavior)
                rel = in_path.relative_to(in_root)
                out_path = out_root / rel
            else:
                # Flat structure - all files in out_root directly
                out_path = out_root / in_path.name
            
            out_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            # In-place processing
            out_path = in_path

        # Apply transformation
        was_processed = transform_fn(in_path, out_path)
        
        if was_processed:
            processed.append(in_path)
        else:
            skipped.append(in_path)

    print(f"\n{'='*60}")
    print(f"Completed: {len(processed)} processed, {len(skipped)} skipped")
    print(f"{'='*60}")

    return processed, skipped



###############################################################################
## RESAMPLE TO 10M RESOLUTION
###############################################################################
from pathlib import Path
from typing import Callable
import rasterio
from rasterio.enums import Resampling


def resample_raster_transformer(
    target_resolution: float = 10.0,
    resampling_method: str = "bilinear",
    compress: str = "deflate",
    tiled: bool = True,
    blocksize: int = 512,
    skip_if_finer: bool = True,
) -> Callable[[Path, Path], bool]:
    """
    Returns a transformation function that resamples rasters to a target resolution.

    Parameters
    ----------
    target_resolution : float
        Target pixel size in meters (e.g., 10.0 for 10x10m)
    resampling_method : str
        Resampling algorithm. Options:
        - "nearest": Nearest neighbor (preserves values, blocky)
        - "bilinear": Bilinear interpolation (smooth, good for SAR)
        - "cubic": Cubic convolution (smoother)
        - "cubic_spline": Cubic spline interpolation
        - "lanczos": Lanczos windowed sinc (high quality)
        - "average": Average of all contributing pixels
        - "mode": Mode of all contributing pixels
    compress : str
        Compression method for output (default: "deflate")
    tiled : bool
        Write tiled output (default: True)
    blocksize : int
        Block/tile size for output (default: 512)
    skip_if_finer : bool
        Skip files that are already at or finer than target resolution

    Returns
    -------
    callable
        A function that can be passed to process_tree
    """
    # Map string names to rasterio Resampling enum
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
            f"Invalid resampling_method '{resampling_method}'. "
            f"Choose from: {list(resampling_map.keys())}"
        )
    
    resampling_enum = resampling_map[resampling_method]

    def _resample(in_path: Path, out_path: Path) -> bool:
        """Returns True if resampled, False if skipped."""
        try:
            with rasterio.open(in_path) as src:
                # Check if resampling is needed
                current_res = src.res[0]
                if skip_if_finer and current_res <= target_resolution:
                    print(
                        f"  - {in_path.name}: skipped "
                        f"(already at {current_res}m <= {target_resolution}m)"
                    )
                    return False

                # Calculate scaling factor and new dimensions
                scale_factor = current_res / target_resolution
                new_height = int(src.height * scale_factor)
                new_width = int(src.width * scale_factor)

                # Read and resample all bands
                data = src.read(
                    out_shape=(src.count, new_height, new_width),
                    resampling=resampling_enum
                )

                # Update the transform for the new resolution
                transform = src.transform * src.transform.scale(
                    (src.width / new_width),
                    (src.height / new_height)
                )

                # Prepare output metadata
                profile = src.profile.copy()
                profile.update({
                    'height': new_height,
                    'width': new_width,
                    'transform': transform,
                    'compress': compress,
                    'tiled': tiled,
                })
                
                if tiled:
                    profile['blockxsize'] = blocksize
                    profile['blockysize'] = blocksize

                # Modify output filename to include resolution
                stem = in_path.stem  # e.g., "VV_stitched"
                out_name = f"{stem}_{int(target_resolution)}m.tif"
                out_path_with_res = out_path.parent / out_name

                # Write the resampled data
                with rasterio.open(out_path_with_res, 'w', **profile) as dst:
                    dst.write(data)

                print(
                    f"  - {in_path.name}: resampled {current_res}m -> {target_resolution}m "
                    f"({src.width}x{src.height} -> {new_width}x{new_height}) -> {out_name}"
                )
                return True

        except Exception as e:
            print(f"  - {in_path.name}: ERROR - {e}")
            return False

    return _resample


def resample_stitched_images_tree(
    in_root: str | Path,
    out_root: str | Path,
    target_resolution: float = 10.0,
    resampling_method: str = "bilinear",
    band_pattern: str = "*_stitched.tif",
    compress: str = "deflate",
    tiled: bool = True,
    blocksize: int = 512,
):
    """
    Walks subfolders of in_root, finds stitched rasters matching band_pattern,
    and resamples them from 20m to 10m resolution.
    Writes output to out_root/<subfolder>/<original_name>_10m.tif

    Parameters
    ----------
    in_root : path
        Root folder containing subfolders with stitched images
    out_root : path
        Output root folder (will be created)
    target_resolution : float
        Target pixel size in meters (default: 10.0 for 10x10m)
    resampling_method : str
        Resampling algorithm (default: "bilinear", good for SAR data)
    band_pattern : str
        Glob pattern to match input files (default: "*_stitched.tif")
    compress : str
        Compression method for output (default: "deflate")
    tiled : bool
        Write tiled output (default: True)
    blocksize : int
        Block/tile size for output (default: 512)
    
    Returns
    -------
    processed : List[Path]
        List of files that were resampled
    skipped : List[Path]
        List of files that were skipped
    """
    transform = resample_raster_transformer(
        target_resolution=target_resolution,
        resampling_method=resampling_method,
        compress=compress,
        tiled=tiled,
        blocksize=blocksize,
        skip_if_finer=True,
    )
    
    processed, skipped = process_tree(
        in_root=in_root,
        out_root=out_root,
        pattern=band_pattern,
        recursive=True,
        transform_fn=transform,
    )
    
    print(f"\nCompleted: {len(processed)} resampled, {len(skipped)} skipped")
    return processed, skipped


###############################################################################
## VV/VH RATIO CALCULATION
###############################################################################
from pathlib import Path
from typing import Tuple, List
import numpy as np
import rasterio
from rasterio.enums import Resampling


def calculate_vv_vh_ratio_tree(
    in_root: str | Path,
    out_root: str | Path,
    vv_pattern: str = "VV*.tif",
    vh_pattern: str = "VH*.tif",
    output_suffix: str = "_ratio",
    compress: str = "deflate",
    tiled: bool = True,
    blocksize: int = 512,
    nodata: float | None = None,
) -> Tuple[List[Path], List[Path]]:
    """
    Walks subfolders of in_root, finds VV and VH rasters, and calculates VV/VH ratio.
    Writes output to out_root/<subfolder>/VV_VH_ratio.tif

    Parameters
    ----------
    in_root : path
        Root folder containing subfolders with VV and VH images
    out_root : path
        Output root folder (will be created)
    vv_pattern : str
        Glob pattern to match VV files (default: "VV*.tif")
    vh_pattern : str
        Glob pattern to match VH files (default: "VH*.tif")
    output_suffix : str
        Suffix for output filename (default: "_ratio")
    compress : str
        Compression method for output (default: "deflate")
    tiled : bool
        Write tiled output (default: True)
    blocksize : int
        Block/tile size for output (default: 512)
    nodata : float, optional
        NoData value for output. If None, uses source nodata or -9999

    Returns
    -------
    processed : List[Path]
        List of folders where ratio was calculated
    skipped : List[Path]
        List of folders that were skipped (missing VV or VH)
    """
    in_root = Path(in_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Find all subfolders (timeframe folders)
    subfolders = [p for p in in_root.iterdir() if p.is_dir()]
    if not subfolders:
        raise ValueError(f"No subfolders found in {in_root}")

    processed: List[Path] = []
    skipped: List[Path] = []

    for sub in sorted(subfolders):
        # Find VV and VH files
        vv_files = sorted(sub.glob(vv_pattern))
        vh_files = sorted(sub.glob(vh_pattern))

        if not vv_files:
            print(f"\nWARNING {sub.name}: skipped (no VV file matching '{vv_pattern}')")
            skipped.append(sub)
            continue

        if not vh_files:
            print(f"\nWARNING {sub.name}: skipped (no VH file matching '{vh_pattern}')")
            skipped.append(sub)
            continue

        # Take first match if multiple files found
        vv_path = vv_files[0]
        vh_path = vh_files[0]

        if len(vv_files) > 1:
            print(f"\nWARNING {sub.name}: multiple VV files found, using {vv_path.name}")
        if len(vh_files) > 1:
            print(f"\nWARNING {sub.name}: multiple VH files found, using {vh_path.name}")

        # Create output directory
        out_sub = out_root / sub.name
        out_sub.mkdir(parents=True, exist_ok=True)

        # Output filename
        out_name = f"VV_VH{output_suffix}.tif"
        out_path = out_sub / out_name

        try:
            # Read VV and VH
            with rasterio.open(vv_path) as vv_src, rasterio.open(vh_path) as vh_src:
                # Verify dimensions match
                if vv_src.shape != vh_src.shape:
                    print(
                        f"\nERROR {sub.name}: dimension mismatch "
                        f"(VV: {vv_src.shape}, VH: {vh_src.shape})"
                    )
                    skipped.append(sub)
                    continue

                # Verify CRS match
                if vv_src.crs != vh_src.crs:
                    print(
                        f"\nWARNING {sub.name}: CRS mismatch "
                        f"(VV: {vv_src.crs}, VH: {vh_src.crs})"
                    )

                # Read data
                vv_data = vv_src.read(1).astype(np.float32)
                vh_data = vh_src.read(1).astype(np.float32)

                # Get nodata values
                vv_nodata = vv_src.nodata
                vh_nodata = vh_src.nodata
                output_nodata = nodata if nodata is not None else (vv_nodata if vv_nodata is not None else -9999)

                # Create mask for valid data
                valid_mask = np.ones_like(vv_data, dtype=bool)
                
                if vv_nodata is not None:
                    valid_mask &= (vv_data != vv_nodata)
                if vh_nodata is not None:
                    valid_mask &= (vh_data != vh_nodata)
                
                # Mask where VH is zero or very small to avoid division by zero
                valid_mask &= (np.abs(vh_data) > 1e-10)

                # Calculate ratio
                ratio = np.full_like(vv_data, output_nodata, dtype=np.float32)
                ratio[valid_mask] = vv_data[valid_mask] / vh_data[valid_mask]

                # Prepare output metadata
                profile = vv_src.profile.copy()
                profile.update({
                    'dtype': 'float32',
                    'nodata': output_nodata,
                    'compress': compress,
                    'tiled': tiled,
                })

                if tiled:
                    profile['blockxsize'] = blocksize
                    profile['blockysize'] = blocksize

                # Write output
                with rasterio.open(out_path, 'w', **profile) as dst:
                    dst.write(ratio, 1)

                # Calculate statistics
                valid_ratio = ratio[valid_mask]
                if len(valid_ratio) > 0:
                    ratio_stats = f"min={valid_ratio.min():.2f}, max={valid_ratio.max():.2f}, mean={valid_ratio.mean():.2f}"
                else:
                    ratio_stats = "no valid pixels"

                print(f"\nSUCCESS {sub.name}: {vv_path.name} / {vh_path.name} -> {out_name} ({ratio_stats})")
                processed.append(sub)

        except Exception as e:
            print(f"\nERROR {sub.name}: {e}")
            skipped.append(sub)
            continue

    print(f"\n{'='*60}")
    print(f"Completed: {len(processed)} ratios calculated, {len(skipped)} skipped")
    print(f"{'='*60}")
    
    return processed, skipped



###############################################################################
## CALCULATE THE STD
###############################################################################
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
from datetime import datetime
import numpy as np
import rasterio


def calculate_quarterly_std_tree(
    in_root: str | Path,
    out_root: str | Path,
    band_pattern: str = "VV*.tif",
    output_prefix: str = "VV_Q",
    date_format: str = "%Y_%m",
    compress: str = "deflate",
    tiled: bool = True,
    blocksize: int = 512,
    nodata: float | None = None,
) -> Dict[Tuple[int, int], Path]:
    """
    Calculate quarterly standard deviation from monthly mosaics.
    
    Groups monthly folders by quarter (based on folder name dates) and calculates
    the pixel-wise standard deviation across all months in each quarter.
    
    Writes output to out_root/<year>_Q<quarter>/<prefix>_std.tif
    
    Parameters
    ----------
    in_root : path
        Root folder containing monthly subfolders (named by date, e.g., "2020_01")
    out_root : path
        Output root folder (will be created with quarterly structure)
    band_pattern : str
        Glob pattern to match files in each monthly subfolder
        Examples: "VV*.tif", "VH*.tif", "*ratio*.tif"
    output_prefix : str
        Prefix for output filename (e.g., "VV_Q", "VH_Q", "ratio_Q")
    date_format : str
        Format of monthly folder names (default: "%Y_%m")
        Examples: "%Y_%m" for "2020_01", "%Y-%m" for "2020-01"
    compress : str
        Compression method for output (default: "deflate")
    tiled : bool
        Write tiled output (default: True)
    blocksize : int
        Block/tile size for output (default: 512)
    nodata : float, optional
        NoData value for output. If None, uses source nodata or -9999
    
    Returns
    -------
    output_paths : dict
        Dictionary mapping (year, quarter) to output Path
    
    Examples
    --------
    # Calculate quarterly std for VV
    >>> vv_std = calculate_quarterly_std_tree(
    ...     in_root=s1_mosaic_10m_res__dir,
    ...     out_root=s1_Q_mosaic__dir,
    ...     band_pattern="VV.tif",
    ...     output_prefix="VV_Q",
    ...     date_format="%Y_%m"
    ... )
    
    # Calculate quarterly std for VH
    >>> vh_std = calculate_quarterly_std_tree(
    ...     in_root=s1_mosaic_10m_res__dir,
    ...     out_root=s1_Q_mosaic__dir,
    ...     band_pattern="VH.tif",
    ...     output_prefix="VH_Q",
    ...     date_format="%Y_%m"
    ... )
    
    # Calculate quarterly std for ratio
    >>> ratio_std = calculate_quarterly_std_tree(
    ...     in_root=s1_mosaic_10m_res__dir,
    ...     out_root=s1_Q_mosaic__dir,
    ...     band_pattern="VV_VH_ratio.tif",
    ...     output_prefix="VV_VH_ratio_Q",
    ...     date_format="%Y_%m"
    ... )
    """
    in_root = Path(in_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    
    # Find all monthly subfolders
    monthly_folders = sorted([p for p in in_root.iterdir() if p.is_dir()])
    
    if not monthly_folders:
        raise ValueError(f"No subfolders found in {in_root}")
    
    # Group folders by quarter
    quarterly_groups: Dict[Tuple[int, int], List[Path]] = defaultdict(list)
    skipped_folders = []
    
    for month_folder in monthly_folders:
        try:
            year, quarter = get_quarter_from_date(month_folder.name, date_format)
            
            # Check if the file exists in this folder
            matching_files = list(month_folder.glob(band_pattern))
            if matching_files:
                quarterly_groups[(year, quarter)].append(month_folder)
            else:
                print(f"WARNING {month_folder.name}: no file matching '{band_pattern}'")
                skipped_folders.append(month_folder)
        except ValueError as e:
            print(f"WARNING {month_folder.name}: skipped - {e}")
            skipped_folders.append(month_folder)
            continue
    
    if not quarterly_groups:
        raise ValueError(f"No valid quarterly groups found in {in_root}")
    
    print(f"\nFound {len(quarterly_groups)} quarters with data:")
    for (year, quarter), folders in sorted(quarterly_groups.items()):
        print(f"  {year} Q{quarter}: {len(folders)} months")
    
    output_paths = {}
    
    # Process each quarter
    for (year, quarter), folders in sorted(quarterly_groups.items()):
        print(f"\nProcessing {year} Q{quarter} ({len(folders)} months)...")
        
        # Collect all rasters for this quarter
        raster_paths = []
        for folder in folders:
            matching_files = list(folder.glob(band_pattern))
            if matching_files:
                raster_paths.append(matching_files[0])  # Take first match
        
        if not raster_paths:
            print(f"  ERROR: No valid rasters found")
            continue
        
        # Need at least 2 images to calculate std
        if len(raster_paths) < 2:
            print(f"  WARNING: Only {len(raster_paths)} image(s) found, need at least 2 for std")
            continue
        
        try:
            # Read metadata from first raster
            with rasterio.open(raster_paths[0]) as src:
                profile = src.profile.copy()
                shape = src.shape
                src_nodata = src.nodata
            
            output_nodata = nodata if nodata is not None else (src_nodata if src_nodata is not None else -9999)
            
            # Initialize list to collect all data
            all_data = []
            valid_files = []
            
            # Read all rasters
            for raster_path in raster_paths:
                with rasterio.open(raster_path) as src:
                    # Verify dimensions match
                    if src.shape != shape:
                        print(f"  WARNING {raster_path.parent.name}/{raster_path.name}: dimension mismatch, skipping")
                        continue
                    
                    data = src.read(1).astype(np.float32)
                    
                    # Mask nodata
                    if src.nodata is not None:
                        data = np.where(data == src.nodata, np.nan, data)
                    
                    all_data.append(data)
                    valid_files.append(raster_path)
            
            if len(all_data) < 2:
                print(f"  ERROR: Less than 2 valid images after filtering")
                continue
            
            # Stack all images
            stacked = np.stack(all_data, axis=0)  # Shape: (n_months, height, width)
            
            # Calculate standard deviation (ignoring NaN)
            with np.errstate(invalid='ignore'):  # Suppress warnings for all-NaN slices
                result = np.nanstd(stacked, axis=0, ddof=1)  # ddof=1 for sample std
            
            # Set nodata where all values were NaN
            result = np.where(np.isnan(result), output_nodata, result)
            
            # Update profile
            profile.update({
                'dtype': 'float32',
                'nodata': output_nodata,
                'compress': compress,
                'tiled': tiled,
                'count': 1,
            })
            
            if tiled:
                profile['blockxsize'] = blocksize
                profile['blockysize'] = blocksize
            
            # Create quarterly output folder
            quarter_folder_name = f"{year}_Q{quarter}"
            quarter_out_dir = out_root / quarter_folder_name
            quarter_out_dir.mkdir(parents=True, exist_ok=True)
            
            # Output filename
            out_name = f"{output_prefix}_std.tif"
            out_path = quarter_out_dir / out_name
            
            # Write output
            with rasterio.open(out_path, 'w', **profile) as dst:
                dst.write(result.astype(np.float32), 1)
            
            # Calculate statistics
            valid_data = result[result != output_nodata]
            if len(valid_data) > 0:
                stats = f"n={len(all_data)} months, min={valid_data.min():.4f}, max={valid_data.max():.4f}, mean={valid_data.mean():.4f}"
            else:
                stats = f"n={len(all_data)} months, no valid pixels"
            
            print(f"  SUCCESS: {quarter_folder_name}/{out_name} ({stats})")
            output_paths[(year, quarter)] = out_path
            
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"Completed: {len(output_paths)} quarterly std images created")
    if skipped_folders:
        print(f"WARNING: Skipped {len(skipped_folders)} monthly folders")
    print(f"{'='*60}")
    
    return output_paths


def calculate_monthly_std_tree(
    in_root: str | Path,
    out_root: str | Path,
    band_pattern: str = "VV*.tif",
    output_suffix: str = "_std",
    compress: str = "deflate",
    tiled: bool = True,
    blocksize: int = 512,
    nodata: float | None = None,
) -> Tuple[List[Path], List[Path]]:
    """
    Calculate standard deviation for each monthly mosaic from the individual scenes.
    
    For each monthly folder, reads all matching images and calculates the pixel-wise
    standard deviation across all scenes in that month.
    
    Writes output to the same monthly folder structure:
    out_root/<month>/<original_filename><suffix>.tif
    
    Parameters
    ----------
    in_root : path
        Root folder containing monthly subfolders with individual scene images
    out_root : path
        Output root folder (will mirror input structure)
    band_pattern : str
        Glob pattern to match files in each monthly subfolder
        Examples: "VV*.tif", "VH*.tif", "*ratio*.tif"
    output_suffix : str
        Suffix to add to output filename (default: "_std")
        Example: "VV_mosaic.tif" + "_std" = "VV_mosaic_std.tif"
    compress : str
        Compression method for output (default: "deflate")
    tiled : bool
        Write tiled output (default: True)
    blocksize : int
        Block/tile size for output (default: 512)
    nodata : float, optional
        NoData value for output. If None, uses source nodata or -9999
    
    Returns
    -------
    processed : List[Path]
        List of successfully created std files
    skipped : List[Path]
        List of monthly folders that were skipped
    
    Examples
    --------
    # Calculate std for VV in each month
    >>> vv_std, skipped = calculate_monthly_std_tree(
    ...     in_root=s1_mosaic_10m_res__dir,
    ...     out_root=s1_mosaic_10m_res_std__dir,
    ...     band_pattern="VV*.tif",
    ...     output_suffix="_std"
    ... )
    
    # Calculate std for VH
    >>> vh_std, skipped = calculate_monthly_std_tree(
    ...     in_root=s1_mosaic_10m_res__dir,
    ...     out_root=s1_mosaic_10m_res_std__dir,
    ...     band_pattern="VH*.tif",
    ...     output_suffix="_std"
    ... )
    """
    in_root = Path(in_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    
    # Find all monthly subfolders
    monthly_folders = sorted([p for p in in_root.iterdir() if p.is_dir()])
    
    if not monthly_folders:
        print(f"WARNING: No subfolders found in {in_root}")
        return [], []
    
    print(f"\nFound {len(monthly_folders)} monthly folders")
    
    processed = []
    skipped = []
    
    for month_folder in monthly_folders:
        print(f"\nProcessing {month_folder.name}...")
        
        # Find all matching files in this month
        raster_paths = sorted(list(month_folder.glob(band_pattern)))
        
        if not raster_paths:
            print(f"  WARNING: No files matching '{band_pattern}'")
            skipped.append(month_folder)
            continue
        
        if len(raster_paths) < 2:
            print(f"  WARNING: Only {len(raster_paths)} image(s) found, need at least 2 for std")
            skipped.append(month_folder)
            continue
        
        print(f"  Found {len(raster_paths)} images")
        
        try:
            # Read metadata from first raster
            with rasterio.open(raster_paths[0]) as src:
                profile = src.profile.copy()
                shape = src.shape
                src_nodata = src.nodata
            
            output_nodata = nodata if nodata is not None else (src_nodata if src_nodata is not None else -9999)
            
            # Initialize list to collect all data
            all_data = []
            valid_files = []
            
            # Read all rasters
            for raster_path in raster_paths:
                with rasterio.open(raster_path) as src:
                    # Verify dimensions match
                    if src.shape != shape:
                        print(f"  WARNING {raster_path.name}: dimension mismatch, skipping")
                        continue
                    
                    data = src.read(1).astype(np.float32)
                    
                    # Mask nodata
                    if src.nodata is not None:
                        data = np.where(data == src.nodata, np.nan, data)
                    
                    all_data.append(data)
                    valid_files.append(raster_path)
            
            if len(all_data) < 2:
                print(f"  ERROR: Less than 2 valid images after filtering")
                skipped.append(month_folder)
                continue
            
            # Stack all images
            stacked = np.stack(all_data, axis=0)  # Shape: (n_images, height, width)
            
            # Calculate standard deviation (ignoring NaN)
            with np.errstate(invalid='ignore'):  # Suppress warnings for all-NaN slices
                result = np.nanstd(stacked, axis=0, ddof=1)  # ddof=1 for sample std
            
            # Set nodata where all values were NaN
            result = np.where(np.isnan(result), output_nodata, result)
            
            # Update profile
            profile.update({
                'dtype': 'float32',
                'nodata': output_nodata,
                'compress': compress,
                'tiled': tiled,
                'count': 1,
            })
            
            if tiled:
                profile['blockxsize'] = blocksize
                profile['blockysize'] = blocksize
            
            # Create output folder
            month_out_dir = out_root / month_folder.name
            month_out_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine output filename based on first valid file
            # Extract stem and add suffix
            base_name = valid_files[0].stem  # e.g., "VV_mosaic"
            out_name = f"{base_name}{output_suffix}.tif"
            out_path = month_out_dir / out_name
            
            # Write output
            with rasterio.open(out_path, 'w', **profile) as dst:
                dst.write(result.astype(np.float32), 1)
            
            # Calculate statistics
            valid_data = result[result != output_nodata]
            if len(valid_data) > 0:
                stats = f"n={len(all_data)}, min={valid_data.min():.4f}, max={valid_data.max():.4f}, mean={valid_data.mean():.4f}"
            else:
                stats = f"n={len(all_data)}, no valid pixels"
            
            print(f"  SUCCESS: {out_name} ({stats})")
            processed.append(out_path)
            
        except Exception as e:
            print(f"  ERROR: {e}")
            skipped.append(month_folder)
            continue
    
    print(f"\n{'='*60}")
    print(f"Completed: {len(processed)} monthly std images created")
    if skipped:
        print(f"WARNING: Skipped {len(skipped)} folders")
    print(f"{'='*60}")
    
    return processed, skipped



###############################################################################
## RESAMPLE TO QUARTERLY MEAN
###############################################################################
from pathlib import Path
from typing import List, Dict, Tuple, Literal
from collections import defaultdict
from datetime import datetime
import numpy as np
import rasterio
from rasterio.transform import Affine


def get_quarter_from_date(date_str: str, date_format: str = "%Y-%m-%d") -> Tuple[int, int]:
    """
    Extract year and quarter from a date string.
    
    Parameters
    ----------
    date_str : str
        Date string (e.g., "2020-01-15")
    date_format : str
        Format of the date string (default: "%Y-%m-%d")
    
    Returns
    -------
    year : int
    quarter : int (1-4)
    """
    try:
        dt = datetime.strptime(date_str, date_format)
        quarter = (dt.month - 1) // 3 + 1
        return dt.year, quarter
    except ValueError as e:
        raise ValueError(f"Could not parse date '{date_str}' with format '{date_format}': {e}")


def calculate_quarterly_means_tree(
    in_root: str | Path,
    out_root: str | Path,
    band_pattern: str = "VV*.tif",
    output_prefix: str = "VV",
    date_format: str = "%Y-%m-%d",
    aggregation: Literal["mean", "median", "min", "max"] = "mean",
    compress: str = "deflate",
    tiled: bool = True,
    blocksize: int = 512,
    nodata: float | None = None,
) -> Dict[Tuple[int, int], Path]:
    """
    Walks subfolders of in_root (named by date), groups them by quarter,
    and calculates quarterly aggregations (mean, median, etc.).
    
    Writes output to out_root/<year>_Q<quarter>/<prefix>_<aggregation>.tif

    Parameters
    ----------
    in_root : path
        Root folder containing date-named subfolders (e.g., "2020-01-15")
    out_root : path
        Output root folder (will be created)
    band_pattern : str
        Glob pattern to match files in each subfolder
        Examples: "VV*.tif", "VH*.tif", "VV_VH_ratio.tif"
    output_prefix : str
        Prefix for output filename (e.g., "VV", "VH", "ratio")
    date_format : str
        Format of subfolder names (default: "%Y-%m-%d")
    aggregation : str
        Aggregation method: "mean", "median", "min", "max"
    compress : str
        Compression method for output (default: "deflate")
    tiled : bool
        Write tiled output (default: True)
    blocksize : int
        Block/tile size for output (default: 512)
    nodata : float, optional
        NoData value for output. If None, uses source nodata or -9999

    Returns
    -------
    output_paths : dict
        Dictionary mapping (year, quarter) to output Path
    """
    in_root = Path(in_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Find all subfolders (date folders)
    subfolders = [p for p in in_root.iterdir() if p.is_dir()]
    if not subfolders:
        raise ValueError(f"No subfolders found in {in_root}")

    # Group folders by quarter
    quarterly_groups: Dict[Tuple[int, int], List[Path]] = defaultdict(list)
    skipped_folders = []

    for sub in subfolders:
        try:
            year, quarter = get_quarter_from_date(sub.name, date_format)
            
            # Check if the file exists in this folder
            matching_files = list(sub.glob(band_pattern))
            if matching_files:
                quarterly_groups[(year, quarter)].append(sub)
            else:
                print(f"WARNING {sub.name}: no file matching '{band_pattern}'")
                skipped_folders.append(sub)
        except ValueError as e:
            print(f"WARNING {sub.name}: skipped - {e}")
            skipped_folders.append(sub)
            continue

    if not quarterly_groups:
        raise ValueError(f"No valid quarterly groups found in {in_root}")

    print(f"\nFound {len(quarterly_groups)} quarters with data:")
    for (year, quarter), folders in sorted(quarterly_groups.items()):
        print(f"  {year} Q{quarter}: {len(folders)} images")

    # Mapping for aggregation functions
    agg_funcs = {
        "mean": np.mean,
        "median": np.median,
        "min": np.min,
        "max": np.max,
    }
    
    if aggregation not in agg_funcs:
        raise ValueError(f"Invalid aggregation '{aggregation}'. Choose from: {list(agg_funcs.keys())}")
    
    agg_func = agg_funcs[aggregation]

    output_paths = {}

    # Process each quarter
    for (year, quarter), folders in sorted(quarterly_groups.items()):
        print(f"\nProcessing {year} Q{quarter} ({len(folders)} images)...")

        # Create quarterly output folder
        quarter_folder_name = f"{year}_Q{quarter}"
        quarter_out_dir = out_root / quarter_folder_name
        quarter_out_dir.mkdir(parents=True, exist_ok=True)

        # Collect all rasters for this quarter
        raster_paths = []
        for folder in folders:
            matching_files = list(folder.glob(band_pattern))
            if matching_files:
                raster_paths.append(matching_files[0])  # Take first match

        if not raster_paths:
            print(f"  ERROR: No valid rasters found")
            continue

        try:
            # Read metadata from first raster
            with rasterio.open(raster_paths[0]) as src:
                profile = src.profile.copy()
                shape = src.shape
                transform = src.transform
                crs = src.crs
                src_nodata = src.nodata

            output_nodata = nodata if nodata is not None else (src_nodata if src_nodata is not None else -9999)

            # Initialize array to collect all data
            all_data = []

            # Read all rasters
            for raster_path in raster_paths:
                with rasterio.open(raster_path) as src:
                    # Verify dimensions match
                    if src.shape != shape:
                        print(f"  WARNING {raster_path.parent.name}: dimension mismatch, skipping")
                        continue

                    data = src.read(1).astype(np.float32)
                    
                    # Mask nodata
                    if src.nodata is not None:
                        data = np.where(data == src.nodata, np.nan, data)
                    
                    all_data.append(data)

            if not all_data:
                print(f"  ERROR: No valid data to aggregate")
                continue

            # Stack all images
            stacked = np.stack(all_data, axis=0)  # Shape: (n_images, height, width)

            # Calculate aggregation (ignoring NaN)
            with np.errstate(invalid='ignore'):  # Suppress warnings for all-NaN slices
                if aggregation == "mean":
                    result = np.nanmean(stacked, axis=0)
                elif aggregation == "median":
                    result = np.nanmedian(stacked, axis=0)
                elif aggregation == "min":
                    result = np.nanmin(stacked, axis=0)
                elif aggregation == "max":
                    result = np.nanmax(stacked, axis=0)

            # Set nodata where all values were NaN
            result = np.where(np.isnan(result), output_nodata, result)

            # Update profile
            profile.update({
                'dtype': 'float32',
                'nodata': output_nodata,
                'compress': compress,
                'tiled': tiled,
                'count': 1,
            })

            if tiled:
                profile['blockxsize'] = blocksize
                profile['blockysize'] = blocksize

            # Output filename (without year_Q prefix since it's in the folder name)
            out_name = f"{output_prefix}_{aggregation}.tif"
            out_path = quarter_out_dir / out_name

            # Write output
            with rasterio.open(out_path, 'w', **profile) as dst:
                dst.write(result.astype(np.float32), 1)

            # Calculate statistics
            valid_data = result[result != output_nodata]
            if len(valid_data) > 0:
                stats = f"min={valid_data.min():.2f}, max={valid_data.max():.2f}, mean={valid_data.mean():.2f}"
            else:
                stats = "no valid pixels"

            print(f"  SUCCESS: {quarter_folder_name}/{out_name} ({stats})")
            output_paths[(year, quarter)] = out_path

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    print(f"\n{'='*60}")
    print(f"Completed: {len(output_paths)} quarterly images created")
    if skipped_folders:
        print(f"WARNING: Skipped {len(skipped_folders)} folders")
    print(f"{'='*60}")

    return output_paths



###############################################################################
## TRANSFORM TO DB SCALE
###############################################################################
from pathlib import Path
from typing import Callable, List, Tuple
import numpy as np
import rasterio


def linear_to_db_transformer(
    min_value: float = 1e-10,
    nodata: float | None = None,
    pattern_out: str | None = None,
) -> Callable[[Path, Path], bool]:
    """
    Returns a transformation function that converts SAR linear power to dB scale.
    
    Formula: dB = 10 * log10(linear)
    
    Parameters
    ----------
    min_value : float
        Minimum value to clip before log conversion (to avoid log(0) or log(negative))
        Default: 1e-10
    nodata : float, optional
        NoData value for output. If None, uses source nodata or -9999
    pattern_out : str, optional
        Output filename pattern. Use '*' as placeholder for the input filename stem.
        Examples: "*_dB.tif", "dB_*.tif", "*_decibel.tif"
        If None, uses the same filename as input
    
    Returns
    -------
    callable
        A function that can be passed to process_tree
    """
    
    def _to_db(in_path: Path, out_path: Path) -> bool:
        """Returns True if converted, False if skipped."""
        try:
            # Modify output filename if pattern_out is specified
            if pattern_out is not None:
                # Extract stem (filename without extension) from input
                stem = in_path.stem
                # Replace '*' in pattern with the stem
                new_name = pattern_out.replace('*', stem)
                # Update output path with new filename
                out_path = out_path.parent / new_name
            
            with rasterio.open(in_path) as src:
                # Read data
                data = src.read(1).astype(np.float32)
                
                # Get nodata value
                src_nodata = src.nodata
                output_nodata = nodata if nodata is not None else (src_nodata if src_nodata is not None else -9999)
                
                # Create mask for valid data
                if src_nodata is not None:
                    valid_mask = (data != src_nodata)
                else:
                    valid_mask = np.ones_like(data, dtype=bool)
                
                # Also mask out zero and negative values
                valid_mask &= (data > 0)
                
                # Initialize output with nodata
                db_data = np.full_like(data, output_nodata, dtype=np.float32)
                
                # Convert valid pixels to dB
                # Clip to minimum value to avoid log issues
                linear_clipped = np.maximum(data[valid_mask], min_value)
                db_data[valid_mask] = 10 * np.log10(linear_clipped)
                
                # Prepare output metadata
                profile = src.profile.copy()
                profile.update({
                    'dtype': 'float32',
                    'nodata': output_nodata,
                })
                
                # Create output directory if needed
                out_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Write output
                with rasterio.open(out_path, 'w', **profile) as dst:
                    dst.write(db_data.astype(np.float32), 1)
                
                # Calculate statistics
                valid_db = db_data[valid_mask]
                if len(valid_db) > 0:
                    stats = f"min={valid_db.min():.2f}dB, max={valid_db.max():.2f}dB, mean={valid_db.mean():.2f}dB"
                else:
                    stats = "no valid pixels"
                
                print(f"  SUCCESS: {in_path.name} -> {out_path.name} ({stats})")
                return True
                
        except Exception as e:
            print(f"  ERROR {in_path.name}: {e}")
            return False
    
    return _to_db


def convert_quarterly_to_db(
    in_root: str | Path,
    out_root: str | Path,
    pattern: str = "*_mean.tif",
    pattern_out: str | None = None,
    min_value: float = 1e-10,
    nodata: float | None = None,
) -> Tuple[List[Path], List[Path]]:
    """
    Convert quarterly SAR images from linear power to dB scale.
    Preserves the quarterly folder structure (e.g., 2017_Q1, 2017_Q2, etc.)
    
    Parameters
    ----------
    in_root : path
        Root folder containing quarterly subfolders with linear power images
    out_root : path
        Output root folder (will be created with same structure)
    pattern : str
        Glob pattern to match input files (default: "*_mean.tif")
    pattern_out : str, optional
        Output filename pattern. Use '*' as placeholder for the input filename stem.
        Examples: "*_dB.tif", "dB_*.tif", "*_decibel.tif"
        If None, uses the same filename as input
        Default: None
    min_value : float
        Minimum value to clip before log conversion (default: 1e-10)
    nodata : float, optional
        NoData value for output. If None, uses source nodata or -9999
    
    Returns
    -------
    processed : List[Path]
        List of successfully processed files
    skipped : List[Path]
        List of files that were skipped or failed
    
    Examples
    --------
    # Convert with default output names (same as input)
    >>> convert_quarterly_to_db(
    ...     in_root="s1_Q_mosaic",
    ...     out_root="s1_Q_mosaic_dB",
    ...     pattern="*_mean.tif"
    ... )
    
    # Convert with "_dB" suffix
    >>> convert_quarterly_to_db(
    ...     in_root="s1_Q_mosaic",
    ...     out_root="s1_Q_mosaic_dB",
    ...     pattern="*_mean.tif",
    ...     pattern_out="*_dB.tif"
    ... )
    # Output: VV_Q_mean.tif -> VV_Q_mean_dB.tif
    
    # Convert with "dB_" prefix
    >>> convert_quarterly_to_db(
    ...     in_root="s1_Q_mosaic",
    ...     out_root="s1_Q_mosaic_dB",
    ...     pattern="*_mean.tif",
    ...     pattern_out="dB_*.tif"
    ... )
    # Output: VV_Q_mean.tif -> dB_VV_Q_mean.tif
    """
    transform = linear_to_db_transformer(
        min_value=min_value,
        nodata=nodata,
        pattern_out=pattern_out,
    )
    
    processed, skipped = process_tree(
        in_root=in_root,
        pattern=pattern,
        recursive=True,
        transform_fn=transform,
        out_root=out_root,
        preserve_structure=True,
    )
    
    return processed, skipped


def convert_monthly_to_db(
    in_root: str | Path,
    out_root: str | Path,
    pattern: str = "*.tif",
    pattern_out: str | None = None,
    min_value: float = 1e-10,
    nodata: float | None = None,
) -> Tuple[List[Path], List[Path]]:
    """
    Convert monthly SAR images from linear power to dB scale.
    Preserves the monthly folder structure (e.g., 2017_01, 2017_02, etc.)
    
    Parameters
    ----------
    in_root : path
        Root folder containing monthly subfolders with linear power images
    out_root : path
        Output root folder (will be created with same structure)
    pattern : str
        Glob pattern to match input files (default: "*.tif")
        Examples: "*.tif", "VV*.tif", "VH*.tif", "*_mosaic.tif"
    pattern_out : str, optional
        Output filename pattern. Use '*' as placeholder for the input filename stem.
        Examples: "*_dB.tif", "dB_*.tif", "*_decibel.tif"
        If None, uses the same filename as input
        Default: None
    min_value : float
        Minimum value to clip before log conversion (default: 1e-10)
    nodata : float, optional
        NoData value for output. If None, uses source nodata or -9999
    
    Returns
    -------
    processed : List[Path]
        List of successfully processed files
    skipped : List[Path]
        List of files that were skipped or failed
    
    Examples
    --------
    # Convert all monthly images with default names
    >>> convert_monthly_to_db(
    ...     in_root="s1_mosaic_10m_res",
    ...     out_root="s1_mosaic_10m_res_dB",
    ...     pattern="*.tif"
    ... )
    
    # Convert with "_dB" suffix
    >>> convert_monthly_to_db(
    ...     in_root="s1_mosaic_10m_res",
    ...     out_root="s1_mosaic_10m_res_dB",
    ...     pattern="VV*.tif",
    ...     pattern_out="*_dB.tif"
    ... )
    # Output: VV_mosaic.tif -> VV_mosaic_dB.tif
    
    # Convert only VH images with custom pattern
    >>> convert_monthly_to_db(
    ...     in_root="s1_mosaic_10m_res",
    ...     out_root="s1_mosaic_10m_res_dB",
    ...     pattern="VH*.tif",
    ...     pattern_out="*_decibel.tif"
    ... )
    # Output: VH_mosaic.tif -> VH_mosaic_decibel.tif
    """
    transform = linear_to_db_transformer(
        min_value=min_value,
        nodata=nodata,
        pattern_out=pattern_out,
    )
    
    processed, skipped = process_tree(
        in_root=in_root,
        pattern=pattern,
        recursive=True,
        transform_fn=transform,
        out_root=out_root,
        preserve_structure=True,
    )
    
    return processed, skipped



