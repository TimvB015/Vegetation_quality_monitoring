from functions.raster_io_funcs import get_nodata_value

################################################################################
## ENTROPY CALCULATOR
################################################################################
from pathlib import Path
from typing import Union, Dict, Optional
import numpy as np
import rasterio


def calculate_entropy(stack: np.ndarray, nodata_value: float) -> np.ndarray:
    """
    Calculate normalized Shannon entropy for each pixel across the ensemble stack.
    ULTRA-LOW-MEMORY VERSION.
    """
    n_bands, height, width = stack.shape
    entropy = np.full((height, width), nodata_value, dtype=np.float32)
    
    # Create mask for pixels with any NoData
    nodata_mask = np.any(stack == nodata_value, axis=0)
    valid_mask = ~nodata_mask
    
    if not np.any(valid_mask):
        return entropy
    
    # Flatten for direct indexing
    entropy_flat = entropy.ravel()
    valid_mask_flat = valid_mask.ravel()
    valid_indices = np.where(valid_mask_flat)[0]
    n_valid = len(valid_indices)
    
    print(f"  Processing {n_valid:,} valid pixels for entropy...")
    
    # Process pixel-by-pixel
    for i, flat_idx in enumerate(valid_indices):
        # Extract values for this pixel across all bands
        row = flat_idx // width
        col = flat_idx % width
        values = stack[:, row, col]
        
        unique, counts = np.unique(values, return_counts=True)
        
        if len(unique) == 1:
            entropy_flat[flat_idx] = 0.0
        else:
            probabilities = counts / n_bands
            h = -np.sum(probabilities * np.log2(probabilities))
            entropy_flat[flat_idx] = h / np.log2(len(unique))
        
        # Progress indicator
        if (i + 1) % 1_000_000 == 0:
            print(f"    Processed {i+1:,}/{n_valid:,} pixels ({100*(i+1)/n_valid:.1f}%)")
    
    # Reshape back
    entropy = entropy_flat.reshape((height, width))
    
    return entropy


################################################################################
## MODAL AND FREQUENCY CALCULATOR
################################################################################
def calculate_modal_and_frequencies(
    stack: np.ndarray, 
    nodata_value: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate modal class, second class, and their frequencies for each pixel.
    ULTRA-LOW-MEMORY VERSION - no large intermediate arrays.
    """
    n_bands, height, width = stack.shape
    
    modal_class = np.full((height, width), nodata_value, dtype=np.int16)
    modal_frequency = np.full((height, width), nodata_value, dtype=np.float32)
    second_class = np.full((height, width), nodata_value, dtype=np.int16)
    second_frequency = np.full((height, width), 0.0, dtype=np.float32)
    
    # Create mask for pixels with any NoData
    nodata_mask = np.any(stack == nodata_value, axis=0)
    valid_mask = ~nodata_mask
    
    if not np.any(valid_mask):
        return modal_class, modal_frequency, second_class, second_frequency
    
    # Flatten arrays for direct indexing (avoids the [mask][idx] double indexing)
    modal_class_flat = modal_class.ravel()
    modal_frequency_flat = modal_frequency.ravel()
    second_class_flat = second_class.ravel()
    second_frequency_flat = second_frequency.ravel()
    valid_mask_flat = valid_mask.ravel()
    
    # Get indices of valid pixels
    valid_indices = np.where(valid_mask_flat)[0]
    n_valid = len(valid_indices)
    
    print(f"  Processing {n_valid:,} valid pixels...")
    
    for i, flat_idx in enumerate(valid_indices):
        # Extract values for this pixel across all bands
        row = flat_idx // width
        col = flat_idx % width
        values = stack[:, row, col]
        
        # Count unique values (only allocates small arrays for this one pixel)
        unique, counts = np.unique(values, return_counts=True)
        
        if len(unique) == 1:
            # Only one class
            modal_class_flat[flat_idx] = unique[0]
            modal_frequency_flat[flat_idx] = 1.0
            second_class_flat[flat_idx] = int(nodata_value)
            second_frequency_flat[flat_idx] = 0.0
        elif len(unique) == 2:
            # Two classes - simple comparison
            if counts[0] >= counts[1]:
                modal_class_flat[flat_idx] = unique[0]
                modal_frequency_flat[flat_idx] = counts[0] / n_bands
                second_class_flat[flat_idx] = unique[1]
                second_frequency_flat[flat_idx] = counts[1] / n_bands
            else:
                modal_class_flat[flat_idx] = unique[1]
                modal_frequency_flat[flat_idx] = counts[1] / n_bands
                second_class_flat[flat_idx] = unique[0]
                second_frequency_flat[flat_idx] = counts[0] / n_bands
        else:
            # Multiple classes - find top 2 using argmax (NO ARGSORT!)
            max_idx = np.argmax(counts)
            modal_class_flat[flat_idx] = unique[max_idx]
            modal_frequency_flat[flat_idx] = counts[max_idx] / n_bands
            
            # Find second by masking out the first
            counts_copy = counts.copy()
            counts_copy[max_idx] = -1
            second_idx = np.argmax(counts_copy)
            second_class_flat[flat_idx] = unique[second_idx]
            second_frequency_flat[flat_idx] = counts[second_idx] / n_bands
        
        # Progress indicator
        if (i + 1) % 1_000_000 == 0:
            print(f"    Processed {i+1:,}/{n_valid:,} pixels ({100*(i+1)/n_valid:.1f}%)")
    
    # Reshape back to 2D
    modal_class = modal_class_flat.reshape((height, width))
    modal_frequency = modal_frequency_flat.reshape((height, width))
    second_class = second_class_flat.reshape((height, width))
    second_frequency = second_frequency_flat.reshape((height, width))
    
    return modal_class, modal_frequency, second_class, second_frequency



################################################################################
## DECISION CATEGORY CALCULATOR
################################################################################
def calculate_decision_category(
    modal_frequency: np.ndarray,
    second_frequency: np.ndarray,
    nodata_value: float,
    true_winner_min: float,
    med_winner_min: float,
    majority_min: float,
    tie_min: float
) -> np.ndarray:
    """
    Calculate decision category for each pixel based on frequency thresholds.
    
    Categories:
    1 = True Winner (>= true_winner_min, default 80-100%)
    2 = Med Winner (>= med_winner_min, default 70-80%)
    3 = Majority (>= majority_min, default 60-70%)
    4 = Tie (both classes > tie_min, default both > 40%)
    5 = Remaining (everything else)
    """
    # Initialize with nodata
    decision = np.full_like(modal_frequency, nodata_value, dtype=np.int16)
    
    # Create NoData mask
    valid_mask = modal_frequency != nodata_value
    
    # Start with category 5 (Remaining) for all valid pixels
    decision[valid_mask] = 5
    
    # Apply decision tree logic using vectorized operations (order matters!)
    # Category 2: Majority
    decision[(valid_mask) & (modal_frequency >= majority_min)] = 3
    
    # Category 2: Med Winner
    decision[(valid_mask) & (modal_frequency >= med_winner_min)] = 2
    
    # Category 4: Tie (can override Med Winner if both are high)
    decision[(valid_mask) & (modal_frequency > tie_min) & (second_frequency > tie_min)] = 4
    
    # Category 1: True Winner (highest priority)
    decision[(valid_mask) & (modal_frequency >= true_winner_min)] = 1
    
    return decision



################################################################################
## PROCESSING RASTER STACK TO PIXEL STABILITY LAYERS
################################################################################
from rasterio.windows import Window

def process_ensemble_stack_to_pixel_stability_layers(
    raster_stack_path: Union[str, Path],
    output_path: Union[str, Path],
    nodata_value: float,
    *,
    year_bands_to_include: Optional[list] = None,
    true_winner_min: float = 0.80,
    med_winner_min: float = 0.70,
    majority_min: float = 0.60,
    tie_min: float = 0.40,
    overwrite: bool = False,
    chunk_size: int = 2048,
) -> Dict[str, np.ndarray]:
    
    import time
    
    raster_stack_path = Path(raster_stack_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("PROCESSING ENSEMBLE STACK TO PIXEL STABILITY LAYERS")
    print("="*80)
    
    total_start = time.time()
    
    # Read the raster stack
    t0 = time.time()
    with rasterio.open(raster_stack_path) as src:
        total_bands = src.count
        height = src.height
        width = src.width
        profile = src.profile.copy()
        
        # Determine which bands to read
        if year_bands_to_include is None:
            bands_to_read = list(range(1, total_bands + 1))
        else:
            if not all(1 <= b <= total_bands for b in year_bands_to_include):
                raise ValueError(
                    f"Invalid band indices in year_bands_to_include. "
                    f"Must be between 1 and {total_bands}."
                )
            bands_to_read = sorted(year_bands_to_include)
        
        n_bands = len(bands_to_read)
        
        print(f"Stack: {n_bands} bands, {height}x{width} pixels")
        print(f"Total pixels: {height * width:,}")
    
        # Initialize output arrays
        modal_class = np.full((height, width), nodata_value, dtype=np.int16)
        modal_frequency = np.full((height, width), nodata_value, dtype=np.float32)
        second_class = np.full((height, width), nodata_value, dtype=np.int16)
        second_frequency = np.full((height, width), 0.0, dtype=np.float32)
        entropy_normalized = np.full((height, width), nodata_value, dtype=np.float32)
        
        # Decide whether to chunk
        total_pixels = height * width
        estimated_memory_gb = (n_bands * height * width * 4) / (1024**3)
        
        if estimated_memory_gb > 8 or total_pixels > 100_000_000:
            print(f"Large raster detected ({estimated_memory_gb:.1f} GB)")
            print(f"Processing in chunks of {chunk_size}x{chunk_size} pixels")
            use_chunking = True
        else:
            print(f"Raster size: {estimated_memory_gb:.1f} GB - processing entire raster at once")
            use_chunking = False
        
        if not use_chunking:
            # Read entire stack
            print(f"Reading {n_bands} bands from disk...")
            t_read = time.time()
            if year_bands_to_include is None:
                stack = src.read()
            else:
                stack = src.read(bands_to_read)
            print(f"  -> Read time: {time.time() - t_read:.1f}s")
            print(f"  Stack shape: {stack.shape}, dtype: {stack.dtype}")
            
            # Count NoData pixels
            t_mask = time.time()
            nodata_mask = np.any(stack == nodata_value, axis=0)
            valid_mask = ~nodata_mask
            n_valid = np.sum(valid_mask)
            print(f"  Valid pixels: {n_valid:,} ({100*n_valid/total_pixels:.1f}%)")
            print(f"  -> Mask calculation: {time.time() - t_mask:.1f}s")
            
            # Calculate modal and frequencies
            print("Calculating modal class and frequencies...")
            t_modal = time.time()
            modal_class, modal_frequency, second_class, second_frequency = \
                calculate_modal_and_frequencies(stack, nodata_value=nodata_value)
            print(f"  -> Modal calculation: {time.time() - t_modal:.1f}s")
            
            # Calculate entropy
            print("Calculating entropy...")
            t_entropy = time.time()
            entropy_normalized = calculate_entropy(stack, nodata_value=nodata_value)
            print(f"  -> Entropy calculation: {time.time() - t_entropy:.1f}s")
            
            # Free memory
            del stack
            
        else:
            # [Keep your existing chunking code]
            from rasterio.windows import Window
            # ... chunking code ...
    
    print(f"Setup and reading: {time.time() - total_start:.1f}s")
    
    # Calculate decision category (fast, vectorized)
    print("Calculating decision categories...")
    t_decision = time.time()
    decision_category = calculate_decision_category(
        modal_frequency,
        second_frequency,
        nodata_value,
        true_winner_min,
        med_winner_min,
        majority_min,
        tie_min
    )
    print(f"  -> Decision category: {time.time() - t_decision:.1f}s")
    
    # Prepare output profile (single band)
    output_profile = profile.copy()
    output_profile.update({
        'count': 1,
        'dtype': 'float32',
        'nodata': nodata_value,
    })
    output_profile.pop('path', None)
    
    # Layer definitions
    layers = {
        'modal_class': {
            'data': modal_class,
            'description': 'Most frequent class',
            'dtype': 'int16',
        },
        'modal_frequency': {
            'data': modal_frequency,
            'description': 'Frequency of most common class [0-1]',
            'dtype': 'float32',
        },
        'second_class': {
            'data': second_class,
            'description': 'Second most frequent class',
            'dtype': 'int16',
        },
        'second_frequency': {
            'data': second_frequency,
            'description': 'Frequency of second most common class [0-1]',
            'dtype': 'float32',
        },
        'entropy_normalized': {
            'data': entropy_normalized,
            'description': 'Normalized Shannon entropy [0-1]',
            'dtype': 'float32',
        },
        'decision_category': {
            'data': decision_category,
            'description': f'Decision tree category: 1=True Winner (>={true_winner_min*100:.0f}%), 2=Med Winner (>={med_winner_min*100:.0f}%), 3=Majority (>={majority_min*100:.0f}%), 4=Tie (both>{tie_min*100:.0f}%), 5=Remaining',
            'dtype': 'int16',
        },
    }
    
    # Export layers
    print("Exporting layers...")
    t_export = time.time()
    result_dict = {}
    exported_count = 0
    
    for layer_name, layer_info in layers.items():
        output_file = output_path / f"{layer_name}.tif"
        layer_data = layer_info['data']
        description = layer_info['description']
        dtype = layer_info['dtype']
        
        # Check if file exists and overwrite is False
        if not overwrite and output_file.exists():
            print(f"  Skipping {layer_name} (already exists)")
            result_dict[layer_name] = layer_data
            continue
        
        # Update profile for this specific layer
        layer_profile = output_profile.copy()
        layer_profile['dtype'] = dtype
        
        # Prepare export data
        export_data = layer_data.astype(dtype)
        
        # Write to file
        with rasterio.open(output_file, 'w', **layer_profile) as dst:
            dst.write(export_data, 1)
            dst.set_band_description(1, description)
        
        result_dict[layer_name] = layer_data
        exported_count += 1
    
    print(f"  -> Export time: {time.time() - t_export:.1f}s")
    print(f"Exported {exported_count}/{len(layers)} layers to: {output_path}")
    
    # Decision category distribution
    valid_mask = decision_category != nodata_value
    if np.any(valid_mask):
        print("\nPixel Stability Distribution:")
        for cat_id, cat_name in [
            (1, "True Winner"),
            (2, "Med Winner"),
            (3, "Majority"),
            (4, "Tie"),
            (5, "Remaining")
        ]:
            count = np.sum(decision_category == cat_id)
            pct = (count / np.sum(valid_mask)) * 100 if np.sum(valid_mask) > 0 else 0
            print(f"   {cat_id} - {cat_name:12s}: {count:8d} ({pct:5.2f}%)")
    
    print(f"\n{'='*80}")
    print(f"TOTAL PROCESSING TIME: {time.time() - total_start:.1f}s")
    print("="*80)
    
    return result_dict



################################################################################
## ENTROPY STACK (WRAPPER FUNCTION)
################################################################################
def build_pixel_stability_stack(
    carto_or_RF: str,
    hab_selection: str,
    train_split_attempt: str,
    habitat_reference_df_path: Union[str, Path],
    raster_stack_dir: Union[str, Path],
    output_stability_dir: Union[str, Path],
    *,
    raster_filename: Optional[str] = None,
    band_selection: Optional[str] = None,
    yearband_selection: Optional[str] = None,
    year_bands_to_include: Optional[list] = None,
    timeframe: Optional[str] = None,
    true_winner_min: float = 0.80,
    med_winner_min: float = 0.70,
    majority_min: float = 0.60,
    tie_min: float = 0.40,
    nodata_value: float | None = None,
    overwrite: bool = False,
) -> Dict[str, np.ndarray] | None:
    """
    Process ensemble stack to generate pixel stability analysis layers.
    
    This function automates the complete workflow:
      1) Locates the input raster stack based on naming convention
      2) Extracts or uses provided NoData value
      3) Processes ensemble stack to generate pixel stability layers
      4) Exports all layers as georeferenced GeoTIFFs
    
    Parameters
    ----------
    carto_or_RF : str
        Model type identifier ('carto', 'RF_row', 'RF_col', etc.).
    hab_selection : str
        Habitat division scheme (e.g., 'WD1', 'WD', 'WD2').
    train_split_attempt : str
        Training split attempt identifier (e.g., 'at1', 'at2').
    habitat_reference_df_path : str or Path
        Path to the habitat reference dataframe (pickle file).
    raster_stack_dir : str or Path
        Directory containing the input raster stack.
    output_stability_dir : str or Path
        Directory where output pixel stability layers will be saved.
    raster_filename : str, optional
        Explicit filename of the raster stack to process.
        If None, filename is constructed based on other parameters.
    band_selection : str, optional
        Band selection identifier (e.g., 'b28ndvwi', 'b15').
        Used for RF models and included in output naming.
    yearband_selection : str, optional
        Year/band selection descriptor for output naming (e.g., '2018_2024', 'recent').
        Included in output directory name after train_split_attempt.
    year_bands_to_include : list of int, optional
        List of band indices (1-based) to include in the analysis.
        If None, all bands are used. Example: [1, 3, 5] to use bands 1, 3, and 5.
    timeframe : str, optional
        Timeframe identifier to include in output directory name.
        If None, timeframe is not included in the name.
        Example: 'Q1234' or 'seasonal'
    true_winner_min : float, default=0.80
        Minimum frequency (0-1) for true winner classification.
    med_winner_min : float, default=0.70
        Minimum frequency (0-1) for medium winner classification.
    majority_min : float, default=0.60
        Minimum frequency (0-1) for majority classification.
    tie_min : float, default=0.40
        Minimum frequency (0-1) for both classes in a tie.
    nodata_value : float or None, default=None
        NoData value to use for raster processing.
        If None, extracted from raster metadata.
    overwrite : bool, default=False
        If False, skips processing if output files already exist.
        If True, overwrites existing output files.
    
    Returns
    -------
    dict or None
        Dictionary containing all calculated layers as numpy arrays.
        Returns None if files exist and overwrite=False.
    """
    
    # Convert paths to Path objects
    habitat_reference_df_path = Path(habitat_reference_df_path)
    raster_stack_dir = Path(raster_stack_dir)
    output_stability_dir = Path(output_stability_dir)
    
    print("="*80)
    print(f"PIXEL STABILITY WORKFLOW: {hab_selection}_{train_split_attempt}")
    if yearband_selection:
        print(f"Yearband selection: {yearband_selection}")
    if timeframe:
        print(f"Timeframe: {timeframe}")
    if band_selection:
        print(f"Band selection: {band_selection}")
    print("="*80)
    
    # =========================================================================
    # Build output directory name
    # =========================================================================
    output_parts = [f"{carto_or_RF}_stability", hab_selection, train_split_attempt]
    
    # Add yearband_selection (before band_selection)
    if yearband_selection:
        output_parts.append(yearband_selection)
    
    # Add timeframe
    if timeframe:
        output_parts.append(timeframe)
    
    # Add band_selection (after yearband_selection)
    if band_selection:
        output_parts.append(band_selection)
    
    output_subdir_name = "__".join(output_parts)
    output_stability_path = output_stability_dir / output_subdir_name
    
    # =========================================================================
    # Check if output already exists
    # =========================================================================
    if not overwrite and output_stability_path.exists():
        # Check if key output files exist
        expected_files = [
            'modal_class.tif',
            'modal_frequency.tif',
            'second_class.tif',
            'second_frequency.tif',
            'entropy_normalized.tif',
            'decision_category.tif'
        ]
        
        existing_files = [f for f in expected_files if (output_stability_path / f).exists()]
        
        if len(existing_files) > 0:
            print(f"Output files already exist ({len(existing_files)}/{len(expected_files)})")
            print(f"Skipping: {output_stability_path}")
            print(f"Set overwrite=True to regenerate.")
            print("="*80)
            return None
    
    # =========================================================================
    # Locate raster stack
    # =========================================================================
    if raster_filename is not None:
        # Use explicitly provided filename
        raster_stack_path = raster_stack_dir / raster_filename
    elif band_selection is not None:
        # RF model pattern: stack_{band_selection}__{hab}_{attempt}_{timeframe}__rstr.tif
        if timeframe:
            raster_filename = f"stack_{band_selection}__{hab_selection}_{train_split_attempt}_{timeframe}__rstr.tif"
        else:
            raster_filename = f"stack_{band_selection}__{hab_selection}_{train_split_attempt}__rstr.tif"
        raster_stack_path = raster_stack_dir / raster_filename
    else:
        # Carto model pattern: {hab_selection}_{train_split_attempt}_gelderland_stacked_rstrs.tif
        raster_filename = f"{hab_selection}_{train_split_attempt}_gelderland_stacked_rstrs.tif"
        raster_stack_path = raster_stack_dir / raster_filename
    
    if not raster_stack_path.exists():
        raise FileNotFoundError(
            f"Raster stack not found: {raster_stack_path}\n"
            f"Expected filename: {raster_filename}"
        )
    
    print(f"Input: {raster_filename}")
    
    # =========================================================================
    # Get NoData value
    # =========================================================================
    if nodata_value is None:
        nodata_value = get_nodata_value(raster_path=raster_stack_path)
        
        if nodata_value is None:
            raise ValueError(
                "Could not determine NoData value from raster metadata. "
                "Please provide nodata_value parameter explicitly."
            )
        
        print(f"NoData value: {nodata_value} (extracted from metadata)")
    else:
        print(f"NoData value: {nodata_value} (provided)")
    
    # =========================================================================
    # Set up output path
    # =========================================================================
    output_stability_path.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_stability_path}")
    
    # =========================================================================
    # Process ensemble stack
    # =========================================================================
    layers = process_ensemble_stack_to_pixel_stability_layers(
        raster_stack_path=raster_stack_path,
        output_path=output_stability_path,
        nodata_value=nodata_value,
        year_bands_to_include=year_bands_to_include,
        true_winner_min=true_winner_min,
        med_winner_min=med_winner_min,
        majority_min=majority_min,
        tie_min=tie_min,
        overwrite=overwrite,
    )
    
    # =========================================================================
    # Summary
    # =========================================================================
    print(f"Successfully processed: {hab_selection}_{train_split_attempt}")
    print(f"Layers returned: {len(layers)}")
    print("="*80)
    
    return layers



################################################################################
## CARTO ENTROPY BATCH FUNC
################################################################################
import gc
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import Union, Optional, Dict, Tuple
from pathlib import Path
import numpy as np


def _process_single_combination_worker(
    args: Tuple[str, str, Path],
    habitat_reference_df_path: Path,
    raster_stack_dir: Path,
    output_stability_dir: Path,
    yearband_selection: Optional[str],
    timeframe: Optional[str],
    year_bands_to_include: Optional[list],
    true_winner_min: float,
    med_winner_min: float,
    majority_min: float,
    tie_min: float,
    nodata_value: Optional[float],
    overwrite: bool,
    return_arrays: bool,
) -> Tuple[str, str, Union[Dict, str, None], str]:
    """
    Worker function for parallel processing of a single combination.
    
    Returns
    -------
    tuple
        (hab, attempt, result, status)
        where status is 'success', 'skipped', or 'failed'
    """
    hab, attempt, raster_path = args
    
    try:
        layers = build_pixel_stability_stack(
            carto_or_RF="carto_col",
            hab_selection=hab,
            train_split_attempt=attempt,
            habitat_reference_df_path=habitat_reference_df_path,
            raster_stack_dir=raster_stack_dir,
            output_stability_dir=output_stability_dir,
            yearband_selection=yearband_selection,
            timeframe=timeframe,
            year_bands_to_include=year_bands_to_include,
            true_winner_min=true_winner_min,
            med_winner_min=med_winner_min,
            majority_min=majority_min,
            tie_min=tie_min,
            nodata_value=nodata_value,
            overwrite=overwrite,
        )
        
        if layers is None:
            # Skipped due to existing output
            gc.collect()
            return hab, attempt, None, 'skipped'
        else:
            # Success - build output path
            output_parts = ["carto_col_stability", hab, attempt]
            if yearband_selection:
                output_parts.append(yearband_selection)
            if timeframe:
                output_parts.append(timeframe)
            
            output_subdir = "__".join(output_parts)
            output_path = output_stability_dir / output_subdir
            
            if return_arrays:
                result = layers
            else:
                # Store only the output path
                result = str(output_path)
                del layers  # Free memory
            
            # Force garbage collection after processing
            gc.collect()
            
            return hab, attempt, result, 'success'
    
    except Exception as e:
        # Clean up memory even on failure
        gc.collect()
        return hab, attempt, str(e), 'failed'


def carto_pixel_stability_stack_batch(
    hab_selections: list[str],
    train_split_attempts: list[str],
    habitat_reference_df_path: Union[str, Path],
    raster_stack_dir: Union[str, Path],
    output_stability_dir: Union[str, Path],
    *,
    yearband_selection: Optional[str] = None,
    timeframe: Optional[str] = None,
    year_bands_to_include: Optional[list] = None,
    true_winner_min: float = 0.80,
    med_winner_min: float = 0.70,
    majority_min: float = 0.60,
    tie_min: float = 0.40,
    nodata_value: float | None = None,
    overwrite: bool = False,
    return_arrays: bool = False,
    n_workers: Optional[int] = None,
    use_parallel: bool = True,
) -> Dict[tuple, Dict[str, np.ndarray] | str | None]:
    """
    Process multiple habitat selections and training splits in batch for Carto models.
    
    Parameters
    ----------
    hab_selections : list of str
        List of habitat selections to process (e.g., ['WD1', 'WD2'])
    train_split_attempts : list of str
        List of training split attempts to process (e.g., ['at1', 'at2'])
    habitat_reference_df_path : str or Path
        Path to habitat reference dataframe
    raster_stack_dir : str or Path
        Directory containing raster stacks
    output_stability_dir : str or Path
        Output directory for pixel stability layers
    yearband_selection : str, optional
        Year/band selection descriptor for output naming (e.g., '2018_2024', 'recent').
        Included in output directory name after train_split_attempt.
    timeframe : str, optional
        Timeframe identifier to include in output directory name.
        Example: 'Q1234' or 'seasonal'
    year_bands_to_include : list of int, optional
        List of band indices (1-based) to include in the analysis.
    true_winner_min : float, default=0.80
        Minimum frequency (0-1) for true winner classification.
    med_winner_min : float, default=0.70
        Minimum frequency (0-1) for medium winner classification.
    majority_min : float, default=0.60
        Minimum frequency (0-1) for majority classification.
    tie_min : float, default=0.40
        Minimum frequency (0-1) for both classes in a tie.
    nodata_value : float or None, default=None
        NoData value to use for raster processing.
    overwrite : bool, default=False
        If False, skips processing for configurations where output files already exist.
    return_arrays : bool, default=False
        If False (default), returns output paths only (memory efficient).
        If True, returns numpy arrays for each layer (memory intensive).
    n_workers : int, optional
        Number of parallel workers. If None, uses (cpu_count() - 1).
        Set to 1 to disable parallel processing.
    use_parallel : bool, default=True
        If False, processes sequentially (useful for debugging).
    
    Returns
    -------
    dict
        Dictionary mapping (hab_selection, train_split_attempt) tuples to either:
        - Layer dicts containing numpy arrays (if return_arrays=True)
        - Output directory paths as strings (if return_arrays=False)
        - None for configurations that were skipped or failed.
    
    Notes
    -----
    Setting return_arrays=False is recommended for batch processing many large
    rasters to avoid memory issues. The raster files are saved to disk regardless
    of this setting.
    
    Parallel processing spawns separate processes, each with their own memory space.
    Memory is automatically cleaned up after each worker completes.
    
    Expected filename pattern: {hab_selection}_{train_split_attempt}_gelderland_stacked_rstrs.tif
    """
    
    # Convert to Path objects
    raster_stack_dir = Path(raster_stack_dir)
    output_stability_dir = Path(output_stability_dir)
    habitat_reference_df_path = Path(habitat_reference_df_path)
    
    total_combinations = len(hab_selections) * len(train_split_attempts)
    
    # Determine number of workers
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)  # Leave one core free
    
    print("="*80)
    print(f"BATCH PROCESSING: CARTO PIXEL STABILITY STACK ({total_combinations} combinations)")
    if use_parallel and n_workers > 1:
        print(f"Parallel processing: {n_workers} workers")
    else:
        print("Sequential processing")
    if yearband_selection:
        print(f"Yearband selection: {yearband_selection}")
    if timeframe:
        print(f"Timeframe: {timeframe}")
    print("="*80)
    
    # Find valid input files
    valid_combinations = []
    for hab in hab_selections:
        for attempt in train_split_attempts:
            raster_filename = f"{hab}_{attempt}_gelderland_stacked_rstrs.tif"
            raster_path = raster_stack_dir / raster_filename
            
            if raster_path.exists():
                valid_combinations.append((hab, attempt, raster_path))
    
    print(f"Found {len(valid_combinations)}/{total_combinations} input files")
    
    if not valid_combinations:
        print("No valid input files found. Exiting.")
        print("="*80)
        return {}
    
    # Process combinations
    results = {}
    successful = 0
    skipped = 0
    failed = 0
    
    if use_parallel and n_workers > 1:
        # ============================================================
        # PARALLEL PROCESSING
        # ============================================================
        print(f"\nStarting parallel processing with {n_workers} workers...")
        
        # Create partial function with fixed parameters
        worker_func = partial(
            _process_single_combination_worker,
            habitat_reference_df_path=habitat_reference_df_path,
            raster_stack_dir=raster_stack_dir,
            output_stability_dir=output_stability_dir,
            yearband_selection=yearband_selection,
            timeframe=timeframe,
            year_bands_to_include=year_bands_to_include,
            true_winner_min=true_winner_min,
            med_winner_min=med_winner_min,
            majority_min=majority_min,
            tie_min=tie_min,
            nodata_value=nodata_value,
            overwrite=overwrite,
            return_arrays=return_arrays,
        )
        
        # Process in parallel
        with Pool(processes=n_workers) as pool:
            # Use imap_unordered for better progress tracking
            for idx, (hab, attempt, result, status) in enumerate(
                pool.imap_unordered(worker_func, valid_combinations), 1
            ):
                print(f"\n[{idx}/{len(valid_combinations)}] {hab}_{attempt}: {status.upper()}")
                
                if status == 'success':
                    results[(hab, attempt)] = result
                    successful += 1
                elif status == 'skipped':
                    results[(hab, attempt)] = None
                    skipped += 1
                elif status == 'failed':
                    print(f"   ERROR: {result}")
                    results[(hab, attempt)] = None
                    failed += 1
        
        # Force garbage collection after all parallel workers complete
        gc.collect()
        
    else:
        # ============================================================
        # SEQUENTIAL PROCESSING (original behavior)
        # ============================================================
        for idx, (hab, attempt, raster_path) in enumerate(valid_combinations, 1):
            print(f"\n[{idx}/{len(valid_combinations)}] Processing {hab}_{attempt}...")
            
            try:
                layers = build_pixel_stability_stack(
                    carto_or_RF="carto_col",
                    hab_selection=hab,
                    train_split_attempt=attempt,
                    habitat_reference_df_path=habitat_reference_df_path,
                    raster_stack_dir=raster_stack_dir,
                    output_stability_dir=output_stability_dir,
                    yearband_selection=yearband_selection,
                    timeframe=timeframe,
                    year_bands_to_include=year_bands_to_include,
                    true_winner_min=true_winner_min,
                    med_winner_min=med_winner_min,
                    majority_min=majority_min,
                    tie_min=tie_min,
                    nodata_value=nodata_value,
                    overwrite=overwrite,
                )
                
                if layers is None:
                    # Skipped due to existing output
                    results[(hab, attempt)] = None
                    skipped += 1
                else:
                    # Success - build output path matching the naming convention
                    output_parts = ["carto_col_stability", hab, attempt]
                    if yearband_selection:
                        output_parts.append(yearband_selection)
                    if timeframe:
                        output_parts.append(timeframe)
                    
                    output_subdir = "__".join(output_parts)
                    output_path = output_stability_dir / output_subdir
                    
                    if return_arrays:
                        results[(hab, attempt)] = layers
                    else:
                        # Store only the output path
                        results[(hab, attempt)] = str(output_path)
                        del layers  # Free memory
                    
                    successful += 1
                
            except Exception as e:
                print(f"   FAILED: {e}")
                results[(hab, attempt)] = None
                failed += 1
            
            # Force garbage collection after each iteration
            gc.collect()
    
    # Add None for missing input files
    all_combinations = [(h, a) for h in hab_selections for a in train_split_attempts]
    valid_keys = {(h, a) for h, a, _ in valid_combinations}
    for combo in all_combinations:
        if combo not in valid_keys:
            results[combo] = None
    
    # Summary
    print("\n" + "="*80)
    print("BATCH PROCESSING COMPLETE")
    print("="*80)
    print(f"Total combinations: {total_combinations}")
    print(f"Input files found: {len(valid_combinations)}")
    print(f"Successfully processed: {successful}")
    print(f"Skipped (existing): {skipped}")
    print(f"Failed: {failed}")
    print(f"Missing input files: {total_combinations - len(valid_combinations)}")
    print("="*80)
    
    return results



################################################################################
## RF ENTROPY BATCH FUNC
################################################################################
import gc
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import Union, Optional, Dict, Tuple
from pathlib import Path
import numpy as np


def _process_single_rf_combination_worker(
    args: Tuple[str, str, Optional[str], str, Path],
    row_or_col: str,
    habitat_reference_df_path: Path,
    output_stability_dir: Path,
    yearband_selection: Optional[str],
    year_bands_to_include: Optional[list],
    true_winner_min: float,
    med_winner_min: float,
    majority_min: float,
    tie_min: float,
    nodata_value: Optional[float],
    overwrite: bool,
    return_arrays: bool,
) -> Tuple[str, str, Optional[str], str, Union[Dict, str, None], str]:
    """
    Worker function for parallel processing of a single RF combination.
    
    Returns
    -------
    tuple
        (hab, attempt, timeframe, band_sel, result, status)
        where status is 'success', 'skipped', or 'failed'
    """
    hab, attempt, timeframe, band_sel, raster_path = args
    
    try:
        layers = build_pixel_stability_stack(
            carto_or_RF=f"RF_{row_or_col}",
            hab_selection=hab,
            train_split_attempt=attempt,
            habitat_reference_df_path=habitat_reference_df_path,
            raster_stack_dir=raster_path.parent,  
            output_stability_dir=output_stability_dir,
            band_selection=band_sel,
            yearband_selection=yearband_selection,
            timeframe=timeframe,
            year_bands_to_include=year_bands_to_include,
            true_winner_min=true_winner_min,
            med_winner_min=med_winner_min,
            majority_min=majority_min,
            tie_min=tie_min,
            nodata_value=nodata_value,
            overwrite=overwrite,
        )
        
        if layers is None:
            # Skipped due to existing output
            gc.collect()
            return hab, attempt, timeframe, band_sel, None, 'skipped'
        else:
            # Success - build output path
            if return_arrays:
                result = layers
            else:
                # Build output name
                output_parts = [f"RF_{row_or_col}_stability", hab, attempt]
                if yearband_selection:
                    output_parts.append(yearband_selection)
                if timeframe:
                    output_parts.append(timeframe)
                output_parts.append(band_sel)
                output_subdir = "__".join(output_parts)
                
                output_path = output_stability_dir / output_subdir
                result = str(output_path)
                del layers  # Free memory
            
            # ✅ CRITICAL: Force garbage collection after processing
            gc.collect()
            
            return hab, attempt, timeframe, band_sel, result, 'success'
    
    except Exception as e:
        # ✅ Clean up memory even on failure
        gc.collect()
        return hab, attempt, timeframe, band_sel, str(e), 'failed'


def rf_pixel_stability_stack_batch(
    row_or_col: str,
    hab_selections: list[str],
    train_split_attempts: list[str],
    band_selections: list[str],
    habitat_reference_df_path: Union[str, Path],
    raster_stack_dir: Union[str, Path],
    output_stability_dir: Union[str, Path],
    *,
    yearband_selection: Optional[str] = None,
    timeframes: Optional[list[str]] = None,
    year_bands_to_include: Optional[list] = None,
    true_winner_min: float = 0.80,
    med_winner_min: float = 0.70,
    majority_min: float = 0.60,
    tie_min: float = 0.40,
    nodata_value: float | None = None,
    overwrite: bool = False,
    return_arrays: bool = False,
    n_workers: Optional[int] = None,
    use_parallel: bool = True,
) -> Dict[tuple, Dict[str, np.ndarray] | str | None]:
    """
    Process multiple habitat selections, training splits, band selections, and timeframes in batch for RF models.
    
    Parameters
    ----------
    row_or_col : str
        Row or column identifier for RF model ('row' or 'col').
    hab_selections : list of str
        List of habitat selections to process (e.g., ['WD1', 'WD2'])
    train_split_attempts : list of str
        List of training split attempts to process (e.g., ['at1', 'at2'])
    band_selections : list of str
        List of band selections to process (e.g., ['b28ndvwi', 'b15', 'b30'])
    habitat_reference_df_path : str or Path
        Path to habitat reference dataframe
    raster_stack_dir : str or Path
        Directory containing raster stacks (organized in subdirectories)
    output_stability_dir : str or Path
        Output directory for pixel stability layers
    yearband_selection : str, optional
        Single yearband descriptor for output naming (e.g., '2017_2024').
        This is just a label added to output directory names, not used for file search.
    timeframes : list of str, optional
        List of timeframe identifiers to process (e.g., ['Q1234', 'Q12']).
        If None, processes files without timeframe in the filename.
    year_bands_to_include : list of int, optional
        List of band indices (1-based) to include in the analysis.
    true_winner_min : float, default=0.80
        Minimum frequency (0-1) for true winner classification.
    med_winner_min : float, default=0.70
        Minimum frequency (0-1) for medium winner classification.
    majority_min : float, default=0.60
        Minimum frequency (0-1) for majority classification.
    tie_min : float, default=0.40
        Minimum frequency (0-1) for both classes in a tie.
    nodata_value : float or None, default=None
        NoData value to use for raster processing.
    overwrite : bool, default=False
        If False, skips processing for configurations where output files already exist.
    return_arrays : bool, default=False
        If False (default), returns output paths only (memory efficient).
        If True, returns numpy arrays for each layer (memory intensive).
    n_workers : int, optional
        Number of parallel workers. If None, uses (cpu_count() - 1).
        Set to 1 to disable parallel processing.
    use_parallel : bool, default=True
        If False, processes sequentially (useful for debugging).
    
    Returns
    -------
    dict
        Dictionary mapping (hab_selection, train_split_attempt, timeframe, band_selection) tuples to either:
        - Layer dicts containing numpy arrays (if return_arrays=True)
        - Output directory paths as strings (if return_arrays=False)
        - None for configurations that were skipped or failed.
        If timeframes is None, timeframe in tuple is None.
    
    Notes
    -----
    Setting return_arrays=False is recommended for batch processing many large
    rasters to avoid memory issues. The raster files are saved to disk regardless
    of this setting.
    
    Parallel processing spawns separate processes, each with their own memory space.
    Memory is automatically cleaned up after each worker completes.
    
    Expected directory structure:
        raster_stack_dir/
            {band_selection}_stacked/
                {hab_selection}/
                    stack_{band_selection}__{hab_selection}_{train_split_attempt}_{timeframe}__rstr.tif
    
    Expected filename patterns:
        - With timeframe: stack_{band}__{hab}_{attempt}_{timeframe}__rstr.tif
        - Without timeframe: stack_{band}__{hab}_{attempt}__rstr.tif
    """
    
    # Convert to Path objects
    raster_stack_dir = Path(raster_stack_dir)
    output_stability_dir = Path(output_stability_dir)
    habitat_reference_df_path = Path(habitat_reference_df_path)
    
    # Handle timeframes - if None, use [None] to process once without timeframe
    timeframes_list = timeframes if timeframes is not None else [None]
    
    total_combinations = (len(hab_selections) * len(train_split_attempts) * 
                         len(timeframes_list) * len(band_selections))
    
    # Determine number of workers
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)  # Leave one core free
    
    print("="*80)
    print(f"BATCH PROCESSING: RF PIXEL STABILITY STACK ({total_combinations} combinations)")
    print(f"Row/Col: {row_or_col}")
    print(f"Band selections: {band_selections}")
    if use_parallel and n_workers > 1:
        print(f"Parallel processing: {n_workers} workers")
    else:
        print("Sequential processing")
    if yearband_selection:
        print(f"Yearband descriptor: {yearband_selection}")
    if timeframes:
        print(f"Timeframes: {timeframes}")
    print("="*80)
    
    # Find valid input files
    valid_combinations = []
    print("\n" + "="*80)
    print("SEARCHING FOR INPUT FILES")
    print("="*80)
    
    for band_sel in band_selections:
        for hab in hab_selections:
            for attempt in train_split_attempts:
                for timeframe in timeframes_list:
                    # Construct path: raster_stack_dir/{band_selection}_stacked/{hab_selection}/
                    search_dir = raster_stack_dir / f"{band_sel}_stacked" / hab
                    
                    # Construct filename based on whether timeframe is specified
                    if timeframe is not None:
                        raster_filename = f"stack_{band_sel}__{hab}_{attempt}_{timeframe}__rstr.tif"
                    else:
                        raster_filename = f"stack_{band_sel}__{hab}_{attempt}__rstr.tif"
                    
                    raster_path = search_dir / raster_filename
                    
                    if raster_path.exists():
                        valid_combinations.append((hab, attempt, timeframe, band_sel, raster_path))
    
    print("="*80)
    print(f"Found {len(valid_combinations)}/{total_combinations} input files")
    print("="*80 + "\n")
    
    if not valid_combinations:
        print("No valid input files found. Exiting.")
        print("="*80)
        return {}
    
    # Process combinations
    results = {}
    successful = 0
    skipped = 0
    failed = 0
    
    if use_parallel and n_workers > 1:
        # ============================================================
        # PARALLEL PROCESSING
        # ============================================================
        print(f"\nStarting parallel processing with {n_workers} workers...")
        
        # Create partial function with fixed parameters
        worker_func = partial(
            _process_single_rf_combination_worker,
            row_or_col=row_or_col,
            habitat_reference_df_path=habitat_reference_df_path,
            output_stability_dir=output_stability_dir,
            yearband_selection=yearband_selection,
            year_bands_to_include=year_bands_to_include,
            true_winner_min=true_winner_min,
            med_winner_min=med_winner_min,
            majority_min=majority_min,
            tie_min=tie_min,
            nodata_value=nodata_value,
            overwrite=overwrite,
            return_arrays=return_arrays,
        )
        
        # Process in parallel
        with Pool(processes=n_workers) as pool:
            # Use imap_unordered for better progress tracking
            for idx, (hab, attempt, timeframe, band_sel, result, status) in enumerate(
                pool.imap_unordered(worker_func, valid_combinations), 1
            ):
                timeframe_str = f"_{timeframe}" if timeframe else ""
                yearband_str = f"_{yearband_selection}" if yearband_selection else ""
                print(f"\n[{idx}/{len(valid_combinations)}] {hab}_{attempt}{timeframe_str}{yearband_str}_{band_sel}: {status.upper()}")
                
                if status == 'success':
                    results[(hab, attempt, timeframe, band_sel)] = result
                    successful += 1
                elif status == 'skipped':
                    results[(hab, attempt, timeframe, band_sel)] = None
                    skipped += 1
                elif status == 'failed':
                    print(f"   ERROR: {result}")
                    results[(hab, attempt, timeframe, band_sel)] = None
                    failed += 1
        
        # ✅ Force garbage collection after all parallel workers complete
        gc.collect()
        
    else:
        # ============================================================
        # SEQUENTIAL PROCESSING (original behavior)
        # ============================================================
        for idx, (hab, attempt, timeframe, band_sel, raster_path) in enumerate(valid_combinations, 1):
            timeframe_str = f"_{timeframe}" if timeframe else ""
            yearband_str = f"_{yearband_selection}" if yearband_selection else ""
            print(f"\n[{idx}/{len(valid_combinations)}] Processing {hab}_{attempt}{timeframe_str}{yearband_str}_{band_sel}...")
            
            try:
                layers = build_pixel_stability_stack(
                    carto_or_RF=f"RF_{row_or_col}",
                    hab_selection=hab,
                    train_split_attempt=attempt,
                    habitat_reference_df_path=habitat_reference_df_path,
                    raster_stack_dir=raster_path.parent,  
                    output_stability_dir=output_stability_dir,
                    band_selection=band_sel,
                    yearband_selection=yearband_selection,
                    timeframe=timeframe,
                    year_bands_to_include=year_bands_to_include,
                    true_winner_min=true_winner_min,
                    med_winner_min=med_winner_min,
                    majority_min=majority_min,
                    tie_min=tie_min,
                    nodata_value=nodata_value,
                    overwrite=overwrite,
                )
                
                if layers is None:
                    # Skipped due to existing output
                    results[(hab, attempt, timeframe, band_sel)] = None
                    skipped += 1
                else:
                    # Success
                    if return_arrays:
                        results[(hab, attempt, timeframe, band_sel)] = layers
                    else:
                        # Store only the output path
                        output_parts = [f"RF_{row_or_col}_stability", hab, attempt]
                        if yearband_selection:
                            output_parts.append(yearband_selection)
                        if timeframe:
                            output_parts.append(timeframe)
                        output_parts.append(band_sel)
                        output_subdir = "__".join(output_parts)
                        
                        output_path = output_stability_dir / output_subdir
                        results[(hab, attempt, timeframe, band_sel)] = str(output_path)
                        del layers  # Free memory
                    
                    successful += 1
                
            except Exception as e:
                print(f"   FAILED: {e}")
                results[(hab, attempt, timeframe, band_sel)] = None
                failed += 1
            
            # ✅ Force garbage collection after each iteration
            gc.collect()
    
    # Add None for missing input files
    all_combinations = [(h, a, t, b) for h in hab_selections 
                       for a in train_split_attempts 
                       for t in timeframes_list 
                       for b in band_selections]
    valid_keys = {(h, a, t, b) for h, a, t, b, _ in valid_combinations}
    for combo in all_combinations:
        if combo not in valid_keys:
            results[combo] = None
    
    # Summary
    print("\n" + "="*80)
    print("BATCH PROCESSING COMPLETE")
    print("="*80)
    print(f"Total combinations: {total_combinations}")
    print(f"Input files found: {len(valid_combinations)}")
    print(f"Successfully processed: {successful}")
    print(f"Skipped (existing): {skipped}")
    print(f"Failed: {failed}")
    print(f"Missing input files: {total_combinations - len(valid_combinations)}")
    print("="*80)
    
    return results



################################################################################
## ADD RASTER INFORMATION
################################################################################
from functions.custom_common_nb_funcs import (
    extract_selection_colors,
)

from pathlib import Path

# READ HAB SELECTION FROM FOLDER NAME
def extract_hab_selection(folder_name, valid_hab_selections):
    """Extract hab_selection from folder name split by '__'"""
    parts = folder_name.split('__')
    
    for part in parts:
        if part in valid_hab_selections:
            return part
    
    return None

# CHECK CLASS MAP
import rasterio
import json

def check_and_add_class_map(raster_path, hab_selection, classes_overview):
    """
    Check if CLASS-MAP exists in raster metadata.
    If not, add it based on hab_selection.
    
    Parameters:
    - raster_path: Path to the raster file
    - hab_selection: str (e.g., 'WD1', 'WD2', 's1')
    - classes_overview: dict mapping hab_selection to [class_map, metadata]
    """
    
    # Open raster and read metadata
    with rasterio.open(raster_path, 'r') as src:
        metadata = src.tags()
    
    # Check if 'CLASS-MAP' exists in metadata
    if 'CLASS-MAP' in metadata:
        print(f"  CLASS-MAP present in {raster_path.parent.name}/{raster_path.name}")
        return  # Already exists, nothing to do
    
    # CLASS-MAP not found → need to add it
    
    # Get class mapping for this hab_selection
    hab_data = classes_overview.get(hab_selection)
    
    if hab_data is None:
        print(f"  Warning: No class mapping found for hab_selection '{hab_selection}'")
        return
    
    # Extract the class map (first element of the list)
    class_map = hab_data[0]
    
    # Convert class_map dict to JSON string for metadata
    class_map_json = json.dumps(class_map)
    
    # Write CLASS-MAP to raster metadata
    with rasterio.open(raster_path, 'r+') as src:
        src.update_tags(**{'CLASS-MAP': class_map_json})
    
    print(f"  CLASS-MAP added to {raster_path.parent.name}/{raster_path.name}")

# ADD PERCENTAGES FOR DECISION CATEGORIES
import numpy as np

def add_value_percentages(raster_path):
    """
    Calculate percentages of each unique value in the raster
    and add to metadata.
    
    Parameters:
    - raster_path: Path to the raster file
    """
    
    # Open raster and read data
    with rasterio.open(raster_path, 'r') as src:
        data = src.read(1)  # Read first band
        metadata = src.tags()
        nodata = src.nodata
    
    # Check if percentages already exist
    if 'VALUE-PERCENTAGES' in metadata:
        print(f"  VALUE-PERCENTAGES already present in {raster_path.parent.name}/{raster_path.name}")
        return
    
    # Mask nodata values if they exist
    if nodata is not None:
        valid_data = data[data != nodata]
    else:
        valid_data = data.flatten()
    
    # Calculate total valid pixels
    total_pixels = valid_data.size
    
    # Get unique values and their counts
    unique_values, counts = np.unique(valid_data, return_counts=True)
    
    # Calculate percentages
    percentages = {}
    for value, count in zip(unique_values, counts):
        percentage = (count / total_pixels) * 100
        percentages[str(int(value))] = round(percentage, 2)
    
    # Convert to JSON string
    percentages_json = json.dumps(percentages)
    
    # Write to metadata
    with rasterio.open(raster_path, 'r+') as src:
        src.update_tags(**{'VALUE-PERCENTAGES': percentages_json})
    
    print(f"  VALUE-PERCENTAGES added to {raster_path.parent.name}/{raster_path.name}")
    print(f"    → {percentages}")

# CREATE UNSTABLE PIXELS RASTER
import pandas as pd
from matplotlib.colors import hex2color

DISTINCT_COLORS = [
    "#800000", 
    "#9A6324", 
    "#808000",  
    "#469990",  
    "#000075",
    "#000000", 
    "#e61948", 
    "#f58231",  
    "#ffe119", 
    "#bfef45",  
    "#3cb44b",  
    "#42d4f4", 
    "#4363d8", 
    "#911eb4",  
    "#f032e6",  
    "#a9a9a9",  
    "#fabed4",
    "#e7a86a", 
    "#ebe179",  
    "#aaffc3",
    "#dcbeff",  
    "#ffffff", 
]

def get_distinct_combination_color(val1, val2):
    """Get a distinct color for a combination using deterministic hash"""
    import hashlib
    
    sorted_combo = tuple(sorted([val1, val2]))
    hash_val = int(hashlib.md5(str(sorted_combo).encode()).hexdigest()[:8], 16)
    return DISTINCT_COLORS[hash_val % len(DISTINCT_COLORS)]


def create_unstable_pixels_raster(subdir, hab_selection, classes_overview, extract_selection_colors):
    """
    Create a 3-band RGB raster for unstable pixels (decision_category = 4 or 5)
    
    Parameters:
    - subdir: Path to subdirectory
    - hab_selection: str (e.g., 'WD1', 'WD2', 's1')
    - classes_overview: dict mapping hab_selection to [class_map, metadata]
    - extract_selection_colors: function to extract color mapping
    """
    
    decision_path = subdir / "decision_category.tif"
    modal_path = subdir / "modal_class.tif"
    second_path = subdir / "second_class.tif"
    
    # Check if all required files exist
    if not all([decision_path.exists(), modal_path.exists(), second_path.exists()]):
        print(f"  Warning: Missing required files for unstable pixel analysis")
        return
    
    # Get class map and df path
    hab_data = classes_overview.get(hab_selection)
    if hab_data is None:
        print(f"  Warning: No data found for hab_selection '{hab_selection}'")
        return
    
    class_map = hab_data[0]
    df_info = hab_data[1]["df"]
    
    # Load reference dataframe - the df_info might already be a dataframe or a path
    if isinstance(df_info, pd.DataFrame):
        habitat_reference_df = df_info
    elif isinstance(df_info, (str, Path)):
        # Convert to string if it's a Path object
        habitat_reference_df = pd.read_csv(str(df_info))
    else:
        # It might be a method or callable that returns the dataframe
        try:
            habitat_reference_df = df_info() if callable(df_info) else df_info
        except:
            print(f"  Error: Could not load habitat reference dataframe")
            return
    
    # Extract colors - CORRECT ORDER: (df, hab_selection)
    color_df = extract_selection_colors(habitat_reference_df, hab_selection)
    
    # Create color lookup from class value to hex color
    # Reverse the class_map to get {type_name: class_value}
    type_to_value = {v: k for k, v in class_map.items()}
    
    # Create {class_value: hex_color}
    value_to_color = {}
    for idx, row in color_df.iterrows():
        type_name = row['type']
        if type_name in type_to_value:
            value_to_color[type_to_value[type_name]] = row['color']
    
    # Open rasters
    with rasterio.open(decision_path) as decision_src:
        decision = decision_src.read(1)
        profile = decision_src.profile.copy()
    
    with rasterio.open(modal_path) as modal_src:
        modal = modal_src.read(1)
    
    with rasterio.open(second_path) as second_src:
        second = second_src.read(1)
    
    # Initialize RGB bands
    height, width = decision.shape
    red = np.zeros((height, width), dtype=np.uint8)
    green = np.zeros((height, width), dtype=np.uint8)
    blue = np.zeros((height, width), dtype=np.uint8)
    
    # Track unique combinations for legend
    legend = {}
    
    # Process decision_category = 4 (unstable with two classes)
    mask_4 = (decision == 4)

    if np.any(mask_4):
        modal_vals = modal[mask_4]
        second_vals = second[mask_4]
        
        # Get unique combinations
        combinations = np.column_stack([modal_vals, second_vals])
        unique_combinations = np.unique(combinations, axis=0)
        
        for combo in unique_combinations:
            val1, val2 = int(combo[0]), int(combo[1])
            
            # Create sorted tuple for consistent color generation
            sorted_combo = tuple(sorted([val1, val2]))
            
            if sorted_combo not in legend:
                # Get DISTINCT color for this combination
                distinct_color_hex = get_distinct_combination_color(val1, val2)
                
                # Convert hex to RGB
                from matplotlib.colors import hex2color
                rgb_float = np.array(hex2color(distinct_color_hex))
                r, g, b = (rgb_float * 255).astype(np.uint8)
                r, g, b = int(r), int(g), int(b)
                
                # Store in legend with class names
                type1 = class_map.get(val1, f"Unknown_{val1}")
                type2 = class_map.get(val2, f"Unknown_{val2}")
                
                legend[sorted_combo] = {
                    'classes': f"{type1} / {type2}",
                    'color': distinct_color_hex,
                    'rgb': [r, g, b]  # Use list instead of tuple for JSON
                }
        
        # Apply colors to raster
        for combo, info in legend.items():
            val1, val2 = combo
            # Match both orderings
            mask = mask_4 & (
                ((modal == val1) & (second == val2)) |
                ((modal == val2) & (second == val1))
            )
            r, g, b = info['rgb']
            red[mask] = r
            green[mask] = g
            blue[mask] = b
    
    # Process decision_category = 5 (highly unstable - red)
    mask_5 = (decision == 5)
    if np.any(mask_5):
        red[mask_5] = 255
        green[mask_5] = 0
        blue[mask_5] = 0
        legend['highly_unstable'] = {
            'classes': 'Highly Unstable',
            'color': '#FF0000',
            'rgb': [255, 0, 0]  # Use list instead of tuple
        }
    
    # Update profile for 3-band RGB
    profile.update({
        'count': 3,
        'dtype': 'uint8'
    })
    
    # Write output raster
    output_path = subdir / "unstable_pixels.tif"
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(red, 1)
        dst.write(green, 2)
        dst.write(blue, 3)
        
        # Add legend to metadata - convert tuple keys to strings
        legend_serializable = {}
        for k, v in legend.items():
            key_str = k if isinstance(k, str) else f"{k[0]}_{k[1]}"
            legend_serializable[key_str] = v
        
        legend_json = json.dumps(legend_serializable)
        dst.update_tags(**{'LEGEND': legend_json})
    
    print(f"  unstable_pixels.tif created with {len(legend)} legend entries")
    
    return output_path

# MAIN FUNCTION
def process_rasters_with_class_map(main_dir, classes_overview, extract_selection_colors):
    """
    Loop through all subdirectories, extract hab_selection,
    and ensure CLASS-MAP metadata exists in modal_class.tif and second_class.tif,
    VALUE-PERCENTAGES in decision_category.tif, and create unstable_pixels.tif
    
    Parameters:
    - main_dir: Path or str to main directory containing subdirectories
    - classes_overview: dict mapping hab_selection to [class_map, metadata]
    - extract_selection_colors: function to extract color mapping
    """
    
    # Convert to Path object if it's a string
    main_dir = Path(main_dir)
    
    # Extract valid_hab_selections from classes_overview keys
    valid_hab_selections = list(classes_overview.keys())
    
    # Get all subdirectories
    subdirs = [path for path in main_dir.iterdir() if path.is_dir()]
    
    for subdir in subdirs:
        print(f"\nProcessing: {subdir.name}")
        
        # Step 1: Extract hab_selection
        hab_selection = extract_hab_selection(subdir.name, valid_hab_selections)
        
        if hab_selection is None:
            print(f"  Warning: Could not extract hab_selection from {subdir.name}")
            continue
        
        print(f"  → hab_selection: {hab_selection}")
        
        # Step 2: Process modal_class.tif
        modal_class_path = subdir / "modal_class.tif"
        if modal_class_path.exists():
            check_and_add_class_map(modal_class_path, hab_selection, classes_overview)
        else:
            print(f"  Warning: modal_class.tif not found in {subdir.name}")
        
        # Step 3: Process second_class.tif
        second_class_path = subdir / "second_class.tif"
        if second_class_path.exists():
            check_and_add_class_map(second_class_path, hab_selection, classes_overview)
        else:
            print(f"  Warning: second_class.tif not found in {subdir.name}")
        
        # Step 4: Process decision_category.tif
        decision_category_path = subdir / "decision_category.tif"
        if decision_category_path.exists():
            add_value_percentages(decision_category_path)
        else:
            print(f"  Warning: decision_category.tif not found in {subdir.name}")
        
        # Step 5: Create unstable_pixels.tif
        create_unstable_pixels_raster(subdir, hab_selection, classes_overview, extract_selection_colors)