################################################################################
#                          CALCULATE CLASS PERCENTAGES                         #
################################################################################
import numpy as np
import pandas as pd
import rasterio
import json
from rasterio.mask import mask

def calculate_carto_class_percentages(raster_stack_path, year_bands, locations_df=None, location_col='Naam'):
    """
    Calculate percentage representation of each class per year from a raster stack.
    """
    
    def process_bands(band_data_list, year_bands, class_map, nodata_value):
        """Helper function to process band data and return DataFrame"""
        results = []
        
        for idx, year in enumerate(year_bands):
            band_data = band_data_list[idx]
            
            # Flatten the array and remove nodata values
            flat_data = band_data.flatten()
            
            # Remove nodata values
            if nodata_value is not None:
                flat_data = flat_data[flat_data != nodata_value]
            
            # Remove any remaining NaN or inf values
            flat_data = flat_data[np.isfinite(flat_data)]
            
            if len(flat_data) == 0:
                year_data = {'Year': year}
                for class_val, class_name in class_map.items():
                    year_data[class_name] = 0.0
                results.append(year_data)
                continue
            
            # Count pixels per class
            unique, counts = np.unique(flat_data, return_counts=True)
            total_pixels = counts.sum()
            
            # Create row for this year
            year_data = {'Year': year}
            
            # Calculate percentage for each class
            for class_val, class_name in class_map.items():
                class_val_int = int(class_val)
                if class_val_int in unique:
                    idx_pos = np.where(unique == class_val_int)[0][0]
                    percentage = counts[idx_pos] / total_pixels
                else:
                    percentage = 0.0
                
                year_data[class_name] = percentage
            
            results.append(year_data)
        
        df = pd.DataFrame(results)
        df.set_index('Year', inplace=True)
        return df
    
    with rasterio.open(raster_stack_path) as src:
        # Read CLASS-MAP from metadata
        class_map_str = src.tags().get('CLASS-MAP')
        
        if class_map_str is None:
            raise ValueError("CLASS-MAP not found in raster metadata")
        
        class_map = json.loads(class_map_str)
        nodata_value = src.nodata
              
        all_results = {}
        
        # 1. Process full raster
        print("Processing full raster...")
        full_raster_bands = []
        for idx in range(1, len(year_bands) + 1):
            band_data = src.read(idx)
            full_raster_bands.append(band_data)
        
        all_results['Veluwe'] = process_bands(full_raster_bands, year_bands, class_map, nodata_value)
        
        # 2. Process each location if provided
        if locations_df is not None:
            from shapely.geometry import mapping
            
            for idx, row in locations_df.iterrows():
                location_name = row[location_col]
                geometry = row['geometry']

                print(f"Processing location: {location_name}...")
                
                # Convert Shapely geometry to GeoJSON-like dict
                geom_dict = mapping(geometry)
                
                # Mask the raster with the geometry
                try:
                    location_bands = []
                    for band_idx in range(1, len(year_bands) + 1):
                        masked_data, masked_transform = mask(
                            src, 
                            [geom_dict],
                            crop=True,
                            indexes=band_idx,
                            filled=True,
                            all_touched=True,
                        )
                        
                        # masked_data has shape (1, height, width), extract the 2D array
                        location_bands.append(masked_data)
                    
                    result_df = process_bands(location_bands, year_bands, class_map, nodata_value)
                    all_results[location_name] = result_df
                    
                    # Print summary for verification
                    valid_pixels = (location_bands[0] != nodata_value).sum()
                    # print(f"  FINAL Valid pixels found: {valid_pixels}")
                    if valid_pixels > 0:
                        unique_classes = np.unique(location_bands[0][location_bands[0] != nodata_value])
                        # print(f"  FINAL Classes present: {unique_classes}")
                    
                except Exception as e:
                    print(f"Warning: Could not process location '{location_name}': {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            print("Processing complete.")
    return all_results



################################################################################
#                          CALCULATE CLASS PERCENTAGES                         #
################################################################################
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from matplotlib.legend_handler import HandlerPatch

# def plot_carto_stacked_area(df, title, habitat_reference_df, hab_selection='s1', figsize=(12, 6), 
#                       save_dir=None, location_name=None, overwrite=True):
#     """
#     Create a stacked area chart with colors matching habitat divisions
    
#     Parameters:
#     -----------
#     df : DataFrame
#         Time series data with years as index and habitat classes as columns
#     title : str
#         Plot title
#     habitat_reference_df : DataFrame
#         Reference dataframe with habitat divisions and colors
#     hab_selection : str
#         Habitat selection prefix (e.g., 's1', 'WD1', 'WD4')
#     figsize : tuple
#         Figure size as (width, height). Default is (12, 6)
#     save_dir : str or Path, optional
#         Directory to save the plot. If None, plot is not saved.
#     location_name : str, optional
#         Location name for filename. Required if save_dir is provided.
#     overwrite : bool
#         If True, overwrites existing file. If False, loads existing file if it exists.
#         Default is True.
#     """
#     # Check if file exists and handle overwrite logic
#     if save_dir is not None:
#         if location_name is None:
#             raise ValueError("location_name must be provided when save_dir is specified")
        
#         # Get start and end years from the dataframe index
#         start_year = df.index.min()
#         end_year = df.index.max()
        
#         # Create filename and path
#         filename = f"LCC_Carto__{hab_selection}__{location_name}__{start_year}_{end_year}.png"
#         save_path = Path(save_dir)
#         carto_path = save_path / "Carto"  # Define the Carto subdirectory
#         full_path = carto_path / filename
        
#         # If file exists and overwrite is False, load and return existing figure
#         if full_path.exists() and not overwrite:
#             print(f"Loading existing file: {full_path}")
#             fig = plt.figure(figsize=figsize)
#             img = plt.imread(full_path)
#             plt.imshow(img)
#             plt.axis('off')
#             return fig

#     # Apply custom sorting rules
#     priority_order = ['Open water', 'Wet Nature', 'Semi-Wet Nature']
    
#     # Separate columns into categories
#     priority_cols = [col for col in priority_order if col in df.columns]
#     remaining_col = ['Remaining'] if 'Remaining' in df.columns else []
#     other_cols = sorted([col for col in df.columns 
#                         if col not in priority_order and col != 'Remaining'])
    
#     # Combine in the desired order: priority -> alphabetical -> remaining
#     ordered_columns = priority_cols + other_cols + remaining_col
    
#     # Reorder dataframe
#     df = df[ordered_columns]

#     # Create new figure
#     fig, ax = plt.subplots(figsize=figsize)
    
#     # Get the column names for this habitat type
#     division_col = f'{hab_selection}_division'
#     color_col = f'{hab_selection}_color'
    
#     # Get unique divisions from the data (these are the classes we're plotting)
#     unique_divisions = df.columns.tolist()
    
#     # Filter reference df for only these divisions and create color map
#     color_mapping = habitat_reference_df.loc[
#         habitat_reference_df[division_col].isin(unique_divisions),
#         [division_col, color_col]
#     ]
#     colors = dict(zip(color_mapping[division_col], color_mapping[color_col]))
    
#     # Get colors in the same order as df columns
#     plot_colors = [colors.get(col, '#CCCCCC') for col in df.columns]
    
#     # Prepare data for stackplot
#     data_arrays = [df[col].values for col in df.columns]
    
#     # Prepare data for stackplot
#     data_arrays = [df[col].values for col in df.columns]

#     # Shift yearly values to the middle of the year
#     x = np.asarray(df.index, dtype=float) + 0.75

#     # Create stacked area plot
#     ax.stackplot(x, 
#                 *data_arrays,
#                 labels=df.columns,
#                 colors=plot_colors,
#                 alpha=0.8,
#     )

#     # Styling
#     ax.set_ylabel('Area proportion', fontsize=12)
#     ax.set_title(title, fontsize=14, fontweight='bold')
#     ax.set_ylim(0, 1)
#     ax.grid(True, alpha=1)

#     # Extend x-axis so last label includes the next year
#     start_year = int(df.index.min())
#     end_year = int(df.index.max())
#     ax.set_xlim(start_year, end_year + 1)
#     ax.set_xticks(range(start_year, end_year + 2))
    
#     # Create legend as tiles below x-axis
#     # Manual legend with dynamic spacing and square tiles
#     fig = plt.gcf()
#     renderer = fig.canvas.get_renderer()

#     tile_size_inches = 0.06
#     tile_y = -0.15  
#     text_offset = 0.005  

#     # Convert tile size to axes coordinates
#     fig_width, fig_height = fig.get_size_inches()
#     ax_bbox = ax.get_position()
#     ax_width_inches = fig_width * ax_bbox.width
#     ax_height_inches = fig_height * ax_bbox.height

#     tile_width_axes = tile_size_inches / ax_width_inches
#     tile_height_axes = tile_size_inches / ax_height_inches  # Different for square!

#     # Calculate positions and widths
#     items = []
#     for category, color in zip(df.columns, plot_colors):
#         # Measure text width
#         t = ax.text(0, 0, category, fontsize=7, transform=ax.transAxes)
#         bbox = t.get_window_extent(renderer=renderer)
#         t.remove()
        
#         text_width_axes = bbox.width / (fig_width * fig.dpi * ax_bbox.width)
#         items.append((category, color, text_width_axes))

#     # Calculate total width and starting position
#     total_width = sum(tile_width_axes + text_offset + width for _, _, width in items)
#     total_width += (len(items) - 1) * 0.001  # Small spacing between items

#     start_x = 0.5 - total_width / 2
#     current_x = start_x

#     for category, color, text_width in items:
#         # Draw square tile
#         tile_rect = Rectangle(
#             (current_x, tile_y),
#             tile_width_axes,
#             tile_height_axes,  # Now truly square in display
#             transform=ax.transAxes,
#             facecolor=color,
#             edgecolor='black',
#             linewidth=0.5,
#             zorder=11,
#             clip_on=False,
#         )
#         ax.add_patch(tile_rect)
        
#         # Add text label
#         ax.text(current_x + tile_width_axes + text_offset, 
#                 tile_y + tile_height_axes/2, 
#                 category, 
#                 transform=ax.transAxes,
#                 fontsize=7,
#                 verticalalignment='center',
#                 horizontalalignment='left',
#                 clip_on=False)
        
#         current_x += tile_width_axes + text_offset + text_width + 0.01

#     plt.tight_layout()
    
#     # Save plot if save_dir is provided
#     if save_dir is not None:
#         # Create save path and Carto subdirectory, ensure they exist
#         carto_path.mkdir(parents=True, exist_ok=True)
        
#         # Save the figure
#         fig.savefig(full_path, dpi=300, bbox_inches='tight')
#         print(f"Saved: {full_path}")
    
#     return fig


def plot_carto_stacked_area(df, title, habitat_reference_df, hab_selection='s1', figsize=(12, 6), 
                      save_dir=None, location_name=None, overwrite=True):
    """
    Annual land-cover composition with Q4 solid bars and faded interpolated
    Q1-Q3 transitions.

    Each measurement is shown as a solid stacked bar in Q4 (Oct-Dec) of its
    labelled year. The transition between consecutive measurements is shown
    as a faded, linearly interpolated stacked area filling Q1-Q3 of the
    intervening year. The faded styling makes the interpolation visually
    distinct from the measured values, avoiding the implication that the
    transition is itself measured.
    """
    # --- save/overwrite handling (unchanged) ---
    if save_dir is not None:
        if location_name is None:
            raise ValueError("location_name must be provided when save_dir is specified")

        start_year = df.index.min()
        end_year = df.index.max()

        filename = f"LCC_Carto__{hab_selection}__{location_name}__{start_year}_{end_year}.png"
        save_path = Path(save_dir)
        carto_path = save_path / "Carto"
        full_path = carto_path / filename

        if full_path.exists() and not overwrite:
            print(f"Loading existing file: {full_path}")
            fig = plt.figure(figsize=figsize)
            img = plt.imread(full_path)
            plt.imshow(img)
            plt.axis('off')
            return fig

    # --- column ordering (unchanged) ---
    priority_order = ['Open water', 'Wet Nature', 'Semi-Wet Nature']
    priority_cols = [col for col in priority_order if col in df.columns]
    remaining_col = ['Remaining'] if 'Remaining' in df.columns else []
    other_cols = sorted([col for col in df.columns
                        if col not in priority_order and col != 'Remaining'])
    ordered_columns = priority_cols + other_cols + remaining_col
    df = df[ordered_columns]

    # --- figure & colors ---
    fig, ax = plt.subplots(figsize=figsize)

    division_col = f'{hab_selection}_division'
    color_col = f'{hab_selection}_color'
    unique_divisions = df.columns.tolist()
    color_mapping = habitat_reference_df.loc[
        habitat_reference_df[division_col].isin(unique_divisions),
        [division_col, color_col]
    ]
    colors = dict(zip(color_mapping[division_col], color_mapping[color_col]))
    plot_colors = [colors.get(col, '#CCCCCC') for col in df.columns]

    years = df.index.astype(int).to_numpy()
    values = df.values  # shape (n_years, n_categories)

    # Compute cumulative tops for stacking (n_years, n_categories + 1)
    # row[i] = [0, cum1, cum2, ..., total] for year i
    cumulative = np.concatenate(
        [np.zeros((len(years), 1)), np.cumsum(values, axis=1)],
        axis=1,
    )

    # --- draw solid Q4 bars for each measurement year ---
    # Q4 spans Oct 1 (year + 0.75) to Dec 31 (year + 1.0 in x-axis terms)
    SOLID_ALPHA = 0.9
    FADED_ALPHA = 0.4

    for i, year in enumerate(years):
        x_left = year + 0.10
        x_right = year + 0.90
        x_pair = np.array([x_left, x_right])

        for cat_idx in range(len(df.columns)):
            y_bottom = np.full(2, cumulative[i, cat_idx])
            y_top = np.full(2, cumulative[i, cat_idx + 1])
            ax.fill_between(
                x_pair, y_bottom, y_top,
                facecolor=plot_colors[cat_idx],
                edgecolor='black',
                linewidth=0.4,
                alpha=SOLID_ALPHA,
                zorder=5,
            )

    # --- draw faded interpolated Q1-Q3 bars between consecutive years ---
    # Q1-Q3 of year N+1 spans Jan 1 (year N + 1.0) to Oct 1 (year N + 1.75)
    # Linear interpolation between measurement N and measurement N+1
    for i in range(len(years) - 1):
        x_left = years[i] + 0.90   # Jan 1 of year N+1
        x_right = years[i] + 1.10 # Oct 1 of year N+1
        x_pair = np.array([x_left, x_right])

        for cat_idx in range(len(df.columns)):
            # Linear interp: bottom and top of this category at left and right
            y_bottom_left = cumulative[i, cat_idx]
            y_bottom_right = cumulative[i + 1, cat_idx]
            y_top_left = cumulative[i, cat_idx + 1]
            y_top_right = cumulative[i + 1, cat_idx + 1]

            y_bottom = np.array([y_bottom_left, y_bottom_right])
            y_top = np.array([y_top_left, y_top_right])

            ax.fill_between(
                x_pair, y_bottom, y_top,
                facecolor=plot_colors[cat_idx],
                edgecolor='none',
                alpha=FADED_ALPHA,
                zorder=4,
            )

    # --- styling ---
    ax.set_ylabel('Area proportion', fontsize=12)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_ylim(0, 1)
    # Pad x-axis slightly so the first Q4 bar isn't flush with the left edge
    ax.set_xlim(years[0] - 1.0, years[-1] + 2)
    ax.set_xticks(np.arange(years[0] - 1, years[-1] + 3))
    ax.grid(True, axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    # --- legend tiles below the axis (unchanged) ---
    renderer = fig.canvas.get_renderer()
    tile_size_inches = 0.06
    tile_y = -0.15
    text_offset = 0.005

    fig_width, fig_height = fig.get_size_inches()
    ax_bbox = ax.get_position()
    ax_width_inches = fig_width * ax_bbox.width
    ax_height_inches = fig_height * ax_bbox.height

    tile_width_axes = tile_size_inches / ax_width_inches
    tile_height_axes = tile_size_inches / ax_height_inches

    items = []
    for category, color in zip(df.columns, plot_colors):
        t = ax.text(0, 0, category, fontsize=12, transform=ax.transAxes)
        bbox = t.get_window_extent(renderer=renderer)
        t.remove()
        text_width_axes = bbox.width / (fig_width * fig.dpi * ax_bbox.width)
        items.append((category, color, text_width_axes))

    total_width = sum(tile_width_axes + text_offset + width for _, _, width in items)
    total_width += (len(items) - 1) * 0.001
    start_x = 0.5 - total_width / 2
    current_x = start_x

    for category, color, text_width in items:
        tile_rect = Rectangle(
            (current_x, tile_y),
            tile_width_axes,
            tile_height_axes,
            transform=ax.transAxes,
            facecolor=color,
            edgecolor='black',
            linewidth=0.5,
            zorder=11,
            clip_on=False,
        )
        ax.add_patch(tile_rect)

        ax.text(current_x + tile_width_axes + text_offset,
                tile_y + tile_height_axes / 2,
                category,
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment='center',
                horizontalalignment='left',
                clip_on=False)

        current_x += tile_width_axes + text_offset + text_width + 0.01

    plt.tight_layout()

    if save_dir is not None:
        carto_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {full_path}")

    return fig


################################################################################
#                                 WORKFLOW CARTO                               #
################################################################################
def carto_area_plot_over_time_workflow(hab_div, hab_selection, all_years, veluwe_plot_windows__gdf, save_dir, overwrite=False):
    """
    Create stacked area plots showing land cover change over time for a habitat selection.
    
    Parameters
    ----------
    hab_div : dict
        Dictionary containing habitat data with structure:
        {hab_key: [hab_name, raster_path, reference_df], ...}
    hab_selection : str
        Key to select from hab_div (e.g., 's1', 'WD1')
    all_years : list
        List of years to analyze
    veluwe_plot_windows__gdf : GeoDataFrame
        GeoDataFrame containing plot window geometries
    save_dir : str or Path
        Directory to save the output plots
    overwrite : bool, default=True
        Whether to overwrite existing files
    
    Returns
    -------
    dict
        Dictionary of matplotlib figures, keyed by location name
    
    Example
    -------
    >>> figures = carto_area_plot_over_time_workflow(
    ...     hab_div=hab_div,
    ...     hab_selection='s1',
    ...     all_years=[2020, 2021, 2022],
    ...     veluwe_plot_windows__gdf=plot_windows_gdf,
    ...     save_dir=path/to/dir,
    ... )
    """

    # Get selection from hab_div
    if hab_selection not in hab_div:
        raise ValueError(f"hab_selection '{hab_selection}' not found in hab_div. Available: {list(hab_div.keys())}")
    
    selection = hab_div[hab_selection]
    hab_name = selection[0]
    raster_path = selection[1]
    reference_df = selection[2]
    
    # Calculate percentages for all locations
    percentage_div_dict = calculate_carto_class_percentages(
        raster_path, 
        all_years, 
        veluwe_plot_windows__gdf
    )
    
    # Create plots for each location
    figures = {}
    for location_name, df in percentage_div_dict.items():
        fig = plot_carto_stacked_area(
            df=df,
            title=f'Land Cover Change: {hab_name} - {location_name}',
            habitat_reference_df=reference_df,
            hab_selection=hab_name,
            figsize=(12, 4),
            save_dir=save_dir,
            location_name=location_name,
            overwrite=overwrite,
        )
        figures[location_name] = fig
        plt.show()
    
    return figures


################################################################################
#                              RF QUARTERLY PLOTS                              #
################################################################################
# def plot_quarterly_stacked_area(df, title, habitat_reference_df, hab_selection, 
#                                band_selection, save_dir=None, location_name=None, 
#                                figsize=(12, 4), overwrite=True):
#     """
#     Create stacked area chart with quarterly data (proportions 0-1).
    
#     Parameters
#     ----------
#     df : DataFrame
#         Time series data with fractional years as index and habitat classes as columns
#     title : str
#         Plot title
#     habitat_reference_df : DataFrame
#         Reference dataframe with habitat divisions and colors
#     hab_selection : str
#         Habitat selection prefix (e.g., 's1', 'WD1')
#     band_selection : str
#         Band selection name
#     save_dir : str or Path, optional
#         Directory to save the plot
#     location_name : str, optional
#         Location name for filename
#     figsize : tuple
#         Figure size
#     overwrite : bool
#         Whether to overwrite existing files
    
#     Returns
#     -------
#     matplotlib.figure.Figure
#     """
#     from pathlib import Path
#     from matplotlib.patches import Rectangle
#     import matplotlib.ticker as ticker
#     import matplotlib.pyplot as plt
    
#     # Check if file exists
#     if save_dir is not None:
#         if location_name is None:
#             raise ValueError("location_name must be provided when save_dir is specified")
        
#         start_year = int(df.index.min())
#         end_year = int(df.index.max())
        
#         filename = f"LCC_RF__{band_selection}__{hab_selection}__{location_name}__{start_year}_{end_year}_quarterly.png"
#         save_path = Path(save_dir)
#         full_path = save_path / filename
        
#         if full_path.exists() and not overwrite:
#             print(f"Loading existing file: {full_path}")
#             fig = plt.figure(figsize=figsize)
#             img = plt.imread(full_path)
#             plt.imshow(img)
#             plt.axis('off')
#             return fig
    
#     # Apply custom sorting rules
#     priority_order = ['Open water', 'Wet Nature', 'Semi-Wet Nature']
    
#     # Separate columns into categories
#     priority_cols = [col for col in priority_order if col in df.columns]
#     remaining_col = ['Remaining'] if 'Remaining' in df.columns else []
#     other_cols = sorted([col for col in df.columns 
#                         if col not in priority_order and col != 'Remaining'])
    
#     # Combine in the desired order: priority -> alphabetical -> remaining
#     ordered_columns = priority_cols + other_cols + remaining_col
    
#     # Reorder dataframe
#     df = df[ordered_columns]

#     # Create new figure
#     fig, ax = plt.subplots(figsize=figsize)
    
#     # Get color mapping
#     division_col = f'{hab_selection}_division'
#     color_col = f'{hab_selection}_color'
    
#     color_mapping = habitat_reference_df.loc[
#         habitat_reference_df[division_col].isin(df.columns),
#         [division_col, color_col]
#     ]
#     colors = dict(zip(color_mapping[division_col], color_mapping[color_col]))
#     plot_colors = [colors.get(col, '#CCCCCC') for col in df.columns]
    
#     # Create stacked area plot
#     data_arrays = [df[col].values for col in df.columns]
#     ax.stackplot(df.index, 
#                  *data_arrays,
#                  labels=df.columns,
#                  colors=plot_colors,
#                  alpha=0.8)
    
#     # Styling
#     ax.set_ylabel('Area Proportion', fontsize=12)
#     ax.set_title(title, fontsize=14, fontweight='bold')
#     ax.set_ylim(0, 1)
#     # ax.spines['top'].set_visible(False)
#     # ax.spines['right'].set_visible(False)
    
#     # Set x-axis ticks
#     # Major ticks: integer years
#     ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    
#     # Minor ticks: quarters
#     start_year = int(df.index.min())
#     end_year = int(df.index.max()) + 1
    
#     quarter_positions = []
#     for year in range(start_year, end_year):
#         for q in [0.125, 0.375, 0.625, 0.875]:
#             quarter_positions.append(year + q)
    
#     ax.set_xticks(quarter_positions, minor=True)
#     ax.tick_params(axis='x', which='minor', length=4, width=0.8, color='gray')
#     ax.tick_params(axis='x', which='major', length=6, width=1.2)
    
#     # Grid settings - major grid only for y-axis
#     ax.grid(True, which='major', axis='y', alpha=0.3, linestyle='-', linewidth=0.8)
    
#     # Draw vertical lines at quarters OVER the plot with higher zorder
#     for qpos in quarter_positions:
#         ax.axvline(qpos, color='gray', linestyle=':', linewidth=0.8, alpha=0.4, zorder=10)

#     # Manual legend with dynamic spacing and square tiles
#     renderer = fig.canvas.get_renderer()
#     tile_size_inches = 0.06
#     tile_y = -0.15
#     text_offset = 0.005
    
#     fig_width, fig_height = fig.get_size_inches()
#     ax_bbox = ax.get_position()
#     ax_width_inches = fig_width * ax_bbox.width
#     ax_height_inches = fig_height * ax_bbox.height
    
#     tile_width_axes = tile_size_inches / ax_width_inches
#     tile_height_axes = tile_size_inches / ax_height_inches
    
#     # Calculate positions and widths
#     items = []
#     for category, color in zip(df.columns, plot_colors):
#         t = ax.text(0, 0, category, fontsize=7, transform=ax.transAxes)
#         bbox = t.get_window_extent(renderer=renderer)
#         t.remove()
#         text_width_axes = bbox.width / (fig_width * fig.dpi * ax_bbox.width)
#         items.append((category, color, text_width_axes))
    
#     total_width = sum(tile_width_axes + text_offset + width for _, _, width in items)
#     total_width += (len(items) - 1) * 0.001
    
#     start_x = 0.5 - total_width / 2
#     current_x = start_x
    
#     for category, color, text_width in items:
#         tile_rect = Rectangle(
#             (current_x, tile_y),
#             tile_width_axes,
#             tile_height_axes,
#             transform=ax.transAxes,
#             facecolor=color,
#             edgecolor='black',
#             linewidth=0.5,
#             zorder=11,
#             clip_on=False,
#         )
#         ax.add_patch(tile_rect)
        
#         ax.text(current_x + tile_width_axes + text_offset, 
#                 tile_y + tile_height_axes/2, 
#                 category, 
#                 transform=ax.transAxes,
#                 fontsize=7,
#                 verticalalignment='center',
#                 horizontalalignment='left',
#                 clip_on=False)
        
#         current_x += tile_width_axes + text_offset + text_width + 0.01
    
#     plt.tight_layout()
    
#     # Save plot
#     if save_dir is not None:
#         save_path.mkdir(parents=True, exist_ok=True)
#         fig.savefig(full_path, dpi=300, bbox_inches='tight')
#         print(f"Saved: {full_path}")
    
#     return fig

def plot_quarterly_stacked_area(df, title, habitat_reference_df, hab_selection,
                                band_selection, save_dir=None, location_name=None,
                                figsize=(12, 4), overwrite=True):
    """
    Stacked bar chart of quarterly land-cover composition.

    Each bar spans exactly one quarter (0.25 on the fractional-year axis) and
    represents the composition for that quarter. The index is interpreted as
    quarter-end times, so a bar at index N covers the period N-0.25 to N.

    Aesthetics match plot_carto_stacked_area (yearly) for direct visual
    comparison: same stack-bar treatment, same legend tile style, same y-grid.
    Black vertical lines mark year boundaries (the yearly chart's visual
    signature) without outlining every individual quarter-bar.
    """
    from pathlib import Path
    from matplotlib.patches import Rectangle
    import matplotlib.ticker as ticker
    import matplotlib.pyplot as plt
    import numpy as np

    # --- save/overwrite handling ---
    if save_dir is not None:
        if location_name is None:
            raise ValueError("location_name must be provided when save_dir is specified")

        start_year = int(df.index.min())
        end_year = int(df.index.max())

        filename = (f"LCC_RF__{band_selection}__{hab_selection}__{location_name}"
                    f"__{start_year}_{end_year}_quarterly_bars.png")
        save_path = Path(save_dir)
        full_path = save_path / filename

        if full_path.exists() and not overwrite:
            print(f"Loading existing file: {full_path}")
            fig = plt.figure(figsize=figsize)
            img = plt.imread(full_path)
            plt.imshow(img)
            plt.axis('off')
            return fig

    # --- column ordering (unchanged) ---
    priority_order = ['Open water', 'Wet Nature', 'Semi-Wet Nature']
    priority_cols = [col for col in priority_order if col in df.columns]
    remaining_col = ['Remaining'] if 'Remaining' in df.columns else []
    other_cols = sorted([col for col in df.columns
                        if col not in priority_order and col != 'Remaining'])
    ordered_columns = priority_cols + other_cols + remaining_col
    df = df[ordered_columns]

    # --- figure & colors ---
    fig, ax = plt.subplots(figsize=figsize)

    division_col = f'{hab_selection}_division'
    color_col = f'{hab_selection}_color'
    color_mapping = habitat_reference_df.loc[
        habitat_reference_df[division_col].isin(df.columns),
        [division_col, color_col]
    ]
    colors = dict(zip(color_mapping[division_col], color_mapping[color_col]))
    plot_colors = [colors.get(col, '#CCCCCC') for col in df.columns]

    # --- stacked bars, one per quarter ---
    # Index is at quarter ENDS: each bar at index N covers (N - 0.25, N].
    # Bar centers are at index - 0.125 (mid-quarter).
    QUARTER_WIDTH = 0.25
    index_arr = df.index.to_numpy(dtype=float)
    bar_centers = index_arr
    values = df.values
    bottoms = np.zeros(len(bar_centers))

    for cat_idx, category in enumerate(df.columns):
        heights = values[:, cat_idx]
        ax.bar(
            bar_centers,
            heights,
            width=QUARTER_WIDTH,
            bottom=bottoms,
            color=plot_colors[cat_idx],
            edgecolor='none',
            linewidth=0,
            align='center',
            label=category,
            alpha=0.85,
            zorder=5,
        )
        bottoms += heights

    # --- year-boundary lines (the visual rhyme with the yearly chart) ---
    # Drawn on top of bars with the same thin black look as outlines.
    start_year = int(np.floor(index_arr.min() - QUARTER_WIDTH / 2 - 1.0))
    end_year = int(np.ceil(index_arr.max() + QUARTER_WIDTH / 2 + 1.0))
    for year in range(start_year, end_year + 1):
        ax.axvline(year, color='black', linewidth=0.4, alpha=0.9, zorder=11)

    # --- styling (mirrors plot_quarterly_stacked_area) ---
    ax.set_ylabel('Area proportion', fontsize=12)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_ylim(0, 1)

    # Major ticks at integer years, minor ticks at every quarter boundary
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    quarter_boundaries = []
    for year in range(start_year, end_year):  
        for q in [0.0, 0.25, 0.50, 0.75]:
            quarter_boundaries.append(year + q)
    quarter_boundaries.append(end_year)  
    ax.set_xticks(quarter_boundaries, minor=True)

    # Apply x-limits LAST, after all tick/locator setup, to prevent autoscaling overrides
    ax.set_xlim(index_arr[0] - QUARTER_WIDTH / 2 - 1.0,
                index_arr[-1] + QUARTER_WIDTH / 2 + 1.0)

    ax.tick_params(axis='x', which='minor', length=4, width=0.8, color='gray')
    ax.tick_params(axis='x', which='major', length=6, width=1.2, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    # Y-grid only, matching the yearly chart
    ax.grid(True, which='major', axis='y', alpha=0.3, linestyle=':', linewidth=0.5)

    # Faint dotted lines at quarter boundaries (carried over from your existing function)
    for qpos in quarter_boundaries:
        if qpos != int(qpos):  # skip year boundaries (drawn separately as solid black)
            ax.axvline(qpos, color='gray', linestyle=':', linewidth=0.6,
                       alpha=0.4, zorder=10)

    # --- legend tiles below the axis (unchanged) ---
    renderer = fig.canvas.get_renderer()
    tile_size_inches = 0.06
    tile_y = -0.15
    text_offset = 0.005

    fig_width, fig_height = fig.get_size_inches()
    ax_bbox = ax.get_position()
    ax_width_inches = fig_width * ax_bbox.width
    ax_height_inches = fig_height * ax_bbox.height

    tile_width_axes = tile_size_inches / ax_width_inches
    tile_height_axes = tile_size_inches / ax_height_inches

    items = []
    for category, color in zip(df.columns, plot_colors):
        t = ax.text(0, 0, category, fontsize=12, transform=ax.transAxes)
        bbox = t.get_window_extent(renderer=renderer)
        t.remove()
        text_width_axes = bbox.width / (fig_width * fig.dpi * ax_bbox.width)
        items.append((category, color, text_width_axes))

    total_width = sum(tile_width_axes + text_offset + width for _, _, width in items)
    total_width += (len(items) - 1) * 0.001
    start_x = 0.5 - total_width / 2
    current_x = start_x

    for category, color, text_width in items:
        tile_rect = Rectangle(
            (current_x, tile_y),
            tile_width_axes,
            tile_height_axes,
            transform=ax.transAxes,
            facecolor=color,
            edgecolor='black',
            linewidth=0.5,
            zorder=11,
            clip_on=False,
        )
        ax.add_patch(tile_rect)

        ax.text(current_x + tile_width_axes + text_offset,
                tile_y + tile_height_axes / 2,
                category,
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment='center',
                horizontalalignment='left',
                clip_on=False)

        current_x += tile_width_axes + text_offset + text_width + 0.01

    plt.tight_layout()

    if save_dir is not None:
        save_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {full_path}")

    print(df.index[0], df.index[-1], len(df))

    return fig




def build_hab_div_dict(base_dir, band_selection, timeframes, habitat_reference_dfs):
    """
    Build hab_div dictionary from directory structure with quarterly stacks.
    
    Parameters
    ----------
    base_dir : str or Path
        Base directory containing band_selection folders
    band_selection : str
        Band selection folder name (e.g., 'b2348', 'b2ndvwi')
    timeframes : list
        List of timeframe identifiers (e.g., ['Q1', 'Q2', 'Q3', 'Q4'])
    habitat_reference_dfs : dict
        Dictionary mapping habitat groups to reference DataFrames
    
    Returns
    -------
    dict
        Dictionary structure: {hab_index: {timeframe: [hab_name, raster_path, reference_df]}}
    """
    from pathlib import Path
    import warnings
    
    base_path = Path(base_dir) / f"{band_selection}_stacked"
    
    # Define habitat selections
    hab_selections = ['s1', 's2', 'WD1', 'WD2', 'WD3', 'WD4', 'WD5', 'WD6', 'WDNF1', 'WDNF2', 'WDNF3', 'WDNF4', 'WDNF5', 'WDNF6']
    
    # Build nested dictionary
    hab_div_nested = {}
    
    for idx, hab_sel in enumerate(hab_selections):
        hab_div_nested[idx] = {}
        
        # Determine which reference df to use
        if hab_sel.startswith('s'):
            ref_df = habitat_reference_dfs['s'][
                [f'{hab_sel}_division', f'{hab_sel}_color']
            ]
        elif hab_sel.startswith('WDNF'):
            ref_df = habitat_reference_dfs['WDNF'][
                [f'{hab_sel}_division', f'{hab_sel}_color']
            ]
        else:  # WD habitats
            ref_df = habitat_reference_dfs['WD'][
                [f'{hab_sel}_division', f'{hab_sel}_color']
            ]
        
        for timeframe in timeframes:
            # Construct file path
            file_pattern = f"stack_{band_selection}__{hab_sel}_at1_{timeframe}__rstr.tif"
            raster_path = base_path / hab_sel / file_pattern
            
            # Check if file exists
            if not raster_path.exists():
                warnings.warn(f"File not found: {raster_path}", UserWarning)
                continue  # Skip this timeframe
            
            # Add to dictionary only if file exists
            hab_div_nested[idx][timeframe] = [hab_sel, str(raster_path), ref_df]
    
    return hab_div_nested


def calculate_quarterly_class_proportions(hab_div_nested, timeframes, all_years, veluwe_plot_windows__gdf):
    """
    Calculate class proportions for all quarters and combine into single timeline.
    
    Parameters
    ----------
    hab_div_nested : dict
        Dictionary: {hab_index: {timeframe: [hab_name, raster_path, reference_df]}}
    timeframes : list
        List of quarters (e.g., ['Q1', 'Q2', 'Q3', 'Q4'])
    all_years : list
        List of years
    veluwe_plot_windows__gdf : GeoDataFrame
        GeoDataFrame containing plot window geometries
    
    Returns
    -------
    dict
        Dictionary of combined DataFrames keyed by location with quarterly index
    """
    import pandas as pd
    import warnings
    
    # Get habitat info from first available timeframe
    first_hab = list(hab_div_nested.values())[0]
    
    # Check if habitat has any timeframes
    if not first_hab:
        warnings.warn("No valid raster files found for this habitat", UserWarning)
        return {}
    
    # Calculate proportions for all timeframes
    proportion_dfs_by_timeframe = {}
    
    for timeframe in timeframes:
        # Check if this timeframe exists in the dictionary
        if timeframe not in first_hab:
            warnings.warn(f"Timeframe {timeframe} not found, skipping...", UserWarning)
            continue
            
        print(f"  Processing {timeframe}...")
        raster_path = first_hab[timeframe][1]
        
        # Calculate proportions for this quarter
        try:
            proportion_dict = calculate_carto_class_percentages(
                raster_path, all_years, veluwe_plot_windows__gdf
            )
            proportion_dfs_by_timeframe[timeframe] = proportion_dict
        except Exception as e:
            warnings.warn(f"Error processing {timeframe}: {e}", UserWarning)
            continue
    
    # Check if we have any valid data
    if not proportion_dfs_by_timeframe:
        warnings.warn("No valid proportion data calculated", UserWarning)
        return {}
    
    # Get location names from first available timeframe
    first_available_timeframe = list(proportion_dfs_by_timeframe.keys())[0]
    location_names = list(proportion_dfs_by_timeframe[first_available_timeframe].keys())
    
    # Combine quarterly data into single timeline
    combined_dfs = {}
    
    for location in location_names:
        all_data = []
        new_index = []
        
        for year in all_years:
            for quarter in timeframes:
                # Skip if this quarter wasn't processed
                if quarter not in proportion_dfs_by_timeframe:
                    continue
                    
                df = proportion_dfs_by_timeframe[quarter][location]
                
                if year in df.index:
                    # Get data for this year-quarter
                    row_data = df.loc[year]
                    all_data.append(row_data)
                    
                    # Create fractional year for smooth plotting
                    # Q1=0.125, Q2=0.375, Q3=0.625, Q4=0.875
                    quarter_num = timeframes.index(quarter)
                    fractional_year = year + (quarter_num + 0.5) / len(timeframes)
                    new_index.append(fractional_year)
        
        # Combine into single DataFrame
        if all_data:
            combined_df = pd.DataFrame(all_data, index=new_index)
            combined_dfs[location] = combined_df
    
    return combined_dfs


def RF_area_plot_quarterly_workflow(base_dir, band_selection, timeframes, habitat_reference_dfs, 
                                    all_years, veluwe_plot_windows__gdf, save_dir, overwrite=True):
    """
    Create quarterly timeline stacked area plots with proportions (0-1 scale).
    
    Parameters
    ----------
    base_dir : str or Path
        Base directory containing band_selection folders
    band_selection : str
        Band selection folder name (e.g., 'b2348')
    timeframes : list
        List of timeframe identifiers (e.g., ['Q1', 'Q2', 'Q3', 'Q4'])
    habitat_reference_dfs : dict
        Dictionary mapping habitat groups to reference DataFrames
    all_years : list
        List of years to analyze
    veluwe_plot_windows__gdf : GeoDataFrame
        GeoDataFrame containing plot window geometries
    save_dir : str or Path
        Directory to save the output plots
    overwrite : bool, default=True
        Whether to overwrite existing files
    
    Returns
    -------
    dict
        Nested dictionary of figures: {hab_name: {location: fig}}
    """
    from pathlib import Path
    import warnings
    
    # Build the hab_div dictionary
    hab_div_nested = build_hab_div_dict(base_dir, band_selection, timeframes, habitat_reference_dfs)
    
    # Create save directory structure
    save_path = Path(save_dir) / band_selection
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Store all figures
    all_figures = {}
    
    # Loop through habitats
    for hab_idx, timeframe_dict in hab_div_nested.items():
        # Check if this habitat has any valid timeframes
        if not timeframe_dict:
            warnings.warn(f"Skipping habitat index {hab_idx} - no valid raster files found", UserWarning)
            continue
        
        # Get habitat info from first available timeframe
        first_available_timeframe = list(timeframe_dict.keys())[0]
        hab_name = timeframe_dict[first_available_timeframe][0]
        reference_df = timeframe_dict[first_available_timeframe][2]
        
        print(f"\n=== Processing {hab_name} ===")
        
        # Pass the timeframe_dict as a single-habitat nested dict
        single_hab_dict = {0: timeframe_dict}
        
        # Calculate quarterly proportions
        combined_dfs = calculate_quarterly_class_proportions(
            single_hab_dict, timeframes, all_years, veluwe_plot_windows__gdf
        )
        
        # Check if we got any valid data
        if not combined_dfs:
            warnings.warn(f"No valid data for {hab_name}, skipping plots", UserWarning)
            continue
        
        all_figures[hab_name] = {}
        
        # Create one plot per location
        for location_name, df in combined_dfs.items():
            try:
                fig = plot_quarterly_stacked_area(
                    df=df,
                    title=f'Land Cover Change: {hab_name} - {location_name}',
                    habitat_reference_df=reference_df,
                    hab_selection=hab_name,
                    band_selection=band_selection,
                    save_dir=save_path,
                    location_name=location_name,
                    figsize=(12, 4),
                    overwrite=overwrite
                )
                
                all_figures[hab_name][location_name] = fig
                
                if fig is not None:
                    plt.show()
            except Exception as e:
                warnings.warn(f"Error creating plot for {hab_name} - {location_name}: {e}", UserWarning)
                continue
    
    return all_figures