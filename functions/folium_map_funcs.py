################################################################################
## ADD A RETURN TO HOME BUTTON
################################################################################
import json
import folium
from branca.element import MacroElement, Template


class _RecenterControl(MacroElement):
    def __init__(self, center, zoom, position="topright", title="Reset view", size=36):
        super().__init__()
        self._name = "RecenterControl"
        self.center = center
        self.zoom = int(zoom)

        self.position = json.dumps(position)
        self.title = json.dumps(title)
        self.size = int(size)

        self._template = Template(
            """
            {% macro script(this, kwargs) %}
            (function() {
                var map = {{ this._parent.get_name() }};

                // Style the button to match Leaflet's LayerControl toggle size (typically 36x36)
                var css = `
                .leaflet-bar a.recenter-btn {
                    width: {{ this.size }}px !important;
                    height: {{ this.size }}px !important;
                    line-height: {{ this.size }}px !important;
                    text-align: center;
                     font-size: 18px;
                     display: block;
                    }`;
                var styleEl = document.createElement('style');
                styleEl.type = 'text/css';
                styleEl.appendChild(document.createTextNode(css));
                document.head.appendChild(styleEl);

                var HomeControl = L.Control.extend({
                    options: { position: {{ this.position }} },

                    onAdd: function(map) {
                        var container = L.DomUtil.create('div', 'leaflet-bar leaflet-control');
                        var btn = L.DomUtil.create('a', 'recenter-btn', container);

                        btn.href = '#';
                        btn.title = {{ this.title }};
                        btn.setAttribute('aria-label', {{ this.title }});
                        btn.innerHTML = '&#8962;';

                        L.DomEvent.disableClickPropagation(container);
                        L.DomEvent.on(btn, 'click', function(e) {
                            L.DomEvent.preventDefault(e);
                            map.setView([{{ this.center[0] }}, {{ this.center[1] }}], {{ this.zoom }});
                        });

                        return container;
                    }
                });

                map.addControl(new HomeControl());
            })();
            {% endmacro %}
            """
        )


def add_recenter_button(m: folium.Map, center, zoom, position="topright", title="Reset view", size=44):
    m.add_child(_RecenterControl(center=center, zoom=zoom, position=position, title=title, size=size))
    return m



################################################################################
## COLLAPSABLE LEGEND
################################################################################
from branca.element import MacroElement, Template
import pandas as pd
import uuid

def create_collapsible_legend(
    df=None,                     
    item_col='idx',
    color_col='color',
    legend_title='Legend',
    legend_id=None,               
    location='bottom-right',      
    offset_px=50,
    width=None,                  
    max_width='60vw',

    # Toggles (default False) for predefined items
    show_veluwe=False,
    show_veluwe_neutral=False,
    show_areas_of_interest=False,

    # Labels (override if you want different names)
    label_veluwe="Veluwe",
    label_veluwe_neutral="Veluwe (neutral)",
    label_areas_of_interest="Areas of interest",

    # Visuals for predefined items (legend swatches only; do not affect map layers)
    veluwe_fill_color="transparent",
    veluwe_border_color="#14c231",
    veluwe_neutral_fill="#ffffff",
    veluwe_neutral_border="#999999",
    areas_interest_fill="transparent",
    areas_interest_border="#ff0000",
):
    """
    Create a collapsible HTML legend (Folium/Branca).
    - df is optional; if None or empty, only the predefined toggle items are shown.
    - Duplicate legend names are skipped (first occurrence wins).
    """

    legend_id = legend_id or f"legend-{uuid.uuid4().hex[:8]}"

    pos_map = {
        'bottom-right': f"bottom: {offset_px}px; right: {offset_px}px;",
        'bottom-left':  f"bottom: {offset_px}px; left: {offset_px}px;",
        'top-right':    f"top: {offset_px}px; right: {offset_px}px;",
        'top-left':     f"top: {offset_px}px; left: {offset_px}px;",
    }
    position_css = pos_map.get(location, pos_map['bottom-right'])

    def swatch_html(fill, border="#333", border_width=1):
        return (
            f'<span style="display:inline-block;width:25px;height:18px;'
            f'background:{fill};margin-right:12px;'
            f'border:{border_width}px solid {border};"></span>'
        )

    rows_html = []
    seen_names = set()

    # ------------------------------------------------------------------
    # Predefined legend items (added first, order preserved)
    # ------------------------------------------------------------------

    if show_veluwe and label_veluwe not in seen_names:
        seen_names.add(label_veluwe)
        rows_html.append(
            f'<div style="margin-bottom:4px; display:flex; align-items:center;">'
            f'{swatch_html(veluwe_fill_color, veluwe_border_color, 2)}'
            f'<span>{label_veluwe}</span></div>'
        )

    if show_veluwe_neutral and label_veluwe_neutral not in seen_names:
        seen_names.add(label_veluwe_neutral)
        rows_html.append(
            f'<div style="margin-bottom:4px; display:flex; align-items:center;">'
            f'{swatch_html(veluwe_neutral_fill, veluwe_neutral_border, 1)}'
            f'<span>{label_veluwe_neutral}</span></div>'
        )

    if show_areas_of_interest and label_areas_of_interest not in seen_names:
        seen_names.add(label_areas_of_interest)
        rows_html.append(
            f'<div style="margin-bottom:4px; display:flex; align-items:center;">'
            f'{swatch_html(areas_interest_fill, areas_interest_border, 2)}'
            f'<span>{label_areas_of_interest}</span></div>'
        )

    # ------------------------------------------------------------------
    # Items from DataFrame
    # ------------------------------------------------------------------

    if isinstance(df, pd.DataFrame) and not df.empty:
        # Use index for names if requested or column not present
        use_index_for_name = (item_col == 'index') or (item_col not in df.columns)

        for idx, row in df.iterrows():
            color = row.get(color_col, None) if color_col in df.columns else None
            name = idx if use_index_for_name else row.get(item_col, None)

            # Skip invalid entries
            if pd.isna(color) or pd.isna(name):
                continue

            # Skip duplicate legend names
            if name in seen_names:
                continue

            seen_names.add(name)

            rows_html.append(
                f'<div style="margin-bottom:4px; display:flex; align-items:center; white-space:nowrap;">'
                f'{swatch_html(color)}'
                f'<span>{name}</span></div>'
            )

    items_html = "\n".join(rows_html) if rows_html else "<div>(No legend items)</div>"

    # ------------------------------------------------------------------
    # Width handling
    # ------------------------------------------------------------------

    if width:
        width_css = f"width: {width};"
        max_width_css = ""
        display_css = ""
        overflow_css = ""
    else:
        width_css = "width: fit-content; width: -moz-fit-content; width: max-content;"
        max_width_css = f"max-width: {max_width};"
        display_css = "display: inline-block;"
        overflow_css = "overflow-x: auto;"

    # ------------------------------------------------------------------
    # HTML + JS
    # ------------------------------------------------------------------

    legend_html = f"""
    <div id='{legend_id}' style="
        position: fixed;
        {position_css}
        {width_css}
        {max_width_css}
        {display_css}
        z-index:9999;
        font-size:14px;
        background: white;
        padding: 10px 10px 5px 10px;
        border: 2px solid grey;
        border-radius: 10px;
        box-shadow: 2px 2px 6px #888888;
        {overflow_css}
    ">
        <div id='{legend_id}-header' style="cursor:pointer; font-weight:bold; margin-bottom:10px;">
            {legend_title} <span id='{legend_id}-toggle' style='float:right;'>[+]</span>
        </div>
        <div id='{legend_id}-content' style="
            display:none;
            max-height: 50vh;
            overflow-y: auto;
            padding-right: 6px;
        ">
            {items_html}
        </div>
    </div>
    <script>
        (function() {{
            var content = document.getElementById('{legend_id}-content');
            var toggle = document.getElementById('{legend_id}-toggle');
            var header = document.getElementById('{legend_id}-header');
            content.style.display = 'none';
            toggle.innerText = '[+]';
            header.onclick = function() {{
                if (content.style.display === 'none') {{
                    content.style.display = '';
                    toggle.innerText = '[–]';
                }} else {{
                    content.style.display = 'none';
                    toggle.innerText = '[+]';
                }}
            }};
        }})();
    </script>
    """

    legend = MacroElement()
    legend._template = Template(f"""
        {{% macro html(this, kwargs) %}}
            {legend_html}
        {{% endmacro %}}
    """)

    return legend



################################################################################
## ADDING A RASTER LAYER
################################################################################
import numpy as np
import folium
from folium.raster_layers import ImageOverlay
import ast

def rstr_to_folium_layer(
    raster,
    colors_df,
    name="Raster",
    opacity=1.0,
    downsample=1,
    nodata=0,
    default_alpha=180,
    zindex=2,
    show=True,
    control=True,
    mercator_project=True,
    fit_bounds_map=None,
    validate_bounds=True,
):
    """
    Create a Folium `ImageOverlay` from a classified raster and a class→RGBA color table.

    The function converts a (masked) 2D raster of **integer class values** (e.g., land-use
    codes) into an RGBA image (dtype uint8, shape H×W×4) by looking up each raster value
    in `colors_df["rgba"]`. The resulting RGBA image is returned as a Folium
    `folium.raster_layers.ImageOverlay`, positioned using the raster bounds.

    Geographic referencing:
    - The overlay bounds are taken from `meta["bounds"]` and are assumed to be in
      **EPSG:4326 (WGS84 lon/lat degrees)**.
    - Leaflet/Folium expects bounds as `[[south, west], [north, east]]` (lat/lon). This
      function converts rasterio-style bounds (left, bottom, right, top) accordingly.
    - If `validate_bounds=True`, the function performs a sanity check to catch common CRS
      mistakes (e.g., accidentally passing RD New / EPSG:28992 bounds).

    Map view convenience:
    - If `fit_bounds_map` is provided (a `folium.Map`), the function calls
      `fit_bounds_map.fit_bounds(...)` with the raster bounds, so the overlay is
      automatically in view.

    Performance:
    - Large rasters can make Folium HTML heavy and slow to render. Use `downsample > 1`
      to reduce the image resolution (every Nth pixel in both directions).
    - If the overlay appears blank in the browser, it can be due to excessive image size;
      increasing `downsample` usually fixes this.

    Transparency / nodata:
    - If the raster is a masked array, masked pixels become fully transparent.
    - Raster pixels equal to `nodata` become fully transparent.
    - `colors_df["rgba"]` may be RGB or RGBA; RGB values are treated as fully opaque,
      unless overridden by `default_alpha` (see below).

    Color table / alpha handling:
    - `colors_df["rgba"]` may contain tuples/lists or string representations of them
      (e.g. "(255, 0, 0, 255)" or "(1.0, 0.0, 0.0, 1.0)").
    - Values may be 0..255 integers or 0..1 floats; floats are converted to 0..255.
    - If a class color has alpha==255 and `default_alpha` is not None, the alpha is
      replaced by `default_alpha` (useful to make a categorical overlay semi-transparent
      by default).
    - The `opacity` parameter is additionally applied by Leaflet as a multiplier.

    Parameters
    ----------
    raster : tuple
        Tuple `(arr, meta)` where:
        - `arr` is a 2D array-like (or squeezable-to-2D) `numpy.ndarray` or
          `numpy.ma.MaskedArray` containing class values.
        - `meta` is a dict containing at least:
          * `meta["bounds"]`: rasterio-style bounds (left, bottom, right, top) in EPSG:4326.
            This can be a rasterio BoundingBox-like object with `.left/.bottom/.right/.top`
            or a 4-tuple `(left, bottom, right, top)`.
    colors_df : pandas.DataFrame
        Class-to-color lookup table. Requirements:
        - `colors_df.index` contains the raster class values (integers).
        - `colors_df["rgba"]` contains RGB(A) values as tuple/list or parseable string.
    name : str
        Layer name shown in Folium's LayerControl.
    opacity : float
        Leaflet opacity multiplier (0..1) applied to the overlay.
    downsample : int
        Downsampling factor for speed. If N>1, uses `arr[::N, ::N]`.
    nodata : int
        Raster value treated as nodata (transparent).
    default_alpha : int | None
        If not None, replaces alpha=255 values in the color table with this alpha (0..255),
        to make otherwise-opaque class colors semi-transparent by default.
        Set to None to preserve the table alpha exactly.
    zindex : int
        Z-index ordering of the overlay (higher draws on top).
    show : bool
        If True, the overlay is visible on initial map load.
    control : bool
        If True, the overlay appears in Folium's LayerControl.
    mercator_project : bool
        Passed through to `ImageOverlay(mercator_project=...)`. Typically True for Folium
        (Leaflet display in WebMercator).
    fit_bounds_map : folium.Map | None
        If provided, the function calls `fit_bounds_map.fit_bounds(bounds)` to zoom the
        map to the overlay extent.
    validate_bounds : bool
        If True, validates that bounds look like EPSG:4326:
        - longitudes within [-180, 180]
        - latitudes within [-90, 90]
        - left < right and bottom < top
        If validation fails, raises a ValueError with a hint about CRS mismatch.

    Returns
    -------
    folium.raster_layers.ImageOverlay
        A Folium overlay that you can add to a map via `.add_to(map)`.

    Raises
    ------
    ValueError
        If `raster` is not a `(arr, meta_dict)` tuple, if `meta["bounds"]` is missing or
        unparsable, if the raster cannot be reduced to a 2D array, or if bounds validation
        fails (when `validate_bounds=True`).
    TypeError
        If `fit_bounds_map` is provided but does not implement `.fit_bounds(...)`.

    Notes
    -----
    This function does not reproject rasters. If your raster is in RD New (EPSG:28992),
    reproject it to EPSG:4326 first and ensure `meta["bounds"]` reflects the EPSG:4326
    raster.

    Example
    -------
    >>> m = folium.Map(location=[52.2, 5.6], zoom_start=8)
    >>> layer = rstr_to_folium_layer(
    ...     raster_tuple,
    ...     colors_df,
    ...     name="LGN 2023",
    ...     downsample=5,
    ...     fit_bounds_map=m,
    ... )
    >>> layer.add_to(m)
    >>> folium.LayerControl().add_to(m)
    """
    # --- unpack raster + meta ---
    if isinstance(raster, tuple) and len(raster) == 2 and isinstance(raster[1], dict):
        arr, meta = raster
    else:
        raise ValueError("raster must be a (masked_array, meta_dict) tuple containing WGS84 bounds.")

    if "bounds" not in meta:
        raise ValueError("meta_dict must include 'bounds' (WGS84).")

    # --- normalize bounds object to numbers ---
    b = meta["bounds"]
    try:
        left, bottom, right, top = float(b.left), float(b.bottom), float(b.right), float(b.top)
    except Exception:
        # allow tuple/list like (left,bottom,right,top)
        try:
            left, bottom, right, top = map(float, b)
        except Exception as e:
            raise ValueError(
                "meta['bounds'] must be a rasterio BoundingBox-like object "
                "(with left/bottom/right/top) or a 4-tuple (left,bottom,right,top)."
            ) from e

    # --- optional sanity check for EPSG:4326 bounds ---
    if validate_bounds:
        problems = []
        if not (left < right and bottom < top):
            problems.append(f"invalid ordering (left={left}, right={right}, bottom={bottom}, top={top})")
        if not (-180.0 <= left <= 180.0 and -180.0 <= right <= 180.0):
            problems.append(f"longitude out of range (left={left}, right={right})")
        if not (-90.0 <= bottom <= 90.0 and -90.0 <= top <= 90.0):
            problems.append(f"latitude out of range (bottom={bottom}, top={top})")

        if problems:
            raise ValueError(
                "meta['bounds'] does not look like EPSG:4326 (WGS84 lon/lat degrees): "
                + "; ".join(problems)
                + ". If your raster is still in a projected CRS (e.g. EPSG:28992), "
                  "reproject it to EPSG:4326 and update bounds before using Folium."
            )

    # --- array to 2D ---
    arr2d = np.ma.squeeze(arr)
    if arr2d.ndim != 2:
        raise ValueError(f"Expected a 2D raster after squeeze, got shape {arr2d.shape}")

    # --- downsample to keep folium fast ---
    if downsample and downsample > 1:
        arr2d = arr2d[::downsample, ::downsample]

    data = arr2d.filled(nodata)

    # --- build LUT from df (fast class->RGBA mapping) ---
    def _parse_rgba(x):
        if isinstance(x, str):
            x = ast.literal_eval(x)
        x = tuple(x)
        if len(x) == 3:
            x = (*x, 1.0)
        if all(isinstance(v, (float, np.floating)) for v in x):
            r, g, b, a = [int(round(v * 255)) for v in x]
        else:
            r, g, b, a = [int(v) for v in x]
        return (r, g, b, a)

    max_class = int(max(int(v) for v in colors_df.index))
    lut_size = max(256, max_class + 1)
    lut = np.zeros((lut_size, 4), dtype=np.uint8)  # default transparent

    for cls_val, row in colors_df.iterrows():
        cls_val = int(cls_val)
        r, g, b, a = _parse_rgba(row["rgba"])
        if a == 255 and default_alpha is not None:
            a = int(default_alpha)
        lut[cls_val] = (r, g, b, a)

    # Ensure nodata is transparent
    if 0 <= int(nodata) < lut.shape[0]:
        lut[int(nodata), 3] = 0

    # --- map raster values to RGBA ---
    data_int = np.asarray(data, dtype=np.int32)
    data_int = np.clip(data_int, 0, lut.shape[0] - 1)
    rgba = lut[data_int].copy()

    mask = np.ma.getmaskarray(arr2d) | (data_int == int(nodata))
    rgba[..., 3] = np.where(mask, 0, rgba[..., 3]).astype(np.uint8)

    # --- folium bounds (lat,lon pairs) ---
    bounds = [[bottom, left], [top, right]]

    # Optional: zoom map to raster bounds so it is reliably visible
    if fit_bounds_map is not None:
        if not hasattr(fit_bounds_map, "fit_bounds"):
            raise TypeError("fit_bounds_map must be a folium.Map (or have a .fit_bounds method).")
        fit_bounds_map.fit_bounds(bounds)

    return ImageOverlay(
        image=rgba,
        bounds=bounds,
        name=name,
        opacity=float(opacity),
        interactive=True,
        cross_origin=False,
        zindex=int(zindex),
        show=bool(show),
        control=bool(control),
        overlay=True,
        mercator_project=bool(mercator_project),
    )



################################################################################
## ADDING LAYERS USING IDX DF AND GEOPACKAGE
################################################################################
import pandas as pd
import folium

def idx_df_to_folium_layers(
    folium_map,
    layer_index_df,
    geometries_gdf,
    legend_df,
    label_scheme,  # "Habitat" | "WD-div"
    default_color="#999999",
    show_layers=True,
    add_layer_numbers=True,
):
    """
    Create and add Folium overlay layers (FeatureGroups) from an index mapping table.

    For each column in `layer_index_df`, a new map overlay layer is created and added to
    `folium_map`. Each row of `layer_index_df` represents a *label* (the row index), and
    each cell is expected to contain a list/tuple of GeoDataFrame indices that belong to
    that label for the given layer column.

    Workflow per layer column
    -------------------------
    - Create a FeatureGroup named after the column (optionally prefixed with a running
      layer number).
    - For each label in `layer_index_df`:
        * read the list of geometry IDs from `layer_index_df.loc[label, layer_col]`
        * select those geometries from `geometries_gdf` (based on `geometries_gdf.index`)
        * add them as a `folium.GeoJson` to the FeatureGroup, styled with a label-specific
          fill color and a tooltip showing the label

    Label -> color mapping
    ----------------------
    Colors are looked up in `robust_names_df` and depend on `label_scheme`:

    - label_scheme="Habitat"
        * label/tooltip values correspond to `robust_names_df["habitatnaam_disp"]`
        * colors are taken from `robust_names_df["habitat_color"]`

    - label_scheme="WD-div"
        * label/tooltip values correspond to `robust_names_df["WD_division"]`
        * colors are taken from `robust_names_df["WD_color"]`

    If a label is not present in the color lookup, `default_color` is used.

    Parameters
    ----------
    folium_map : folium.Map
        Target map to which the layers are added.
    layer_index_df : pandas.DataFrame
        Mapping table where each column becomes a Folium layer. Rows are labels (index
        values). Cells contain lists/tuples of `geometries_gdf` index values.
    geometries_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing the geometries to display. Must be indexed by the IDs
        referenced inside `layer_index_df`.
    robust_names_df : pandas.DataFrame
        Table providing label-to-color information (columns depend on `label_scheme`).
    label_scheme : str
        Controls which columns in `robust_names_df` are used for label matching and
        color: "Habitat" or "WD-div".
    default_color : str
        Hex color used when a label has no color in `robust_names_df`.
    show_layers : bool
        Whether each added FeatureGroup is visible by default.
    add_layer_numbers : bool
        If True, prefixes layer names with a zero-padded running number (e.g. "01. ...").

    Returns
    -------
    folium.Map
        The same `folium_map` instance with the new FeatureGroups added.

    Raises
    ------
    ValueError
        If `label_scheme` is not one of {"Habitat", "WD-div"}.
    """

    # --- configuration based on label_scheme ---
    if label_scheme == "Habitat":
        label_col_in_names = "habitatnaam_disp"
        color_col_in_names = "habitat_color"
    elif label_scheme == "WD-div":
        label_col_in_names = "WD_division"
        color_col_in_names = "WD_color"
    else:
        raise ValueError("label_scheme must be one of {'Habitat', 'WD-div'}")

    # --- build label -> color lookup ---
    label_to_color = (
        legend_df[[label_col_in_names, color_col_in_names]]
        .dropna(subset=[label_col_in_names])
        .drop_duplicates(subset=[label_col_in_names], keep="first")
        .set_index(label_col_in_names)[color_col_in_names]
        .to_dict()
    )

    layer_columns = list(layer_index_df.columns)
    pad = max(2, len(str(len(layer_columns))))
    geometry_id_set = set(geometries_gdf.index)

    for layer_nr, layer_col in enumerate(layer_columns, start=1):
        layer_name = f"{layer_nr:0{pad}d}. {layer_col}" if add_layer_numbers else str(layer_col)
        feature_group = folium.FeatureGroup(name=layer_name, show=show_layers)

        for label, row in layer_index_df.iterrows():
            geometry_ids = row[layer_col]
            if not isinstance(geometry_ids, (list, tuple)) or not geometry_ids:
                continue

            ids_found = [gid for gid in geometry_ids if gid in geometry_id_set]
            if not ids_found:
                continue

            selection_gdf = geometries_gdf.loc[ids_found].copy()
            if selection_gdf.empty:
                continue

            fill_color = label_to_color.get(label, default_color)
            selection_gdf["_tooltip"] = str(label)

            folium.GeoJson(
                selection_gdf,
                style_function=lambda feature, color=fill_color: {
                    "fillColor": color,
                    "fillOpacity": 1,
                    "stroke": False,
                },
                tooltip=folium.GeoJsonTooltip(fields=["_tooltip"], labels=False),
            ).add_to(feature_group)

        feature_group.add_to(folium_map)

    return folium_map