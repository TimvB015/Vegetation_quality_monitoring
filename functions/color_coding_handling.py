################################################################################
## HEX TO RGB
################################################################################
import numpy as np

def hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    """
    Convert hex color to RGB tuple in [0, 1] range.
    
    Parameters
    ----------
    hex_color : str
        Hex color string (e.g., '#ff0000' or 'ff0000')
    
    Returns
    -------
    tuple[float, float, float]
        RGB values in range [0, 1]
    
    Examples
    --------
    >>> hex_to_rgb('#ff0000')
    (1.0, 0.0, 0.0)
    >>> hex_to_rgb('#00ff00')
    (0.0, 1.0, 0.0)
    >>> hex_to_rgb('0000ff')
    (0.0, 0.0, 1.0)
    """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

def rgb_to_hsv(r: float, g: float, b: float) -> tuple[float, float, float]:
    """
    Convert RGB to HSV.
    
    Args:
        r, g, b: RGB values in [0, 1]
    
    Returns:
        h, s, v: HSV values where h in [0, 360], s and v in [0, 1]
    """
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    delta = max_c - min_c
    
    # Value
    v = max_c
    
    # Saturation
    if max_c == 0:
        s = 0
    else:
        s = delta / max_c
    
    # Hue
    if delta == 0:
        h = 0
    elif max_c == r:
        h = 60 * (((g - b) / delta) % 6)
    elif max_c == g:
        h = 60 * (((b - r) / delta) + 2)
    else:  # max_c == b
        h = 60 * (((r - g) / delta) + 4)
    
    return h, s, v



################################################################################
## HSV TO RGB
################################################################################
def hsv_to_rgb(h: float, s: float, v: float) -> np.ndarray:
    """
    Convert HSV to RGB.
    
    Args:
        h: Hue in [0, 360]
        s: Saturation in [0, 1]
        v: Value in [0, 1]
    
    Returns:
        RGB array with values in [0, 1]
    """
    c = v * s
    x = c * (1 - abs(((h / 60) % 2) - 1))
    m = v - c
    
    if h < 60:
        r, g, b = c, x, 0
    elif h < 120:
        r, g, b = x, c, 0
    elif h < 180:
        r, g, b = 0, c, x
    elif h < 240:
        r, g, b = 0, x, c
    elif h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    
    return np.array([r + m, g + m, b + m], dtype=np.float32)



################################################################################
## HEX TO RGB
################################################################################
def hex_to_rgb(hex_color: str) -> np.ndarray:
    """
    Convert hex color to RGB.
    
    Args:
        hex_color: Hex color string (e.g., "#ff0000" or "#ff0000ff")
    
    Returns:
        RGB array with values in [0, 1]
    """
    hex_color = hex_color.lstrip('#')
    
    # Handle both 6-digit (RGB) and 8-digit (RGBA) hex codes
    if len(hex_color) >= 6:
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        return np.array([r, g, b], dtype=np.float32)
    else:
        raise ValueError(f"Invalid hex color: {hex_color}")