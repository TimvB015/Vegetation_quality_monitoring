################################################################################
#                               RGBA to HEX                                    #
################################################################################
import pandas as pd

def rgba_to_hex_func(rgba, include_alpha=False):
    r, g, b, a = rgba
    r = round(r * 255)
    g = round(g * 255)
    b = round(b * 255)
    if include_alpha:
        aa = round(a * 255)
        return f"#{r:02x}{g:02x}{b:02x}{aa:02x}"
    return f"#{r:02x}{g:02x}{b:02x}"