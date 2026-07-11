"""
Shared icon utilities for the ISRO AQI & HCHO dashboard.

Provides a helper to render Google Material Symbols Outlined icons inline
and a CSS-dot helper for color-coded AQI/severity status indicators.
"""

# --- Material Symbols stylesheet injection (call once per page) ---
MATERIAL_SYMBOLS_CSS = """
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap" />
<style>
  .material-symbols-outlined {
    vertical-align: middle;
    display: inline-flex;
    align-items: center;
    font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
  }
</style>
"""


def inject_material_icons(st_module):
    """Inject the Material Symbols stylesheet once at the top of a page.
    
    Call this near the top of every page/app file:
        from dashboard.icon_utils import inject_material_icons
        inject_material_icons(st)
    """
    st_module.markdown(MATERIAL_SYMBOLS_CSS, unsafe_allow_html=True)


def icon(name: str, size: int = 20, color: str = "inherit") -> str:
    """Return an HTML span that renders a Material Symbols Outlined icon.

    Args:
        name:  Material Symbols icon name (e.g. "satellite_alt", "search").
        size:  Icon size in pixels.
        color: CSS color value.

    Returns:
        An HTML string safe for use with unsafe_allow_html=True.
    """
    return (
        f'<span class="material-symbols-outlined" '
        f'style="font-size:{size}px; color:{color}; vertical-align:middle;">'
        f'{name}</span>'
    )


def status_dot(color_hex: str, size: int = 10) -> str:
    """Return a small colored CSS circle for AQI/severity status badges.

    Use this instead of emoji color circles (🟢 🟡 🟠 🔴).

    Args:
        color_hex: The fill color, e.g. "#00e400".
        size:      Diameter of the dot in pixels.

    Returns:
        An HTML string for the dot.
    """
    return (
        f'<span style="'
        f'display:inline-block; '
        f'width:{size}px; height:{size}px; '
        f'border-radius:50%; '
        f'background-color:{color_hex}; '
        f'vertical-align:middle; '
        f'margin-right:4px;'
        f'"></span>'
    )
