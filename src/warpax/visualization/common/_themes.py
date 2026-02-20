"""Rendering theme presets for paper and presentation output.

Two built-in themes:
- ``PAPER_THEME``: White background, CQG-compatible styling for publication figures.
- ``PRESENTATION_THEME``: Dark background with specular highlights for talks.

Every render function accepts ``theme="paper"`` or ``theme="presentation"``.
"""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class RenderTheme:
    """Immutable collection of rendering parameters for a visual style.

    Parameters
    ----------
    background : str
        Background color (name or hex).
    font_color : str
        Text color for annotations and colorbars.
    font_size : int
        Base font size for annotations.
    ambient : float
        Ambient lighting coefficient (0--1).
    diffuse : float
        Diffuse lighting coefficient (0--1).
    specular : float
        Specular lighting coefficient (0--1).
    specular_power : float
        Specular exponent (higher = tighter highlight).
    window_size : tuple[int, int]
        Off-screen render window ``(width, height)`` in pixels.
    """

    background: str
    font_color: str
    font_size: int
    ambient: float
    diffuse: float
    specular: float
    specular_power: float
    window_size: tuple[int, int]


# ---------------------------------------------------------------------------
# Built-in presets
# ---------------------------------------------------------------------------

PAPER_THEME = RenderTheme(
    background="white",
    font_color="black",
    font_size=12,
    ambient=0.2,
    diffuse=0.8,
    specular=0.1,
    specular_power=10.0,
    window_size=(2400, 1800),
)
"""CQG publication style: white background, minimal specular, clean lines."""

PRESENTATION_THEME = RenderTheme(
    background="#1a1a2e",
    font_color="white",
    font_size=16,
    ambient=0.05,
    diffuse=0.6,
    specular=0.5,
    specular_power=50.0,
    window_size=(2400, 1800),
)
"""Conference talk style: dark background, dramatic specular highlights."""

_THEMES: dict[str, RenderTheme] = {
    "paper": PAPER_THEME,
    "presentation": PRESENTATION_THEME,
}


def get_theme(name: str) -> RenderTheme:
    """Look up a named rendering theme.

    Parameters
    ----------
    name : str
        Theme identifier: ``"paper"`` or ``"presentation"``.

    Returns
    -------
    RenderTheme

    Raises
    ------
    ValueError
        If *name* is not a recognized theme.
    """
    try:
        return _THEMES[name]
    except KeyError:
        available = ", ".join(sorted(_THEMES))
        raise ValueError(
            f"Unknown theme {name!r}. Available themes: {available}"
        ) from None
