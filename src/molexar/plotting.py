"""Shared plotting style helpers for Molexar analysis scripts."""

import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


MACARON_PINK = "#F7A8B8"
MACARON_MINT = "#A8E6CF"
MACARON_BLUE = "#A7C7E7"
MACARON_LEMON = "#FFF2A8"
MACARON_PEACH = "#FFD3B6"
MACARON_LAVENDER = "#CDB4DB"
MACARON_CORAL = "#FFAAA5"
MACARON_PISTACHIO = "#D5E8D4"
MACARON_LILAC = "#B8C0FF"
MACARON_CREAM = "#FFF8E7"
MACARON_ROSE = "#F48FB1"
MACARON_RASPBERRY = "#F26D7D"
MACARON_SKY = "#8ECBEA"
MACARON_TEAL = "#70D6C4"
MACARON_APRICOT = "#F6B26B"
MACARON_GRAPE = "#B39DDB"
MACARON_GOLD = "#F4D35E"

MACARON_PALETTE = [
    MACARON_ROSE,
    MACARON_TEAL,
    MACARON_SKY,
    MACARON_APRICOT,
    MACARON_GRAPE,
    MACARON_MINT,
    MACARON_PEACH,
    MACARON_LILAC,
    MACARON_GOLD,
    MACARON_PINK,
    MACARON_BLUE,
    MACARON_LAVENDER,
]

MACARON_ACCENT = MACARON_RASPBERRY
MACARON_SECONDARY = "#5CC8B2"
MACARON_DARK = "#4A4E69"
MACARON_TEXT = "#2F3047"
MACARON_GRID = "#ECE7F2"
MACARON_BACKGROUND = "#FFFFFF"
MACARON_EDGE = "#D7CEE8"

MOLEXAR_COLOR = MACARON_RASPBERRY
MOLEXAR_LIGHT = "#F8B8C4"
BASELINE_DARK = "#6C6F93"
BASELINE_LIGHT = "#D8D6E8"

METHOD_COLORS = {
    "Molexar": MOLEXAR_COLOR,
    "SAFE": MACARON_SKY,
    "GenMol": MACARON_GRAPE,
    "ChemFM-1B": MACARON_APRICOT,
    "ChemFM-3B": MACARON_GOLD,
    "Training": MACARON_TEAL,
    "Generated": MACARON_ROSE,
    "Obedience": MACARON_ROSE,
}


def macaron_palette(n: Optional[int] = None) -> List[str]:
    """Return a repeated soft pastel macaron palette with length n if requested."""
    if n is None:
        return list(MACARON_PALETTE)
    if n <= 0:
        return []
    return [MACARON_PALETTE[index % len(MACARON_PALETTE)] for index in range(n)]


def macaron_method_color(name: str, fallback_index: int = 0) -> str:
    """Return a stable macaron color for a known method or a palette fallback."""
    return METHOD_COLORS.get(name, MACARON_PALETTE[fallback_index % len(MACARON_PALETTE)])


def apply_macaron_style(
    sns=None,
    plt=None,
    context: str = "talk",
    font_size: Optional[float] = None,
    axes_linewidth: float = 1.25,
    grid: bool = True,
) -> None:
    """Apply the shared macaron palette to seaborn/matplotlib plots."""
    rc = {
        "axes.facecolor": MACARON_BACKGROUND,
        "figure.facecolor": MACARON_BACKGROUND,
        "savefig.facecolor": MACARON_BACKGROUND,
        "axes.edgecolor": MACARON_TEXT,
        "axes.labelcolor": MACARON_TEXT,
        "text.color": MACARON_TEXT,
        "xtick.color": MACARON_TEXT,
        "ytick.color": MACARON_TEXT,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "axes.linewidth": axes_linewidth,
        "xtick.major.width": axes_linewidth,
        "ytick.major.width": axes_linewidth,
        "grid.color": MACARON_GRID,
        "grid.alpha": 0.55,
        "grid.linewidth": 0.8,
        "axes.grid": grid,
        "legend.frameon": False,
        "legend.framealpha": 0.0,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    }
    if font_size is not None:
        rc["font.size"] = font_size
    if sns is not None:
        sns.set_theme(style="whitegrid", context=context, palette=MACARON_PALETTE, rc=rc)
    if plt is not None:
        from cycler import cycler

        plt.rcParams.update(rc)
        plt.rcParams["axes.prop_cycle"] = cycler(color=MACARON_PALETTE)


def apply_axis_style(ax, grid_axis: str = "y", despine: bool = True) -> None:
    """Apply consistent grid and spine styling to one matplotlib axis."""
    ax.grid(True, axis=grid_axis, color=MACARON_GRID, alpha=0.55, linewidth=0.8)
    if despine:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for spine_name in ("left", "bottom"):
        if spine_name in ax.spines:
            ax.spines[spine_name].set_color(MACARON_TEXT)


def color_molexar_ticklabels(ax, names: Sequence[str], color: str = MOLEXAR_COLOR) -> None:
    """Highlight Molexar tick labels while keeping other labels neutral."""
    for label, name in zip(ax.get_xticklabels(), names):
        if name == "Molexar" or str(name).startswith("Molexar"):
            label.set_color(color)
            label.set_fontweight("bold")
        else:
            label.set_color(MACARON_TEXT)
            label.set_fontweight("normal")


def macaron_sequential_cmap(name: str = "molexar_macaron"):
    """Build a soft sequential colormap for density plots."""
    from matplotlib.colors import LinearSegmentedColormap

    colors: Sequence[str] = [MACARON_CREAM, MACARON_MINT, MACARON_BLUE, MACARON_LAVENDER]
    return LinearSegmentedColormap.from_list(name, colors)


def macaron_diverging_cmap(name: str = "molexar_macaron_diverging"):
    """Build a soft diverging colormap for correlation heatmaps."""
    from matplotlib.colors import LinearSegmentedColormap

    colors: Sequence[str] = [MACARON_SKY, MACARON_CREAM, MACARON_ROSE]
    return LinearSegmentedColormap.from_list(name, colors)


def macaron_score_cmap(name: str = "molexar_macaron_score"):
    """Build a soft score colormap where low values are rose and high values blue."""
    from matplotlib.colors import LinearSegmentedColormap

    colors: Sequence[str] = [MACARON_ROSE, MACARON_CREAM, MACARON_SKY]
    return LinearSegmentedColormap.from_list(name, colors)


def macaron_density_cmap(base_color: str, name: str = "molexar_density"):
    """Build a white-to-color colormap for point-density overlays."""
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap, to_rgb

    base_rgb = np.array(to_rgb(base_color))
    light = 0.55 * np.array([1.0, 1.0, 1.0]) + 0.45 * base_rgb
    return LinearSegmentedColormap.from_list(name, [light, base_rgb])


def macaron_heatmap_cmap(base_color: str, name: str = "molexar_heatmap"):
    """Build a soft white-to-saturated color map for small count heatmaps."""
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap, rgb_to_hsv, hsv_to_rgb, to_rgb

    base_rgb = np.array(to_rgb(base_color))
    hue, saturation, value = rgb_to_hsv(base_rgb)
    saturated = hsv_to_rgb([hue, min(1.0, saturation * 1.8), value])
    return LinearSegmentedColormap.from_list(name, [MACARON_BACKGROUND, base_rgb, saturated])


def transparent_rdkit_svg(svg_text: str) -> str:
    """Make the standard RDKit SVG canvas transparent."""
    return svg_text.replace("opacity:1.0;fill:#FFFFFF;", "opacity:0.0;fill:#FFFFFF;")


def opaque_rdkit_svg(svg_text: str) -> str:
    """Restore an opaque white RDKit SVG canvas."""
    return svg_text.replace("opacity:0.0;fill:#FFFFFF;", "opacity:1.0;fill:#FFFFFF;")


def save_rdkit_svg_bundle(
    svg_text: str,
    output_base: str | os.PathLike[str],
    png_scale: float = 2.0,
) -> None:
    """Save a transparent SVG/PDF bundle and a white-background PNG."""
    import cairosvg

    base = Path(output_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    transparent_svg = transparent_rdkit_svg(svg_text)
    base.with_suffix(".svg").write_text(transparent_svg, encoding="utf-8")
    cairosvg.svg2png(
        bytestring=opaque_rdkit_svg(transparent_svg).encode("utf-8"),
        write_to=str(base.with_suffix(".png")),
        scale=png_scale,
    )
    cairosvg.svg2pdf(
        bytestring=transparent_svg.encode("utf-8"),
        write_to=str(base.with_suffix(".pdf")),
    )


def save_figure_bundle(
    fig,
    output_base: str | os.PathLike[str],
    exts: Iterable[str] = ("svg", "pdf", "png"),
    dpi: int = 300,
    **savefig_kwargs,
) -> list[Path]:
    """Save one matplotlib figure to multiple formats using one base path."""
    base = Path(output_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    if base.suffix.lstrip(".") in set(exts):
        base = base.with_suffix("")
    written = []
    for ext in exts:
        out = base.with_suffix(f".{ext}")
        kwargs = dict(savefig_kwargs)
        if ext.lower() in {"png", "jpg", "jpeg", "tif", "tiff"}:
            kwargs.setdefault("dpi", dpi)
        fig.savefig(out, **kwargs)
        written.append(out)
    return written
