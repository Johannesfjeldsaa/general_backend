"""Utility functions and settings for consistent plotting visualizations.

This module provides utility functions and settings for plotting in a
consistent way across different visualizations. It includes functions
for setting matplotlib and seaborn styles, managing colormaps, and
ensuring visual consistency across the project.
"""

from typing import Any, Literal

import cmocean  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore[import-untyped]

from general_backend.logging.setup_logging import get_logger

logger = get_logger(__name__)


def set_mpl_sns_style(
    style: Literal[
        "ticks", "whitegrid", "darkgrid", "white", "dark"
    ] = "ticks",
    context: Literal["notebook", "paper", "talk", "poster"] = "notebook",
) -> None:
    """Set the style for matplotlib and seaborn plots.

    Parameters
    ----------
    style : str, optional
        The style to use for the plots, by default 'ticks'.
        See https://seaborn.pydata.org/generated/seaborn.set_theme.html
        for available styles.
    context : str, optional
        The context to use for the plots, by default 'notebook'.
        This affects the scale of the plot elements (e.g., labels, titles)
        and is useful for adjusting the appearance of plots for different
        environments (e.g., paper, notebook, talk).
        See seaborn documentation for available contexts:
        https://seaborn.pydata.org/generated/seaborn.plotting_context.html#seaborn.plotting_context
    """
    sns.set_theme(context=context, style=style, font_scale=1)
    plt.style.use(f"seaborn-v0_8-{style}")
    plt.rcParams.update(
        {
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.figsize": (10, 5),
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def get_colormap(key: str) -> Any:
    """Get a colormap based on what you want to visualize.

    Parameters
    ----------
    key : str
        The name of the context for which you want a colormap.
        This can be one of the following:
        - 'heatmap_diverging': for diverging heatmaps
        - 'heatmap_positive': for positive heatmaps
        - 'diff': for difference maps
        - 'anomaly': for anomaly maps
        - 'mask_visualization': for visualizing masks
        Also accepts any valid cmocean colormap name.
        See https://matplotlib.org/cmocean/ for available colormaps.

    Returns
    -------
    np.ndarray
        The colormap corresponding to the key.

    Raises
    ------
    ValueError
        If the key does not match any predefined colormap
        or a valid cmocean colormap.
    """
    # Map variable names to appropriate cmocean colormaps
    mapping = {
        "heatmap_diverging": cmocean.tools.crop_by_percent(
            cmocean.cm.curl, 20, which="both", N=None
        ),
        "heatmap_positive": cmocean.cm.matter,
        "diff": cmocean.cm.diff,
        "anomaly": cmocean.cm.tarn,
        "mask_visualization": cmocean.tools.lighten(cmocean.cm.haline, 0.5),
    }
    if key in mapping:
        return mapping[key]
    elif key in cmocean.cm.__dict__:
        return cmocean.cm.__dict__[key]
    else:
        err_msg = (
            f"No colormap defined for context '{key}'. "
            "Please define a mapping in mapping)."
        )
        logger.error(err_msg)
        raise ValueError(err_msg)


# Style mapping for common statistics measures and reference/observation cases
STATISTIC_STYLES = {
    "mean": {"linestyle": "-", "linewidth": 2, "label": "Mean"},
    "median": {"linestyle": "--", "linewidth": 2, "label": "Median"},
    "min": {"linestyle": ":", "linewidth": 2, "label": "Min"},
    "max": {"linestyle": ":", "linewidth": 2, "label": "Max"},
    "range": {
        "linestyle": ":",
        "linewidth": 1,
        "label": "Range",
        "alpha": 0.5,
    },
    "std": {"linestyle": "-.", "linewidth": 2, "label": "Std dev"},
    "stdrange": {
        "linestyle": "--",
        "linewidth": 2,
        "label": "Std range",
        "alpha": 0.5,
    },
    "percentile_10": {
        "linestyle": ":",
        "linewidth": 1,
        "label": "10th Percentile",
    },
    "percentile_90": {
        "linestyle": ":",
        "linewidth": 1,
        "label": "90th Percentile",
    },
    "members": {
        "color": "#A8A8A8",
        "alpha": 0.5,
        "linestyle": "-",
        "linewidth": 1,
        "label": "Members",
    },
    "historic": {
        "color": "#333333",
        "linestyle": "-.",
        "linewidth": 2,
        "label": "Historic",
    },
    "SSP1-1.9": {
        "color": "#3FA191",
        "linestyle": "-",
        "linewidth": 2,
        "label": "SSP1-1.9",
    },
    "SSP5-3.4-OS": {
        "color": "#922A3F",
        "linestyle": "-",
        "linewidth": 2,
        "label": "SSP5-3.4-OS",
    },
    "SSP2-4.5": {
        "color": "#EDD937",
        "linestyle": "-",
        "linewidth": 2,
        "label": "SSP2-4.5",
    },
}
