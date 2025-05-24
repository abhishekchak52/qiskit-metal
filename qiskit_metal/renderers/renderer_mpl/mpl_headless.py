import logging

import matplotlib.pyplot as plt

from .mpl_renderer import QMplRenderer



logging.basicConfig(level=logging.INFO)



def plot_design_headless(design, **kwargs):
    """Plot a design in headless mode.

    Args:
        design (QDesign): The design to plot.
        **kwargs: Additional keyword arguments to pass to plt.subplots.

    Returns:
        fig (matplotlib.figure.Figure): The figure.
        ax (matplotlib.axes.Axes): The axis.
    """
    headless_plot_logger = logging.getLogger(__name__)

    design_renderer = QMplRenderer(canvas=None, design=design, logger=headless_plot_logger)
    fig, ax = plt.subplots(1, 1, layout='constrained', **kwargs)
    xmin, ymin, xmax, ymax = design.get_x_y_for_chip("main")[0]
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    design_renderer.render(ax)
    ax.set_aspect('equal')
    
    return fig, ax
