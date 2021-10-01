"""Helper functions for all kinds of 2D/3D visualization"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import HoverTool
from bokeh.io import output_notebook
# from bokeh.palettes import Spectral as palette
import itertools


def bokeh_2d_scatter(x, y, desc, figsize=(700, 700), colors=None, use_nb=False, title="Bokeh scatter plot"):

    if use_nb:
        output_notebook()

    # define colors to be assigned
    if colors is None:
        # applies the same color
        # create a color iterator: pick a random color and apply it to all points
        # colors = [np.random.choice(itertools.cycle(palette))] * len(x)
        colors = [np.random.choice(["red", "green", "blue", "yellow", "pink", "black", "gray"])] * len(x)

        # # applies different colors
        # colors = np.array([ [r, g, 150] for r, g in zip(50 + 2*x, 30 + 2*y) ], dtype="uint8")


    # define the df of data to plot
    source = ColumnDataSource(
            data=dict(
                x=x,
                y=y,
                desc=desc,
                color=colors,
            )
        )

    # define the attributes to show on hover
    hover = HoverTool(
            tooltips=[
                ("index", "$index"),
                ("(x, y)", "($x, $y)"),
                ("Desc", "@desc"),
            ]
        )

    p = figure(
        plot_width=figsize[0], plot_height=figsize[1], tools=[hover], title=title,
    )
    p.circle('x', 'y', size=10, source=source, fill_color="color")
    show(p)




def bokeh_2d_scatter_new(
        df, x, y, hue, label, color_column=None,
        figsize=(700, 700), use_nb=False, title="Bokeh scatter plot"
    ):

    if use_nb:
        output_notebook()
    
    assert {x, y, hue, label}.issubset(set(df.keys()))

    if isinstance(color_column, str) and color_column in df.keys():
        color_column_name = color_column
    else:
        colors = list(mcolors.BASE_COLORS.keys()) + list(mcolors.TABLEAU_COLORS.values())
        colors = itertools.cycle(np.unique(colors))

        hue_to_color = dict()
        unique_hues = np.unique(df[hue].values)
        for _hue in unique_hues:
            hue_to_color[_hue] = next(colors)
        df["color"] = df[hue].apply(lambda k: hue_to_color[k])
        color_column_name = "color"

    source = ColumnDataSource(
        dict(
            x = df[x].values,
            y = df[y].values,
            hue = df[hue].values,
            label = df[label].values,
            color = df[color_column_name].values,
        )
    )

    # define the attributes to show on hover
    hover = HoverTool(
            tooltips=[
                ("index", "$index"),
                ("(x, y)", "($x, $y)"),
                ("Desc", "@label"),
                ("Cluster", "@hue"),
            ]
        )

    p = figure(
        plot_width=figsize[0],
        plot_height=figsize[1],
        tools=["pan","wheel_zoom","box_zoom","save","reset","help"] + [hover],
        title=title,
    )
    p.circle('x', 'y', size=10, source=source, fill_color="color", legend_group="hue")
    p.legend.location = "bottom_left"
    p.legend.click_policy="hide"

    show(p)
