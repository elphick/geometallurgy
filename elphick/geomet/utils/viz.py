from typing import Optional

import pandas as pd

import plotly.graph_objects as go


def plot_parallel(data: pd.DataFrame, color: Optional[str] = None, title: Optional[str] = None) -> go.Figure:
    """Create an interactive parallel plot

    Useful to explore multi-dimensional data like mass-composition data

    Args:
        data: Dataframe to plot
        color: Optional color variable
        title: Optional plot title

    Returns:

    """

    # Kudos: https://stackoverflow.com/questions/72125802/parallel-coordinate-plot-in-plotly-with-continuous-
    # and-categorical-data

    categorical_columns = data.select_dtypes(include=['category', 'object'])
    col_list = []

    for col in data.columns:
        if col in categorical_columns:  # categorical columns
            values = data[col].unique()
            value2dummy = dict(zip(values, range(
                len(values))))  # works if values are strings, otherwise we probably need to convert them
            data[col] = [value2dummy[v] for v in data[col]]
            col_dict = dict(
                label=col,
                tickvals=list(value2dummy.values()),
                ticktext=list(value2dummy.keys()),
                values=data[col],
            )
        else:  # continuous columns
            col_dict = dict(
                range=(data[col].min(), data[col].max()),
                label=col,
                values=data[col],
            )
        col_list.append(col_dict)

    if color is None:
        fig = go.Figure(data=go.Parcoords(dimensions=col_list))
    else:
        fig = go.Figure(data=go.Parcoords(dimensions=col_list, line=dict(color=data[color])))

    fig.update_layout(title=title)

    return fig
