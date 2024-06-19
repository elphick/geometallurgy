from typing import Optional, List, Union, Dict, Tuple

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from elphick.geomet.utils.size import mean_size
from elphick.geomet.utils.viz import plot_parallel


def parallel_plot(data: pd.DataFrame,
                  color: Optional[str] = None,
                  vars_include: Optional[List[str]] = None,
                  vars_exclude: Optional[List[str]] = None,
                  title: Optional[str] = None,
                  include_dims: Optional[Union[bool, List[str]]] = True,
                  plot_interval_edges: bool = False) -> go.Figure:
    """Create an interactive parallel plot

    Useful to explore multidimensional data like mass-composition data

    Args:
        data: The DataFrame to plot
        color: Optional color variable
        vars_include: Optional List of variables to include in the plot
        vars_exclude: Optional List of variables to exclude in the plot
        title: Optional plot title
        include_dims: Optional boolean or list of dimension to include in the plot.  True will show all dims.
        plot_interval_edges: If True, interval edges will be plotted instead of interval mid

    Returns:

    """
    df: pd.DataFrame = data.copy()
    if vars_include is not None:
        missing_vars = set(vars_include).difference(set(df.columns))
        if len(missing_vars) > 0:
            raise KeyError(f'var_subset provided contains variable not found in the data: {missing_vars}')
        df = df[vars_include]
    if vars_exclude:
        df = df[[col for col in df.columns if col not in vars_exclude]]

    if include_dims is True:
        df.reset_index(inplace=True)
    elif isinstance(include_dims, List):
        for d in include_dims:
            df.reset_index(d, inplace=True)

    interval_cols: Dict[str, int] = {col: i for i, col in enumerate(df.columns) if df[col].dtype == 'interval'}

    for col, pos in interval_cols.items():
        if plot_interval_edges:
            df.insert(loc=pos + 1, column=f'{col}_left', value=df[col].array.left)
            df.insert(loc=pos + 2, column=f'{col}_right', value=df[col].array.right)
            df.drop(columns=col, inplace=True)
        else:
            # workaround for https://github.com/Elphick/mass-composition/issues/1
            if col == 'size':
                df[col] = mean_size(pd.arrays.IntervalArray(df[col]))
            else:
                df[col] = df[col].array.mid

    fig = plot_parallel(data=df, color=color, title=title)
    return fig


def comparison_plot(data: pd.DataFrame,
                    x: str, y: str,
                    facet_col_wrap: int = 3,
                    color: Optional[str] = None,
                    trendline: bool = False,
                    trendline_kwargs: Optional[Dict] = None) -> go.Figure:
    """Comparison Plot with multiple x-y scatter plots

    Args:
        data: DataFrame, in tidy (tall) format, with columns for x and y
        x: The x column
        y: The y column
        facet_col_wrap: the number of subplots per row before wrapping
        color: The optional variable to color by. If None color will be by Node
        trendline: If True add trendlines
        trendline_kwargs: Allows customising the trendline: ref: https://plotly.com/python/linear-fits/.  Note: Axis
         scaling across components can be affected if using {'trendline_scope': 'trendline_scope'}.

    Returns:
        plotly Figure
    """
    if trendline:
        if trendline_kwargs is None:
            trendline_kwargs = {'trendline': 'ols'}
        else:
            if 'trendline' not in trendline_kwargs:
                trendline_kwargs['trendline'] = "ols"
    else:
        trendline_kwargs = {'trendline': None}

    data['residual'] = data[x] - data[y]
    fig = px.scatter(data, x=x, y=y, color=color,
                     facet_col='variable', facet_col_wrap=facet_col_wrap,
                     hover_data=['residual'],
                     **trendline_kwargs)

    # fig.print_grid()
    # add y=x based on data per subplot
    variable_order = list(data['variable'].unique())
    d_subplots = subplot_index_by_title(fig, variable_order)

    for k, v in d_subplots.items():
        tmp_df = data.query('variable==@k')
        limits = [min([tmp_df[x].min(), tmp_df[y].min()]),
                  max([tmp_df[x].max(), tmp_df[y].max()])]

        equal_trace = go.Scatter(x=limits, y=limits,
                                 line_color="gray", name="y=x", mode='lines', legendgroup='y=x', showlegend=False)
        fig.add_trace(equal_trace, row=v[0], col=v[1], exclude_empty_subplots=True)
        sp = fig.get_subplot(v[0], v[1])
        fig.update_xaxes(scaleanchor=sp.xaxis.anchor, scaleratio=1, row=v[0], col=v[1])

    fig.update_traces(selector=-1, showlegend=True)
    fig.for_each_yaxis(lambda _y: _y.update(showticklabels=True, matches=None))
    fig.for_each_xaxis(lambda _x: _x.update(showticklabels=True, matches=None))

    return fig


def subplot_index_by_title(fig, variable_order: List[str]) -> Dict['str', Tuple[int, int]]:
    """Map of subplot index by title

    Assumes consistency by plotly between axes numbering and annotation order.

    Args:
        fig: The figure including subplots with unique titles
        variable_order: the variables in order top-left to bottom-right

    Returns:
        Dict keyed by title with tuple of subplot positions
    """

    d_subplots: Dict = {}
    i = 0
    for r in range(len(fig._grid_ref), 0, -1):
        for c in range(1, len(fig._grid_ref[0]) + 1, 1):
            if i < len(variable_order):
                d_subplots[variable_order[i]] = (r, c)
            i += 1

    return d_subplots
