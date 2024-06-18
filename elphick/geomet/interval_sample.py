from pathlib import Path
from typing import Optional, Literal

import numpy as np
import pandas as pd

from elphick.geomet import MassComposition
import plotly.graph_objects as go


class IntervalSample(MassComposition):
    """
    A class to represent a sample of data with an interval index.
    This exposes methods to split the sample by a partition definition.
    """

    def __init__(self,
                 data: Optional[pd.DataFrame] = None,
                 name: Optional[str] = None,
                 moisture_in_scope: bool = True,
                 mass_wet_var: Optional[str] = None,
                 mass_dry_var: Optional[str] = None,
                 moisture_var: Optional[str] = None,
                 component_vars: Optional[list[str]] = None,
                 composition_units: Literal['%', 'ppm', 'ppb'] = '%',
                 components_as_symbols: bool = True,
                 ranges: Optional[dict[str, list]] = None,
                 config_file: Optional[Path] = None):
        super().__init__(data=data, name=name, moisture_in_scope=moisture_in_scope,
                         mass_wet_var=mass_wet_var, mass_dry_var=mass_dry_var,
                         moisture_var=moisture_var, component_vars=component_vars,
                         composition_units=composition_units, components_as_symbols=components_as_symbols,
                         ranges=ranges, config_file=config_file)

    def split_by_partition(self, partition_definition, name_1: str, name_2: str):
        """
        Split the sample into two samples based on the partition definition.
        :param partition_definition: A function that takes a data frame and returns a boolean series.
        :param name_1: The name of the first sample.
        :param name_2: The name of the second sample.
        :return: A tuple of two IntervalSamples.
        """
        raise NotImplementedError('Not yet ready...')
        mask = partition_definition(self._data)
        sample_1 = self._data[mask]
        sample_2 = self._data[~mask]
        return IntervalSample(sample_1, name_1), IntervalSample(sample_2, name_2)

    def is_2d_grid(self):
        """
        Check if the sample is a 2d grid.
        :return: True if the sample has 2 levels of intervals, False otherwise.
        """
        res = False
        if self.mass_data is not None and self.mass_data.index.nlevels >= 2:
            # get the type of the index levels
            level_types = [type(level) for level in self.mass_data.index.levels]
            # get the counts of each type
            level_counts = {level_type: level_types.count(level_type) for level_type in set(level_types)}
            # check if there are 2 levels of intervals
            res = level_counts.get(pd.Interval, 0) == 2

        return res

    @property
    def is_rectilinear_grid(self):
        """If rectilinear we can plot with a simple heatmap"""
        res = False
        if self.mass_data is not None and self._mass_data.index.nlevels >= 2:
            # Get the midpoints of the intervals for X and Y
            x_midpoints = self.mass_data.index.get_level_values(0).mid
            y_midpoints = self.mass_data.index.get_level_values(1).mid

            # Get unique midpoints for X and Y
            unique_x_midpoints = set(x_midpoints)
            unique_y_midpoints = set(y_midpoints)

            # Check if the grid is full (i.e., no steps in the lines that define the grid edges)
            # todo: fix this logic - it is not correct
            if len(unique_x_midpoints) == len(x_midpoints) and len(unique_y_midpoints) == len(y_midpoints):
                res = True
        return res

    def plot_heatmap(self, components: list[str], **kwargs):
        """
        Plot the sample as a heatmap.
        :param components: The list of components to plot.
        :param kwargs: Additional keyword arguments to pass to the plot method.
        :return: The axis with the plot.
        """
        # if not self.is_rectilinear_grid:
        #     raise ValueError('The sample is not a rectilinear grid.')

        # convert IntervalIndex to nominal values df.index = df.index.map(lambda x: x.mid)

        x_label = self.mass_data.index.names[1]
        y_label = self.mass_data.index.names[0]
        z_label = self.mass_data.columns[0]

        # create a pivot table for the heatmap
        pivot_df = self.mass_data[components].copy().unstack()

        # Get the midpoints of the intervals for X and Y
        x_midpoints = [interval.mid for interval in self.mass_data.index.get_level_values(x_label)]
        y_midpoints = [interval.mid for interval in self.mass_data.index.get_level_values(y_label)]

        # Get interval edges for x and y axes
        x_edges = self._get_unique_edges(self.mass_data.index.get_level_values(x_label))
        y_edges = self._get_unique_edges(self.mass_data.index.get_level_values(y_label))

        # Create hover text
        hover_text = [[f"{x_label}: {x_mid}, {y_label}: {y_mid}, {z_label}: {z_val}"
                       for x_mid, z_val in zip(x_midpoints, z_values)]
                      for y_mid, z_values in zip(y_midpoints, pivot_df.values)]

        # plot the heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_df.values,
            x=x_edges,
            y=y_edges,
            text=hover_text,
            hoverinfo='text'))

        # update the layout to use logarithmic x-axis
        fig.update_layout(yaxis_type="log")
        # set the title and x and y labels dynamically
        fig.update_layout(title=f'{self.name} Heatmap',
                          xaxis_title=self.mass_data.index.names[1],
                          yaxis_title=self.mass_data.index.names[0])

        return fig

    @staticmethod
    def _get_unique_edges(interval_index):
        # Get the left and right edges of the intervals
        left_edges = interval_index.left.tolist()
        right_edges = interval_index.right.tolist()

        # Concatenate the two lists
        all_edges = left_edges + right_edges

        # Get the unique edges
        unique_edges = np.unique(all_edges)

        return unique_edges
