from __future__ import annotations
from pathlib import Path
from typing import Optional, Literal, Callable, Union, Iterable, TYPE_CHECKING

import numpy as np
import pandas as pd
from pandas import IntervalIndex
from pandas.core.indexes.frozen import FrozenList

import plotly.graph_objects as go
import plotly.express as px

from elphick.geomet.utils.amenability import amenability_index
from elphick.geomet.utils.interp import mass_preserving_interp
from elphick.geomet.utils.pandas import MeanIntervalIndex, weight_average, calculate_recovery, calculate_partition, \
    cumulate, mass_to_composition
from elphick.geomet.utils.sampling import random_int

from elphick.geomet.base import MassComposition

if TYPE_CHECKING:
    from elphick.geomet.flowsheet.stream import Stream


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

        # manage the interval indexes
        if self.data is not None:
            self.data = self._create_interval_indexes(data)

    def _create_interval_indexes(self, data: pd.DataFrame) -> pd.DataFrame:
        original_indexes = data.index.names
        interval_indexes = []
        for pair in self.config['intervals']['suffixes']:
            if data.index.names != FrozenList([None]):
                suffix_candidates: dict = {n: n.split('_')[-1].lower() for n in data.index.names}
                suffixes: dict = {k: v for k, v in suffix_candidates.items() if v in pair}
                if suffixes:
                    data.reset_index(list(suffixes.keys()), inplace=True)
                    num_interval_indexes: int = int(len(suffixes.keys()) / 2)
                    for i in range(0, num_interval_indexes):
                        keys = list(suffixes.keys())[i: i + 2]
                        base_name: str = '_'.join(keys[0].split('_')[:-1])
                        index = IntervalIndex.from_arrays(left=data[keys[0]], right=data[keys[1]],
                                                          closed=self.config['intervals']['closed'])
                        index.name = base_name
                        # left and right names are only preserved for a single interval index.
                        # when a multiindex is used, the names are not preserved.
                        index.left.name = keys[0].split('_')[-1]
                        index.right.name = keys[1].split('_')[-1]
                        interval_indexes.append(index)

                        # drop the index columns from the dataframe columns
                        data.drop(columns=keys, inplace=True)

        if interval_indexes:
            new_indexes = {}  # Use dict to preserve order and uniqueness
            # we need to set the index to include the new interval index, but respect the order of the original.
            for i in original_indexes:
                if i.split('_')[0] not in [ii.name for ii in interval_indexes]:
                    new_indexes[i] = data.index.get_level_values(i)
                else:
                    # Find the corresponding interval index and append it to the new_indexes list
                    for ii in interval_indexes:
                        if ii.name == i.split('_')[0]:
                            new_indexes[ii.name] = ii
                            break

            if len(new_indexes) > 1:
                data.index = pd.MultiIndex.from_frame(pd.DataFrame(new_indexes.values()).T, names=new_indexes.keys())
            else:
                data.index = list(new_indexes.values())[0]

        return data

    def split_by_partition(self, partition_definition, name_1: str = 'preferred', name_2: str = 'complement'):
        """
        Split the sample into two samples based on the partition definition.

        .. math::
            K = \\frac{{m_{preferred}}}{{m_{feed}}}

        :param partition_definition: A function that takes a data frame and returns a boolean series with a
         range [0, 1].
        :param name_1: The name of the first sample.
        :param name_2: The name of the second sample.
        :return: A tuple of two IntervalSamples.
        """
        if not isinstance(partition_definition, Callable):
            raise TypeError("The definition is not a callable function")
        if 'dim' not in partition_definition.keywords.keys():
            raise NotImplementedError("The callable function passed does not have a dim")

        dim = partition_definition.keywords['dim']
        partition_definition.keywords.pop('dim')

        # get the mean of the intervals - the geomean if the interval is called size
        index = self.mass_data.index.get_level_values(dim)
        # check the index is an interval index
        if not isinstance(index, pd.IntervalIndex):
            raise ValueError(f"The index is not an IntervalIndex.  The index is {type(index)}")
        index = MeanIntervalIndex(index)
        x = index.mean

        self.to_stream()
        self: Stream

        pn: pd.Series = pd.Series(partition_definition(x), name='K', index=index)
        sample_1 = self.create_congruent_object(name=name_1).to_stream()
        sample_1.mass_data = self.mass_data.copy().multiply(pn, axis=0)
        sample_1.set_nodes([self.nodes[1], random_int()])
        sample_2 = self.create_congruent_object(name=name_2)
        sample_2.mass_data = self.mass_data.copy().multiply((1 - pn), axis=0)
        sample_2.set_nodes([self.nodes[1], random_int()])

        return sample_1, sample_2

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

    def ideal_incremental_separation(self, discard_from: Literal["lowest", "highest"] = "lowest") -> pd.DataFrame:
        """Incrementally separate a fractionated sample.

        This method sorts by the provided direction prior to incrementally removing and discarding the first fraction
         (of the remaining fractions) and recalculating the mass-composition and recovery of the portion remaining.
         This is equivalent to incrementally applying a perfect separation (partition) at every interval edge.

        This method is only applicable to a 1D object where the single dimension is a pd.Interval type.

        See also: ideal_incremental_composition, ideal_incremental_recovery.

        Args:
            discard_from: Defines the discarded direction.  discard_from = "lowest" will discard the lowest value
             first, then the next lowest, etc.

        Returns:
            A pandas DataFrame
        """
        self._check_one_dim_interval()

        sample: pd.DataFrame = self.data

        is_decreasing: bool = sample.index.is_monotonic_decreasing
        if discard_from == "lowest":
            sample.sort_index(ascending=True, inplace=True)
            new_index: pd.Index = pd.Index(sample.index.left)
        else:
            sample.sort_index(ascending=False, inplace=True)
            new_index: pd.Index = pd.Index(sample.index.right)
        new_index.name = f"{sample.index.name}_cut-point"

        aggregated_chunks: list = []
        recovery_chunks: list = []
        head: pd.Series = sample.pipe(weight_average)

        for i, indx in enumerate(sample.index):
            tmp_composition: pd.DataFrame = sample.iloc[i:, :].pipe(weight_average).to_frame().T
            aggregated_chunks.append(tmp_composition)
            recovery_chunks.append(tmp_composition.pipe(calculate_recovery, df_ref=head.to_frame().T))

        res_composition: pd.DataFrame = pd.concat(aggregated_chunks).assign(attribute="composition").set_index(
            new_index)
        res_recovery: pd.DataFrame = pd.concat(recovery_chunks).assign(attribute="recovery").set_index(
            new_index)

        if is_decreasing:
            res_composition.sort_index(ascending=False, inplace=True)
            res_recovery.sort_index(ascending=False, inplace=True)

        res: pd.DataFrame = pd.concat([res_composition, res_recovery]).reset_index().set_index(
            [new_index.name, 'attribute'])

        return res

    def _check_one_dim_interval(self):
        if self.mass_data.index.ndim > 1:
            raise NotImplementedError(f"This object is {self.mass_data.index.ndim} dimensional. "
                                      f"Only 1D interval objects are valid")
        index_var: str = self.mass_data.index.name
        if not isinstance(self.mass_data.index, pd.IntervalIndex):
            raise NotImplementedError(f"The {index_var} of this object is not a pd.Interval. "
                                      f" Only 1D interval objects are valid")

    def ideal_incremental_composition(self, discard_from: Literal["lowest", "highest"] = "lowest") -> pd.DataFrame:
        """Incrementally separate a fractionated sample.

        This method sorts by the provided direction prior to incrementally removing and discarding the first fraction
         (of the remaining fractions) and recalculating the mass-composition of the portion remaining.
         This is equivalent to incrementally applying a perfect separation (partition) at every interval edge.

        This method is only applicable to a 1D object where the single dimension is a pd.Interval type.

        See also: ideal_incremental_separation, ideal_incremental_recovery.

        Args:
            discard_from: Defines the discarded direction.  discard_from = "lowest" will discard the lowest value
             first, then the next lowest, etc.

        Returns:
            A pandas DataFrame
        """
        df: pd.DataFrame = self.ideal_incremental_separation(discard_from=discard_from).query(
            'attribute=="composition"').droplevel('attribute')
        return df

    def ideal_incremental_recovery(self, discard_from: Literal["lowest", "highest"] = "lowest",
                                   apply_closure: bool = True) -> pd.DataFrame:
        """Incrementally separate a fractionated sample.

        This method sorts by the provided direction prior to incrementally removing and discarding the first fraction
         (of the remaining fractions) and recalculating the recovery of the portion remaining.
         This is equivalent to incrementally applying a perfect separation (partition) at every interval edge.

        This method is only applicable to a 1D object where the single dimension is a pd.Interval type.

        See also: ideal_incremental_separation, ideal_incremental_composition.

        Args:
            discard_from: Defines the discarded direction.  discard_from = "lowest" will discard the lowest value
             first, then the next lowest, etc.
            apply_closure: If True, Add the missing record (zero recovery) that closes the recovery envelope.

        Returns:
            A pandas DataFrame
        """
        columns_to_drop: list[str] = ['mass_wet', 'H2O'] if self.moisture_in_scope else []
        df: pd.DataFrame = self.ideal_incremental_separation(discard_from=discard_from).query(
            'attribute=="recovery"').droplevel('attribute').rename(columns={'mass_dry': 'mass'}).drop(
            columns=columns_to_drop)
        if apply_closure:
            # add zero recovery record to close the envelope.
            indx = np.inf if df.index.min() == 0.0 else 0.0
            indx_name: str = df.index.name
            df = pd.concat([df, pd.Series(0, index=df.columns, name=indx).to_frame().T]).sort_index(ascending=True)
            df.index.name = indx_name
        return df

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

    def plot_intervals(self,
                       variables: list[str],
                       cumulative: bool = True,
                       direction: str = 'descending',
                       show_edges: bool = True,
                       min_x: Optional[float] = None) -> go.Figure:
        """Plot "The Grade-Tonnage" curve.

        Mass and grade by bins for a cut-off variable.

        Args:
            variables: List of variables to include in the plot
            cumulative: If True, the results are cumulative weight averaged.
            direction: 'ascending'|'descending', if cumulative is True, the direction of accumulation
            show_edges: If True, show the edges on the plot.  Applicable to cumulative plots only.
            min_x: Optional minimum value for the x-axis, useful to set reasonable visual range with a log
            scaled x-axis when plotting size data
        """

        res: pd.DataFrame = self.data[variables]

        plot_kwargs: dict = dict(line_shape='vh')
        if cumulative:
            res = self.mass_data.pipe(cumulate, direction=direction).pipe(mass_to_composition)
            plot_kwargs = dict(line_shape='spline')

        interval_data: pd.DataFrame = res

        # Get the first IntervalIndex - TODO: specify or check...
        interval_index: Optional[pd.IntervalIndex] = None
        for level in range(interval_data.index.nlevels):
            if isinstance(interval_data.index.get_level_values(level), pd.IntervalIndex):
                interval_index = interval_data.index.get_level_values(level)
                break
        if interval_index is None:
            raise ValueError("No IntervalIndex found in the index levels")
        left_name: str = interval_index.left.name if interval_index.left.name else 'left'
        right_name: str = interval_index.right.name if interval_index.right.name else 'right'
        left: pd.Series = pd.Series(interval_index.left, name=left_name, index=interval_index)
        right: pd.Series = pd.Series(interval_index.right, name=right_name, index=interval_index)
        df_intervals = pd.concat([left, right, interval_data], axis='columns')
        x_var: str = interval_data.index.name
        if not cumulative:
            # append on the largest fraction right edge for display purposes
            is_ascending: bool = interval_index.is_monotonic_increasing
            df_end: pd.DataFrame = df_intervals.loc[df_intervals.index.max(), :].to_frame().T
            df_end[left_name] = df_end[right_name]
            df_end[right_name] = np.inf
            df = pd.concat([df_end.reset_index(drop=True), df_intervals], axis='index')
            df[interval_data.index.name] = df[left_name]
            df = df.sort_values(by=interval_data.index.name, ascending=is_ascending)
        else:
            if direction == 'ascending':
                x_var = right_name
            elif direction == 'descending':
                x_var = left_name
            df = df_intervals

        if res.index.name.lower() == 'size':
            if not min_x:
                min_x = interval_data.index.min().right / 2.0
            # set zero to the minimum x value (for display only) to enable the tooltips on that point.
            df.loc[df[x_var] == df[x_var].min(), x_var] = min_x
            hover_data = {'component': True,  # add other column, default formatting
                          x_var: ':.3f',  # add other column, customized formatting
                          'value': ':.2f'
                          }
            plot_kwargs = {**plot_kwargs,
                           **dict(log_x=True,
                                  range_x=[min_x, interval_data.index.max().right],
                                  hover_data=hover_data)}

        df = df[[x_var] + variables].melt(id_vars=[x_var], var_name='component')

        if cumulative and show_edges:
            plot_kwargs['markers'] = True

        fig = px.line(df, x=x_var, y='value', facet_row='component', **plot_kwargs)
        fig.for_each_annotation(lambda a: a.update(text=a.text.replace("component=", "")))
        fig.update_yaxes(matches=None)
        fig.update_layout(title=self.name)

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

    def plot_grade_recovery(self, target_analyte,
                            discard_from: Literal["lowest", "highest"] = "lowest",
                            title: Optional[str] = None,
                            ) -> go.Figure:
        """The grade-recovery plot.

        The grade recovery curve is generated by assuming an ideal separation (for the chosen property, or dimension)
        at each fractional interval.  It defines the theoretical maximum performance, which can only be improved if
        liberation is improved by comminution.

        This method is only applicable to a 1D object where the single dimension is a pd.Interval type.

        Args:
            target_analyte: The analyte of value.
            discard_from: Defines the discarded direction.  discard_from = "lowest" will discard the lowest value
             first, then the next lowest, etc.
            title: Optional plot title

        Returns:
            A plotly.GraphObjects figure
        """
        title = title if title is not None else 'Ideal Grade - Recovery'
        cols_to_drop: list[str] = ['mass_wet', 'mass_dry', 'H2O'] if self.moisture_in_scope else ['mass_dry']

        df: pd.DataFrame = self.ideal_incremental_separation(discard_from=discard_from)
        df_recovery: pd.DataFrame = df.loc[(slice(None), 'recovery'), [target_analyte, 'mass_dry']].droplevel(
            'attribute').rename(
            columns={'mass_dry': 'Yield', target_analyte: f"{target_analyte}_recovery"})
        df_composition: pd.DataFrame = df.loc[(slice(None), 'composition'), :].droplevel('attribute').drop(
            columns=cols_to_drop)

        df_plot: pd.DataFrame = pd.concat([df_recovery, df_composition], axis=1).reset_index()
        fig = px.line(df_plot, x=target_analyte,
                      y=f"{target_analyte}_recovery",
                      hover_data=df_plot.columns,
                      title=title)
        # fig.update_layout(xaxis_title=f"Grade of {target_analyte}", yaxis_title=f"Recovery of {target_analyte}",
        #                   title=title)

        return fig

    def plot_amenability(self, target_analyte: str,
                         discard_from: Literal["lowest", "highest"] = "lowest",
                         gangue_analytes: Optional[str] = None,
                         title: Optional[str] = None,
                         ) -> go.Figure:
        """The yield-recovery plot.

        The yield recovery curve provides an understanding of the amenability of a sample.

        This method is only applicable to a 1D object where the single dimension is a pd.Interval type.

        Args:
            target_analyte: The analyte of value.
            discard_from: Defines the discarded direction.  discard_from = "lowest" will discard the lowest value
             first, then the next lowest, etc.
            gangue_analytes: The analytes to be rejected
            title: Optional plot title

        Returns:
            A plotly.GraphObjects figure
        """
        title = title if title is not None else 'Amenability Plot'
        df: pd.DataFrame = self.ideal_incremental_recovery(discard_from=discard_from)
        amenability_indices: pd.Series = amenability_index(df, col_target=target_analyte, col_mass_recovery='mass')

        analytes = [col for col in df.columns if col != "mass"] if gangue_analytes is None else [
            target_analyte + gangue_analytes]

        mass_rec: pd.DataFrame = df["mass"]
        df = df[analytes]

        fig = go.Figure()
        for analyte in analytes:
            fig.add_trace(
                go.Scatter(x=mass_rec, y=df[analyte], mode="lines",
                           name=f"{analyte} ({round(amenability_indices[analyte], 2)})",
                           customdata=df.index.values,
                           hovertemplate='<b>Recovery: %{y:.3f}</b><br>Cut-point: %{customdata:.3f} '))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name='y=x',
                                 line=dict(shape='linear', color='gray', dash='dash'),
                                 ))
        fig.update_layout(xaxis_title='Yield (Mass Recovery)', yaxis_title='Recovery', title=title,
                          hovermode='x')
        return fig

    def calculate_partition(self, preferred: 'MassComposition') -> pd.DataFrame:
        """Calculate the partition number (K) [0, 1] of the preferred stream relative to self

        .. math::
            K = \\frac{{m_{preferred}}}{{m_{feed}}}

        """
        self._check_one_dim_interval()
        return calculate_partition(df_feed=self.data, df_preferred=preferred.data,
                                   col_mass_dry='mass_dry')

    def resample_1d(self, interval_edges: Union[Iterable, int],
                    precision: Optional[int] = None,
                    include_original_edges: bool = False) -> 'IntervalSample':
        """Resample a 1D fractional dim/index

        Args:
            interval_edges: The values of the new grid (interval edges).  If an int, will up-sample by that factor, for
             example the value of 10 will automatically define edges that create 10 x the resolution (up-sampled).
            precision: Optional integer for the number of decimal places to round the grid values to.
            include_original_edges: If True include the original edges in the grid.

        Returns:
            A new IntervalSample object interpolated onto the new grid
        """

        # TODO: add support for supplementary variables

        # test the index contains a single interval index
        self._check_one_dim_interval()

        df_upsampled: pd.DataFrame = mass_preserving_interp(self.data,
                                                            interval_edges=interval_edges, precision=precision,
                                                            include_original_edges=include_original_edges)

        obj: IntervalSample = IntervalSample(df_upsampled, name=self.name, moisture_in_scope=False)
        obj.status.ranges = self.status.ranges
        return obj
