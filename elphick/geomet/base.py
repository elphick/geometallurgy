import copy
import inspect
import logging
import re
from abc import ABC
from pathlib import Path
from typing import Optional, Union, Literal, TypeVar, TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from elphick.geomet.config import read_yaml
from elphick.geomet.utils.components import get_components, is_compositional
from elphick.geomet.utils.moisture import solve_mass_moisture
from elphick.geomet.utils.pandas import mass_to_composition, composition_to_mass, composition_factors
from elphick.geomet.utils.sampling import random_int
from elphick.geomet.utils.timer import log_timer
from .config.config_read import get_column_config
from .plot import parallel_plot, comparison_plot
import plotly.express as px
import plotly.graph_objects as go

if TYPE_CHECKING:
    from elphick.geomet.flowsheet.stream import Stream

# generic type variable, used for type hinting, to indicate that the type is a subclass of MassComposition
MC = TypeVar('MC', bound='MassComposition')


def filter_kwargs(cls, **kwargs):
    valid_params = inspect.signature(cls.__init__).parameters
    res = {k: v for k, v in kwargs.items() if k in valid_params}
    return res


class MassComposition(ABC):
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
        """

        Args:
            data: The input data
            name: The name of the sample
            moisture_in_scope: Whether the moisture is in scope.  If False, only dry mass is processed.
            mass_wet_var: The name of the wet mass column
            mass_dry_var: The name of the dry mass column
            moisture_var: The name of the moisture column
            component_vars: The names of the chemical columns
            components_as_symbols: If True, convert the composition variables to symbols, e.g. Fe
            ranges: The range of valid data for each column in the data
            config_file: The configuration file
        """

        self._logger = logging.getLogger(name=self.__class__.__name__)

        if config_file is None:
            config_file = Path(__file__).parent / './config/mc_config.yml'
        self.config = read_yaml(config_file)

        self.name: str = name
        self.moisture_in_scope: bool = moisture_in_scope
        self.mass_wet_var: Optional[str] = mass_wet_var
        self.mass_dry_var: str = mass_dry_var
        self.moisture_var: Optional[str] = moisture_var
        self.component_vars: Optional[list[str]] = component_vars  # TODO: check if this is redundant and remove.
        self.composition_units: Literal['%', 'ppm', 'ppb'] = composition_units
        self.composition_factor: int = composition_factors[composition_units]
        self.components_as_symbols: bool = components_as_symbols

        self._mass_data: Optional[pd.DataFrame] = None
        self._supplementary_data = None
        self._aggregate = None

        # set the data
        self.data = data

        # add the OOR status object
        self.status = OutOfRangeStatus(self, ranges)

    @property
    @log_timer
    def data(self) -> Optional[pd.DataFrame]:
        if self._mass_data is not None:
            # convert chem mass to composition
            mass_comp_data = mass_to_composition(self._mass_data,
                                                 mass_wet=self.mass_wet_var, mass_dry=self.mass_dry_var,
                                                 moisture_column_name='H2O' if self.components_as_symbols else (
                                                     self.moisture_var if self.moisture_var is not None else 'h2o'),
                                                 component_columns=self.composition_columns,
                                                 composition_units=self.composition_units)

            # append the supplementary vars
            return pd.concat([mass_comp_data, self._supplementary_data], axis=1)
        return None

    @data.setter
    @log_timer
    def data(self, value):
        if value is not None:
            # Convert column names to symbols if components_as_symbols is True
            if self.components_as_symbols:
                symbol_dict = is_compositional(value.columns, strict=False)
                value.columns = [symbol_dict.get(col, col) for col in value.columns]

            # the config provides regex search keys to detect mass and moisture columns if they are not specified.
            mass_totals = self._solve_mass(value)
            composition, supplementary_data = self._get_non_mass_data(value)

            self._supplementary_data = supplementary_data

            self.mass_data = composition_to_mass(pd.concat([mass_totals, composition], axis=1),
                                                 mass_wet=self.mass_wet_var, mass_dry=self.mass_dry_var,
                                                 moisture_column_name=self.moisture_column,
                                                 component_columns=composition.columns,
                                                 composition_units=self.composition_units)
            self._logger.debug(f"Data has been set.")

        else:
            self._mass_data = None

    @property
    def mass_data(self):
        return self._mass_data

    @mass_data.setter
    def mass_data(self, value):
        self._mass_data = value
        # Recalculate the aggregate whenever the data changes
        self.aggregate = self.weight_average()

    @property
    def aggregate(self) -> pd.DataFrame:
        if self._aggregate is None and self._mass_data is not None:
            self._aggregate = self.weight_average()
        return self._aggregate

    @aggregate.setter
    def aggregate(self, value):
        self._aggregate = value

    @property
    def variable_map(self) -> Optional[dict[str, str]]:
        """A map from lower case standard names to the actual column names"""
        if self._mass_data is not None:
            existing_columns = list(self._mass_data.columns)
            res = {}
            if self.moisture_in_scope and self.mass_wet_var in existing_columns:
                res['mass_wet'] = self.mass_wet_var
            if self.mass_dry_var in existing_columns:
                res['mass_dry'] = self.mass_dry_var
            if self.moisture_in_scope:
                res['moisture'] = self.moisture_var
                if self.components_as_symbols:
                    res['moisture'] = is_compositional([self.moisture_var], strict=False).get(self.moisture_var,
                                                                                              self.moisture_var)
            if self.composition_columns:
                for col in self.composition_columns:
                    res[col.lower()] = col
            return res
        return None

    @property
    def mass_columns(self) -> Optional[list[str]]:
        if self._mass_data is not None:
            existing_columns = list(self._mass_data.columns)
            res = []
            if self.moisture_in_scope and self.mass_wet_var in existing_columns:
                res.append(self.mass_wet_var)
            if self.mass_dry_var in existing_columns:
                res.append(self.mass_dry_var)
            return res
        return None

    @property
    def moisture_column(self) -> Optional[str]:
        res = 'h2o'
        if self.moisture_in_scope:
            res = self.moisture_var
            if self.components_as_symbols:
                res = is_compositional([res], strict=False).get(res, res)
        return res

    @property
    def composition_columns(self) -> Optional[list[str]]:
        res = None
        if self._mass_data is not None:
            if self.moisture_in_scope:
                res = list(self._mass_data.columns)[2:]
            else:
                res = list(self._mass_data.columns)[1:]
        return res

    @property
    def supplementary_columns(self) -> Optional[list[str]]:
        res = None
        if self._supplementary_data is not None:
            res = list(self._supplementary_data.columns)
        return res

    @property
    def data_columns(self) -> list[str]:
        return [col for col in
                (self.mass_columns + [self.moisture_column] + self.composition_columns + self.supplementary_columns) if
                col is not None]

    def balance_composition(self) -> MC:
        """Balance the composition data

        For records where the component mass exceeds the dry mass, the component masses are reduced proportionally
        to equal the dry mass.  Records where the component mass is less than the dry mass are left unchanged.

        """
        if self._mass_data is not None:
            # calculate the ratio of the sum of the components to the dry mass
            ratio = self._mass_data[self.composition_columns].sum(axis=1) / self._mass_data[self.mass_dry_var]
            if ratio.max() <= 1.0:
                return self
            epsilon = 1e-6
            # add a small value to the ratio to avoid component sums marginally over 100.0
            ratio[ratio > 1.0] = ratio[ratio > 1.0] + epsilon
            # to avoid reducing compliant records, clip the ratio at the lower side to 1.0
            ratio = ratio.clip(lower=1.0)
            # apply the ratio to the components
            self._mass_data[self.composition_columns] = self._mass_data[self.composition_columns].div(ratio, axis=0)

        return self

    def set_moisture(self, moisture: Union[pd.Series, float, int], mass_to_adjust: Literal['wet', 'dry'] = 'wet') -> MC:
        """Set the moisture to the specified value

        A convenience method for an mc object that modifies the concrete mass to deliver the specified moisture.

        Args:
            moisture: The moisture value to set.  Can be a constant or series.
            mass_to_adjust: The mass to adjust, either 'wet' or 'dry'.

        Returns:

        """

        if not self.moisture_in_scope:
            raise AssertionError("This method is not applicable unless moisture_in_scope property is True.")

        if isinstance(moisture, float) or isinstance(moisture, int):
            # create a series with the same index as the mass data
            moisture = pd.Series(float(moisture), index=self._mass_data.index)
        elif not isinstance(moisture, pd.Series):
            raise TypeError(f"moisture must be a float or a pd.Series, not {type(moisture)}")

        if mass_to_adjust == 'wet':
            self._mass_data[self.mass_wet_var] = solve_mass_moisture(mass_dry=self._mass_data[self.mass_dry_var],
                                                                     moisture=moisture)
        elif mass_to_adjust == 'dry':
            self._mass_data[self.mass_dry_var] = solve_mass_moisture(mass_wet=self._mass_data[self.mass_wet_var],
                                                                     moisture=moisture)
        else:
            raise ValueError(f"mass_to_adjust must be 'wet' or 'dry', not {mass_to_adjust}")

        return self

    def clip_composition(self, ranges: Optional[dict[str, list[float]]] = None) -> MC:
        """Clip the components

        Clip to the components to within the range provided or the default range for each component.
        This method does not clip moisture - see set_moisture and solve_moisture for that.

        Args:
            ranges: An optional dict defining a list of [lo, hi] floats for each component.  If not provided,
            the default range from the config file will be used.

        Returns:
            The object with clipped composition.
        """

        # load the default ranges from the config file
        component_ranges: dict = self._get_component_ranges(ranges)

        # define a small value to ensure the clipped values lie marginally inside the specified range.
        epsilon: float = 0.0  # 1.0e-05
        # clip the components
        affected_indexes = set()
        for component, component_range in component_ranges.items():
            before_clip = self._mass_data[component].copy()
            # define the component mass that aligns with the lower and upper bounds
            component_mass_limits = self._mass_data[self.mass_dry_var].values[:, np.newaxis] * np.array(
                component_range) / self.composition_factor
            # apply the clip to the mass data
            self._mass_data[component] = self._mass_data[component].clip(lower=component_mass_limits[:, 0] + epsilon,
                                                                         upper=component_mass_limits[:, 1] - epsilon)
            affected_indexes.update(self._mass_data.index[before_clip != self._mass_data[component]])

        # log the action, including the first 50 indexes affected
        affected_indexes_list = sorted(affected_indexes)[:50]
        self._logger.info(
            f"{len(affected_indexes)} records where composition has been clipped to the range: {component_ranges}."
            f" Affected indexes (first 50): {affected_indexes_list}")

        return self

    def plot_parallel(self, color: Optional[str] = None,
                      vars_include: Optional[list[str]] = None,
                      vars_exclude: Optional[list[str]] = None,
                      title: Optional[str] = None,
                      include_dims: Optional[Union[bool, list[str]]] = True,
                      plot_interval_edges: bool = False) -> go.Figure:
        """Create an interactive parallel plot

        Useful to explore multidimensional data like mass-composition data

        Args:
            color: Optional color variable
            vars_include: Optional list of variables to include in the plot
            vars_exclude: Optional list of variables to exclude in the plot
            title: Optional plot title
            include_dims: Optional boolean or list of dimension to include in the plot.  True will show all dims.
            plot_interval_edges: If True, interval edges will be plotted instead of interval mid

        Returns:

        """

        if not title and hasattr(self, 'name'):
            title = self.name

        fig = parallel_plot(data=self.data, color=color, vars_include=vars_include, vars_exclude=vars_exclude,
                            title=title,
                            include_dims=include_dims, plot_interval_edges=plot_interval_edges)
        return fig

    def plot_comparison(self, other: MC,
                        color: Optional[str] = None,
                        vars_include: Optional[list[str]] = None,
                        vars_exclude: Optional[list[str]] = None,
                        facet_col_wrap: int = 3,
                        trendline: bool = False,
                        trendline_kwargs: Optional[dict] = None,
                        title: Optional[str] = None) -> go.Figure:
        """Create an interactive parallel plot

        Useful to compare the difference in component values between two objects.

        Args:
            other: the object to compare with self.
            color: Optional color variable
            vars_include: Optional List of variables to include in the plot
            vars_exclude: Optional List of variables to exclude in the plot
            trendline: If True and trendlines
            trendline_kwargs: Allows customising the trendline: ref: https://plotly.com/python/linear-fits/
            title: Optional plot title
            facet_col_wrap: The number of subplot columns per row.

        Returns:

        """
        df_self: pd.DataFrame = self.data.to_dataframe()
        df_other: pd.DataFrame = other.data.to_dataframe()

        if vars_include is not None:
            missing_vars = set(vars_include).difference(set(df_self.columns))
            if len(missing_vars) > 0:
                raise KeyError(f'var_subset provided contains variable not found in the data: {missing_vars}')
            df_self = df_self[vars_include]
        if vars_exclude:
            df_self = df_self[[col for col in df_self.columns if col not in vars_exclude]]
        df_other = df_other[df_self.columns]
        # Supplementary variables are the same for each stream and so will be unstacked.
        supp_cols: list[str] = self.supplementary_columns
        if supp_cols:
            df_self.set_index(supp_cols, append=True, inplace=True)
            df_other.set_index(supp_cols, append=True, inplace=True)

        index_names = list(df_self.index.names)
        cols = list(df_self.columns).copy()

        df_self = df_self[cols].assign(name=self.name).reset_index().melt(id_vars=index_names + ['name'])
        df_other = df_other[cols].assign(name=other.name).reset_index().melt(id_vars=index_names + ['name'])

        df_plot: pd.DataFrame = pd.concat([df_self, df_other])
        df_plot = df_plot.set_index(index_names + ['name', 'variable'], drop=True).unstack(['name'])
        df_plot.columns = df_plot.columns.droplevel(0)
        df_plot.reset_index(level=list(np.arange(-1, -len(index_names) - 1, -1)), inplace=True)

        # set variables back to standard order
        variable_order: dict = {col: i for i, col in enumerate(cols)}
        df_plot = df_plot.sort_values(by=['variable'], key=lambda x: x.map(variable_order))

        fig: go.Figure = comparison_plot(data=df_plot, x=self.name, y=other.name, facet_col_wrap=facet_col_wrap,
                                         color=color, trendline=trendline, trendline_kwargs=trendline_kwargs)
        fig.update_layout(title=title)
        return fig

    def plot_ternary(self, variables: list[str], color: Optional[str] = None,
                     title: Optional[str] = None) -> go.Figure:
        """Plot a ternary diagram

            variables: List of 3 components to plot
            color: Optional color variable
            title: Optional plot title

        """

        df = self.data
        vars_missing: list[str] = [v for v in variables if v not in df.columns]
        if vars_missing:
            raise KeyError(f'Variable/s not found in the dataset: {vars_missing}')

        cols: list[str] = variables
        if color is not None:
            cols.append(color)

        if color:
            fig = px.scatter_ternary(df[cols], a=variables[0], b=variables[1], c=variables[2], color=color)
        else:
            fig = px.scatter_ternary(df[cols], a=variables[0], b=variables[1], c=variables[2])

        if not title and hasattr(self, 'name'):
            title = self.name

        fig.update_layout(title=title)

        return fig

    def weight_average(self, group_by: Optional[str] = None) -> pd.DataFrame:

        if group_by is None:
            composition: pd.DataFrame = pd.DataFrame(
                self._mass_data[self.composition_columns].sum(axis=0) / self._mass_data[
                    self.mass_dry_var].sum() * self.composition_factor).T

            mass_sum = pd.DataFrame(self._mass_data[self.mass_columns].sum(axis=0)).T

            # Recalculate the moisture
            if self.moisture_in_scope:
                mass_sum[self.moisture_column] = solve_mass_moisture(mass_wet=mass_sum[self.mass_columns[0]],
                                                                     mass_dry=mass_sum[self.mass_columns[1]])

            # Create a DataFrame from the weighted averages
            weighted_averages_df = pd.concat([mass_sum, composition], axis=1)
        else:
            group_var: pd.Series = self._supplementary_data[group_by]
            weighted_averages_df = self._mass_data.groupby(group_var).apply(
                lambda x: pd.DataFrame(
                    x[self.composition_columns].sum(axis=0) / x[self.mass_dry_var].sum() * self.composition_factor).T)
            weighted_averages_df.index = weighted_averages_df.index.droplevel(-1)
            mass_sum = self._mass_data[self.mass_columns].groupby(group_var).sum()
            weighted_averages_df = pd.concat([mass_sum, weighted_averages_df], axis=1)
            if self.moisture_in_scope:
                weighted_averages_df.insert(loc=2, column=self.moisture_column, value=solve_mass_moisture(
                    mass_wet=mass_sum[self.mass_columns[0]],
                    mass_dry=mass_sum[self.mass_columns[1]]))

        return weighted_averages_df

    def _solve_mass(self, value) -> pd.DataFrame:
        """Solve mass_wet and mass_dry from the provided columns.

        Args:
            value: The input data with the column-names provided by the user\

        Returns: The mass data, with the columns mass_wet and mass_dry.  Only mass_dry if moisture_in_scope is False.
        """
        # Auto-detect columns if they are not provided
        mass_dry, mass_wet, moisture = self._extract_mass_moisture_columns(value)

        if mass_dry is None:
            if mass_wet is not None and moisture is not None:
                value[self.mass_dry_var] = solve_mass_moisture(mass_wet=mass_wet, moisture=moisture)
            else:
                msg = (f"mass_dry_var is not provided and cannot be calculated from mass_wet_var and moisture_var "
                       f"for {self.name}")
                self._logger.error(msg)
                raise ValueError(msg)

        if self.moisture_in_scope:
            if mass_wet is None:
                if mass_dry is not None and moisture is not None:
                    value[self.mass_wet_var] = solve_mass_moisture(mass_dry=mass_dry, moisture=moisture)
                else:
                    msg = (
                        f"mass_wet_var is not provided and cannot be calculated from mass_dry_var and moisture_var.  "
                        f"Consider specifying the mass_wet_var, mass_dry_var and moisture_var, or alternatively set "
                        f"moisture_in_scope to False for {self.name}")
                    self._logger.error(msg)
                    raise ValueError(msg)

            if moisture is None:
                if mass_wet is not None and mass_dry is not None:
                    value[self.moisture_var] = solve_mass_moisture(mass_wet=mass_wet, mass_dry=mass_dry)
                else:
                    msg = f"moisture_var is not provided and cannot be calculated from mass_wet_var and mass_dry_var."
                    self._logger.error(msg)
                    raise ValueError(msg)

            mass_totals: pd.DataFrame = value[[self.mass_wet_var, self.mass_dry_var]]
        else:
            mass_totals: pd.DataFrame = value[[self.mass_dry_var]]

        return mass_totals

        # Helper method to extract column

    def _extract_column(self, value, var_type):
        var = getattr(self, f"{var_type}_var")
        if var is None:
            var = next((col for col in value.columns if
                        re.search(self.config['vars'][var_type]['search_regex'], col,
                                  re.IGNORECASE)), self.config['vars'][var_type]['default_name'])
        return var

    def _extract_mass_moisture_columns(self, value):
        if self.mass_wet_var is None:
            self.mass_wet_var = self._extract_column(value, 'mass_wet')
        if self.mass_dry_var is None:
            self.mass_dry_var = self._extract_column(value, 'mass_dry')
        if self.moisture_var is None:
            self.moisture_var = self._extract_column(value, 'moisture')
        mass_wet = value.get(self.mass_wet_var)
        mass_dry = value.get(self.mass_dry_var)
        moisture = value.get(self.moisture_var)
        return mass_dry, mass_wet, moisture

    def _get_non_mass_data(self, value: Optional[pd.DataFrame]) -> (Optional[pd.DataFrame], Optional[pd.DataFrame]):
        """
        Get the composition data and supplementary data.  Extract only the composition columns specified,
        otherwise detect the compositional columns
        """
        composition = None
        supplementary = None
        if value is not None:
            if self.component_vars is None:
                non_mass_cols: list[str] = [col for col in value.columns if
                                            col not in [self.mass_wet_var, self.mass_dry_var, self.moisture_var, 'h2o',
                                                        'H2O', 'H2O']]
                component_cols: list[str] = get_components(value[non_mass_cols], strict=False)
            else:
                component_cols: list[str] = self.component_vars
            composition = value[component_cols]

            supplementary_cols: list[str] = [col for col in value.columns if
                                             col not in component_cols + [self.mass_wet_var, self.mass_dry_var,
                                                                          self.moisture_var, 'h2o',
                                                                          'H2O', 'H2O']]
            supplementary = value[supplementary_cols]

        return composition, supplementary

    def __deepcopy__(self, memo):
        # Create a new instance of our class
        new_obj = self.__class__()
        memo[id(self)] = new_obj

        # Copy each attribute
        for attr, value in self.__dict__.items():
            setattr(new_obj, attr, copy.deepcopy(value, memo))

        return new_obj

    def update_mass_data(self, value: pd.DataFrame):
        if self._mass_data is not None:
            self._mass_data = value
        if self._supplementary_data is not None:
            if self._supplementary_data.index.names != self._mass_data.index.names:  # if indexes have been dropped
                self._supplementary_data.index = self._mass_data.index
            self._supplementary_data = self._supplementary_data.loc[value.index]
        self.aggregate = self.weight_average()

    def filter_by_index(self, index: pd.Index):
        """Update the data by index"""
        if self._mass_data is not None:
            self._mass_data = self._mass_data.loc[index]
        if self._supplementary_data is not None:
            self._supplementary_data = self._supplementary_data.loc[index]
        self.aggregate = self.weight_average()

    def split(self,
              fraction: float,
              name_1: Optional[str] = None,
              name_2: Optional[str] = None,
              include_supplementary_data: bool = False) -> tuple['Stream', 'Stream']:
        """Split the object by mass

        A simple mass split maintaining the same composition

        Args:
            fraction: A constant in the range [0.0, 1.0]
            name_1: The name of the reference object created by the split
            name_2: The name of the complement object created by the split
            include_supplementary_data: Whether to inherit the supplementary variables

        Returns:
            tuple of two objects, the first with the mass fraction specified, the other the complement
        """

        # create_congruent_objects to preserve properties like constraints

        name_1 = name_1 if name_1 is not None else f"{self.name}_1"
        name_2 = name_2 if name_2 is not None else f"{self.name}_2"

        ref: MassComposition = self.create_congruent_object(name=name_1, include_mc_data=True,
                                                            include_supp_data=include_supplementary_data)
        ref.update_mass_data(self._mass_data * fraction)

        comp: MassComposition = self.create_congruent_object(name=name_2, include_mc_data=True,
                                                             include_supp_data=include_supplementary_data)
        comp.update_mass_data(self._mass_data * (1 - fraction))

        # Ensure self and other are Stream objects
        self._convert_to_stream(self)
        self._convert_to_stream(ref)
        self._convert_to_stream(comp)
        self: 'Stream'
        ref: 'Stream'
        comp: 'Stream'

        # create the relationships
        ref.nodes = [self.nodes[1], random_int()]
        comp.nodes = [self.nodes[1], random_int()]

        return ref, comp

    def add(self, other: MC, name: Optional[str] = None,
            include_supplementary_data: bool = False) -> 'Stream':
        """Add two objects together

        Args:
            other: The other object
            name: The name of the new object
            include_supplementary_data: Whether to include the supplementary data

        Returns:
            The new object
        """

        res: MC = self.create_congruent_object(name=name, include_mc_data=True,
                                               include_supp_data=include_supplementary_data)
        res.update_mass_data(self._mass_data + other._mass_data)

        # Ensure self and other are Stream objects
        self: 'Stream' = self.to_stream()
        other: 'Stream' = self._convert_to_stream(other)
        res: 'Stream' = self._convert_to_stream(res)

        # create the relationships
        other.nodes = [other.nodes[0], self.nodes[1]]
        res.nodes = [self.nodes[1], random_int()]

        return res

    def sub(self, other: MC, name: Optional[str] = None,
            include_supplementary_data: bool = False) -> 'Stream':
        """Subtract other from self

        Args:
            other: The other object
            name: The name of the new object
            include_supplementary_data: Whether to include the supplementary data

        Returns:
            The new object
        """
        res = self.create_congruent_object(name=name, include_mc_data=True,
                                           include_supp_data=include_supplementary_data)
        res.update_mass_data(self._mass_data - other._mass_data)

        # Ensure self and other are Stream objects
        self._convert_to_stream(self)
        self._convert_to_stream(other)
        self: 'Stream'
        other: 'Stream'

        # create the relationships
        res.nodes = [self.nodes[1], random_int()]

        return res

    def div(self, other: MC, name: Optional[str] = None,
            include_supplementary_data: bool = False) -> MC:
        """Divide two objects

        Divides self by other, with optional name of the returned object
        Args:
            other: the denominator (or reference) object
            name: name of the returned object
            include_supplementary_data: Whether to include the supplementary data

        Returns:

        """
        new_obj = self.create_congruent_object(name=name, include_mc_data=True,
                                               include_supp_data=include_supplementary_data)
        new_obj.update_mass_data(self._mass_data / other._mass_data)
        return new_obj

    def __str__(self):
        return f"{self.__class__.__name__}: {self.name}\n{self.aggregate.to_dict()}"

    def create_congruent_object(self, name: str,
                                include_mc_data: bool = False,
                                include_supp_data: bool = False) -> MC:
        """Create an object with the same attributes"""
        # Create a new instance of our class
        new_obj = self.__class__()

        # Copy each attribute
        for attr, value in self.__dict__.items():
            if attr == '_mass_data' and not include_mc_data:
                continue
            if attr == '_supplementary_data' and not include_supp_data:
                continue
            setattr(new_obj, attr, copy.deepcopy(value))
        new_obj.name = name
        return new_obj

    def __add__(self, other: MC) -> 'Stream':
        """Add two objects

        Perform the addition with the mass-composition variables only and then append any attribute variables.
        Presently ignores any attribute vars in other
        Args:
            other: object to add to self

        Returns:

        """
        return self.add(other, include_supplementary_data=True)

    def __sub__(self, other: MC) -> 'Stream':
        """Subtract the supplied object from self

        Perform the subtraction with the mass-composition variables only and then append any attribute variables.
        Args:
            other: object to subtract from self

        Returns:

        """
        return self.sub(other, include_supplementary_data=True)

    def to_stream(self) -> 'Stream':
        from elphick.geomet.flowsheet.stream import Stream  # Local import to avoid circular dependency
        if not isinstance(self, Stream):
            self.__class__ = type(self.__class__.__name__, (self.__class__, Stream), {})
            filtered_kwargs = filter_kwargs(self.__class__, **self.__dict__)
            filtered_kwargs['data'] = self.data
            Stream.__init__(self, **filtered_kwargs)  # Initialize Stream properties
        return self

    @staticmethod
    def _convert_to_stream(obj) -> 'Stream':
        from elphick.geomet.flowsheet.stream import Stream  # Local import to avoid circular dependency
        if not isinstance(obj, Stream):
            obj.__class__ = type(obj.__class__.__name__, (obj.__class__, Stream), {})
            filtered_kwargs = filter_kwargs(obj.__class__, **obj.__dict__)
            filtered_kwargs['data'] = obj.data
            Stream.__init__(obj, **filtered_kwargs)  # Initialize Stream properties
        return obj

    def __truediv__(self, other: MC) -> MC:
        """Divide self by the supplied object

        Perform the division with the mass-composition variables only and then append any attribute variables.
        Args:
            other: denominator object, self will be divided by this object

        Returns:

        """

        return self.div(other, include_supplementary_data=True)

    def __eq__(self, other):
        if isinstance(other, MassComposition):
            return self.__dict__ == other.__dict__
        return False

    @classmethod
    def from_mass_dataframe(cls, mass_df: pd.DataFrame,
                            mass_wet: Optional[str] = 'mass_wet',
                            mass_dry: str = 'mass_dry',
                            moisture_column_name: Optional[str] = None,
                            component_columns: Optional[list[str]] = None,
                            composition_units: Literal['%', 'ppm', 'ppb'] = '%',
                            **kwargs) -> MC:
        """
        Class method to create a MassComposition object from a mass dataframe.

        Args:
            mass_df: DataFrame with mass data.
            **kwargs: Additional arguments to pass to the MassComposition constructor.

        Returns:
            A new MassComposition object.
        """
        # Convert mass to composition using the function from the pandas module
        composition_df = mass_to_composition(mass_df, mass_wet=mass_wet, mass_dry=mass_dry,
                                             moisture_column_name=moisture_column_name,
                                             component_columns=component_columns,
                                             composition_units=composition_units)

        # Create a new instance of the MassComposition class
        return cls(data=composition_df, **kwargs)

    def query(self, expr: str, name: Optional[str] = None) -> MC:
        """Reduce the data by a query expression

        Args:
            expr: A pandas query expression
            name: name of the new object

        Returns:
            A new object with the reduced data

        """
        name = name if name is not None else self.name
        res = self.create_congruent_object(name=f"{name} ({expr})", include_mc_data=True,
                                           include_supp_data=True)
        filtered_index = self.data.query(expr).index
        res.update_mass_data(self._mass_data.loc[filtered_index])
        if res.supplementary_columns is not None:
            res._supplementary_data = self._supplementary_data.loc[filtered_index]

        return res

    def reset_index(self, index_name: str) -> MC:
        res = self.create_congruent_object(name=f"{self.name} (reset_index)", include_mc_data=True,
                                           include_supp_data=True)
        res.update_mass_data(self._mass_data.reset_index(level=index_name, drop=True))
        if res.supplementary_columns is not None:
            res._supplementary_data = self._supplementary_data.reset_index(level=index_name, drop=False)
        else:
            res._supplementary_data = pd.DataFrame(index=self._mass_data.index, columns=[index_name],
                                                   data=self._mass_data.index.get_level_values(index_name))

        return res

    def _get_component_ranges(self, ranges: dict[str, list]) -> dict[str, list]:

        d_ranges: dict = get_column_config(config_dict=self.config, var_map=self.variable_map,
                                           config_key='range')
        # filter to include only components
        d_ranges = {k: v for k, v in d_ranges.items() if k in self.composition_columns}

        # modify the default dict based on any user passed constraints
        if ranges:
            for k, v in ranges.items():
                d_ranges[k] = v

        return d_ranges


class OutOfRangeStatus:
    """A class to check and report out-of-range records in an MC object."""

    def __init__(self, mc: 'MC', ranges: dict[str, list]):
        """Initialize with an MC object."""
        self._logger = logging.getLogger(__name__)
        self.mc: 'MC' = mc
        self.ranges: Optional[dict[str, list]] = None
        self.oor: Optional[pd.DataFrame] = None
        self.num_oor: Optional[int] = None
        self.failing_components: Optional[list[str]] = None

        if mc.mass_data is not None:
            self.ranges = self.get_ranges(ranges)
            self.oor: pd.DataFrame = self._check_range()
            self.num_oor: int = len(self.oor)
            self.failing_components: Optional[list[str]] = list(
                self.oor.dropna(axis=1).columns) if self.num_oor > 0 else None

    def get_ranges(self, ranges: dict[str, list]) -> dict[str, list]:

        d_ranges: dict = get_column_config(config_dict=self.mc.config, var_map=self.mc.variable_map,
                                           config_key='range')

        # modify the default dict based on any user passed constraints
        if ranges:
            for k, v in ranges.items():
                d_ranges[k] = v

        return d_ranges

    def _check_range(self) -> pd.DataFrame:
        """Check if all records are within the constraints."""
        if self.mc._mass_data is not None:
            df: pd.DataFrame = self.mc.data[self.ranges.keys()]
            chunks = []
            for variable, bounds in self.ranges.items():
                chunks.append(df.loc[(df[variable] < bounds[0]) | (df[variable] > bounds[1]), variable])
            oor: pd.DataFrame = pd.concat(chunks, axis='columns')
        else:  # An empty object will have ok status
            oor: pd.DataFrame = pd.DataFrame(columns=list(self.ranges.keys()))
        return oor

    @property
    def ok(self) -> bool:
        """Return True if all records are within range, False otherwise."""
        if self.num_oor > 0:
            self._logger.warning(f'{self.num_oor} out of range records exist.')
        return True if self.num_oor == 0 else False

    def __str__(self) -> str:
        """Return a string representation of the status."""
        res: str = f'status.ok: {self.ok}\n'
        res += f'num_oor: {self.num_oor}'
        return res

    def __eq__(self, other: object) -> bool:
        """Return True if other Status has the same out-of-range records."""
        if isinstance(other, OutOfRangeStatus):
            return self.oor.equals(other.oor)
        return False
