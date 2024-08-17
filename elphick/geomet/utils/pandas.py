"""
Pandas utils
"""
import inspect
import logging
import tokenize
from io import StringIO
from token import STRING
from typing import List, Dict, Optional, Literal

import pandas as pd
from scipy.stats import gmean

from elphick.geomet.utils.components import is_compositional, get_components
from elphick.geomet.utils.moisture import solve_mass_moisture, detect_moisture_column
from elphick.geomet.utils.size import mean_size

composition_factors: dict[str, int] = {'%': 100, 'ppm': 1e6, 'ppb': 1e9}


def column_prefixes(columns: List[str]) -> Dict[str, List[str]]:
    return {prefix: [col for col in columns if prefix == col.split('_')[0]] for prefix in
            list(dict.fromkeys([col.split('_')[0] for col in columns if len(col.split('_')) > 1]))}


def column_prefix_counts(columns: List[str]) -> Dict[str, int]:
    return {k: len(v) for k, v in column_prefixes(columns).items()}


def mass_to_composition(df: pd.DataFrame,
                        mass_wet: Optional[str] = 'mass_wet',
                        mass_dry: str = 'mass_dry',
                        moisture_column_name: Optional[str] = None,
                        component_columns: Optional[list[str]] = None,
                        composition_units: Literal['%', 'ppm', 'ppb'] = '%') -> pd.DataFrame:
    """Convert a mass DataFrame to composition

    Supplementary columns (columns that are not mass or composition) are ignored.

    Args:
        df: The pd.DataFrame containing mass.  H2O if provided will be ignored.  All columns other than the
         mass_wet and mass_dry are assumed to be `additive`, that is, dry mass weighting is valid.
         Assumes composition is in %w/w units.
        mass_wet: The wet mass column, optional. If not provided, it's assumed to be equal to mass_dry.
        mass_dry: The dry mass column, not optional.  Consider solve_mass_moisture prior to this call if needed.
        moisture_column_name: if mass_wet is provided, the resultant moisture will be returned with this column name.
         If None, and moisture is detected in the input, that column name will be used instead.

        component_columns: The composition columns to be used for the calculation.  If not provided, the columns
         will be auto-detected using a case in-sensitive match to all elements and oxides.  H2O is excluded
        composition_units: determines the factor to convert mass to composition.

    Returns:
        A pd.Dataframe containing mass (wet and dry mass) and composition
    """

    moisture_column_name, mass_moisture_cols, component_cols = prepare_columns(df, mass_wet, mass_dry,
                                                                               moisture_column_name, component_columns)

    if mass_wet and mass_wet in df.columns:
        mass: pd.DataFrame = df[[mass_wet, mass_dry]]
    else:
        mass: pd.DataFrame = df[[mass_dry]]

    component_mass: pd.DataFrame = df[component_cols]
    composition: pd.DataFrame = component_mass.div(mass[mass_dry], axis=0) * composition_factors[composition_units]

    if mass_wet and (mass_wet in df.columns):
        moisture: pd.Series = solve_mass_moisture(mass_wet=mass[mass_wet], mass_dry=mass[mass_dry]).rename(
            moisture_column_name)
        return pd.concat([mass, moisture, composition], axis='columns')
    else:
        return pd.concat([mass, composition], axis=1)


def composition_to_mass(df: pd.DataFrame,
                        mass_wet: Optional[str] = None,
                        mass_dry: str = 'mass_dry',
                        component_columns: Optional[list[str]] = None,
                        moisture_column_name: Optional[str] = None,
                        composition_units: Literal['%', 'ppm', 'ppb'] = '%',
                        return_moisture: bool = False) -> pd.DataFrame:
    """ Convert a composition DataFrame to mass

        Supplementary columns (columns that are not mass or composition) are ignored.

    Args:
        df: The pd.DataFrame containing mass.  H2O if provided will be ignored.  All columns other than the
         mass_wet and mass_dry are assumed to be `additive`, that is, dry mass weighting is valid.
         Assumes composition is in %w/w units.
        mass_wet: The wet mass column, optional. If not provided, it's assumed to be equal to mass_dry.
        mass_dry: The dry mass column, not optional.  Consider solve_mass_moisture prior to this call if needed.
        moisture_column_name: if mass_wet is provided, the resultant moisture will be returned with this column name.
         If None, and moisture is detected in the input, that column name will be used instead.
        component_columns: The composition columns to be used for the calculation.  If not provided, the columns
         will be auto-detected using a case in-sensitive match to all elements and oxides.  H2O is excluded
        composition_units: determines the factor to convert composition to mass.
        return_moisture: If True, the moisture column will be returned.

    Returns:
        A pd.Dataframe containing the mass representation of mass totals and components
    """

    moisture_column_name, mass_moisture_cols, component_cols = prepare_columns(df, mass_wet, mass_dry,
                                                                               moisture_column_name, component_columns)

    if mass_wet and mass_wet in df.columns:
        mass: pd.DataFrame = df[[mass_wet, mass_dry]]
    else:
        mass: pd.DataFrame = df[[mass_dry]]

    composition: pd.DataFrame = df[component_cols]
    component_mass: pd.DataFrame = composition.mul(mass[mass_dry], axis=0) / composition_factors[composition_units]

    if mass_wet and (mass_wet in df.columns) and return_moisture:
        moisture: pd.Series = (mass[mass_wet] - mass[mass_dry]).rename(moisture_column_name)
        return pd.concat([mass, moisture, component_mass], axis='columns')
    else:
        return pd.concat([mass, component_mass], axis=1)


def prepare_columns(df: pd.DataFrame, mass_wet: Optional[str], mass_dry: str, moisture_column_name: Optional[str],
                    component_columns: Optional[list[str]]) -> tuple[str, List[str], List[str]]:
    if moisture_column_name is None:
        moisture_column_name = detect_moisture_column(df.columns)
        # if moisture_column_name is None:
        #     moisture_column_name = 'h2o'  # set default value to 'h2o' if not detected
    mass_moisture_cols = [mass_wet, mass_dry, moisture_column_name]

    if component_columns is None:
        non_mass_cols: list[str] = [col for col in df.columns if col.lower() not in mass_moisture_cols]
        component_cols: list[str] = get_components(df[non_mass_cols], strict=False)
    else:
        component_cols: list[str] = component_columns

    return moisture_column_name, mass_moisture_cols, component_cols


def weight_average(df: pd.DataFrame,
                   mass_wet: Optional[str] = None,
                   mass_dry: str = 'mass_dry',
                   moisture_column_name: Optional[str] = None,
                   component_columns: Optional[list[str]] = None,
                   composition_units: Literal['%', 'ppm', 'ppb'] = '%') -> pd.Series:
    """Weight Average a DataFrame containing mass-composition

    Args:
        df: The pd.DataFrame containing mass-composition.  H2O if provided will be ignored.  All columns other than the
         mass_wet and mass_dry are assumed to be `additive`, that is, dry mass weighting is valid.
         Assumes composition is in %w/w units.
        mass_wet: The optional wet mass column.
        mass_dry: The dry mass column, not optional.  Consider solve_mass_moisture prior to this call if needed.
        moisture_column_name: if mass_wet is provided, the resultant moisture will be returned with this column name.
         If None, and moisture is detected in the input, that column name will be used instead.
        component_columns: The composition columns to be used for the calculation.  If not provided, the columns
         will be auto-detected using a case in-sensitive match to all elements and oxides.  H2O is excluded
        composition_units: determines the factor to convert mass to composition.

    Returns:
        A pd.Series containing the total mass and weight averaged composition.
    """
    moisture_column_name, mass_moisture_cols, component_cols = prepare_columns(df, mass_wet, mass_dry,
                                                                               moisture_column_name, component_columns)

    mass_sum: pd.DataFrame = df.pipe(composition_to_mass, mass_wet=mass_wet, mass_dry=mass_dry,
                                     moisture_column_name=moisture_column_name,
                                     component_columns=component_columns,
                                     composition_units=composition_units).sum(axis="index").to_frame().T

    component_cols = [col for col in component_cols if
                      col.lower() not in [mass_wet, mass_dry, 'h2o', 'moisture']]

    weighted_composition: pd.Series = mass_sum[component_cols].div(mass_sum[mass_dry], axis=0) * composition_factors[
        composition_units]

    if mass_wet and (mass_wet in df.columns):
        moisture: pd.Series = solve_mass_moisture(mass_wet=mass_sum[mass_wet], mass_dry=mass_sum[mass_dry])
        return pd.concat([mass_sum[[mass_wet, mass_dry]], moisture, weighted_composition], axis=1).iloc[0].rename(
            'weight_average')
    else:
        return pd.concat([mass_sum[[mass_dry]], weighted_composition], axis=1).iloc[0].rename('weight_average')


def calculate_recovery(df: pd.DataFrame,
                       df_ref: pd.DataFrame,
                       mass_wet: str = 'mass_wet',
                       mass_dry: str = 'mass_dry') -> pd.DataFrame:
    """Calculate recovery of mass-composition for two DataFrames

    Args:
        df: The pd.DataFrame containing mass-composition.  H2O if provided will be ignored.  All columns other than the
         mass_wet and mass_dry are assumed to be `additive`, that is, dry mass weighting is valid.
         Assumes composition is in %w/w units.
        df_ref: The stream that df will be divided by to calculate the recovery.  Often the feed stream.
        mass_wet: The wet mass column, not optional.  Consider solve_mass_moisture prior to this call if needed.
        mass_dry: The dry mass column, not optional.  Consider solve_mass_moisture prior to this call if needed.

    Returns:
        A pd.Series containing the total mass and weight averaged composition.
    """

    res: pd.DataFrame = df.pipe(composition_to_mass, mass_wet=mass_wet, mass_dry=mass_dry) / df_ref.pipe(
        composition_to_mass, mass_wet=mass_wet, mass_dry=mass_dry)
    return res


def calculate_partition(df_feed: pd.DataFrame,
                        df_preferred: pd.DataFrame,
                        col_mass_dry: str = 'mass_dry') -> pd.DataFrame:
    """Calculate the partition curve from two streams

    .. math::
        K = \\frac{{m_{preferred}}}{{m_{feed}}}

    Applicable to the one dimensional case only.  The PN is bounded [0, 1].
    The interval mean for size is the geometric mean, otherwise the arithmetic mean.
    The interval mean is named `da`, which can be interpreted as `diameter-average` or `density-average`.
    TODO: consider a generalised name, fraction-average -> fa?

    Args:
        df_feed: The pd.DataFrame containing mass-composition representing the fractionated feed.
        df_preferred: The pd.DataFrame containing mass-composition representing the fractionated preferred stream.
        col_mass_dry: The dry mass column, not optional.

    Returns:
        A pd.DataFrame containing the partition data with a range [0, 1].
    """

    res: pd.DataFrame = df_preferred[[col_mass_dry]].div(df_feed[[col_mass_dry]]).rename(columns={col_mass_dry: 'K'})
    if df_preferred.index.name.lower() == 'size':
        res.insert(loc=0, column='da', value=mean_size(res.index))
    else:
        res.insert(loc=0, column='da', value=res.index.mid)
    return res


def cumulate(mass_data: pd.DataFrame, direction: str) -> pd.DataFrame:
    """Cumulate along the index

    Expected use case is only for Datasets that have been reduced to 1D.

    Args:
        mass_data: The mass data to cumulate - note composition must be represented as mass
        direction: 'ascending'|'descending'

    Returns:

    """

    valid_dirs: List[str] = ['ascending', 'descending']
    if direction not in valid_dirs:
        raise KeyError(f'Invalid direction provided.  Valid arguments are: {valid_dirs}')

    d_dir: Dict = {'ascending': True if direction == 'ascending' else False,
                   'descending': True if direction == 'descending' else False}

    if mass_data.index.ndim > 1:
        raise NotImplementedError('DataFrames having indexes > 1D have not been tested.')

    index_var: str = mass_data.index.name
    if not isinstance(mass_data.index, pd.IntervalIndex):
        raise NotImplementedError(f"The {index_var} of this object is not a pd.Interval. "
                                  f" Only 1D interval objects are valid")

    interval_index = mass_data.index.get_level_values(index_var)
    if not (interval_index.is_monotonic_increasing or interval_index.is_monotonic_decreasing):
        raise ValueError('Index is not monotonically increasing or decreasing')

    in_data_ascending: bool = True
    if interval_index.is_monotonic_decreasing:
        in_data_ascending = False

    # sort by the direction provided, first save the index
    original_index: pd.Index = mass_data.index
    try:
        mass_data: pd.DataFrame = mass_data.sort_index(ascending=d_dir[direction])
        mass_cum: pd.DataFrame = mass_data.cumsum()

    finally:
        # reset the index to the original
        mass_data = mass_data.reindex(original_index)

    return mass_cum


def _detect_non_float_columns(df):
    _logger: logging.Logger = logging.getLogger(inspect.stack()[1].function)
    non_float_cols: List = [col for col in df.columns if col not in df.select_dtypes(include=[float, int]).columns]
    if len(non_float_cols) > 0:
        _logger.info(f"The following columns are not float columns and will be ignored: {non_float_cols}")
    return non_float_cols


def _detect_non_component_columns(df):
    _logger: logging.Logger = logging.getLogger(inspect.stack()[1].function)
    chemistry_vars = [col.lower() for col in is_compositional(df.columns, strict=False).values() if col not in ['H2O']]

    non_float_cols: List = [col for col in df.columns if
                            col not in (list(df.select_dtypes(include=[float, int]).columns) + chemistry_vars + [
                                'mass_wet', 'mass_dry', 'h2o'])]
    if len(non_float_cols) > 0:
        _logger.info(f"The following columns are not float columns and will be ignored: {non_float_cols}")
    return non_float_cols


class MeanIntervalIndex(pd.IntervalIndex):
    """MeanIntervalIndex is a subclass of pd.IntervalIndex that calculates the mean of the interval bounds."""

    def __new__(cls, data, mean_values=None):
        obj = pd.IntervalIndex.__new__(cls, data)
        return obj

    def __init__(self, data, mean_values=None):
        self.mean_values = mean_values

    @property
    def mean(self):
        if self.mean_values is not None:
            return self.mean_values
        elif self.name == 'size':
            # Calculate geometric mean
            return mean_size(self)
        else:
            # Calculate arithmetic mean
            return (self.right + self.left) / 2


import pandas as pd
import numpy as np
from scipy.stats import gmean


class MeanIntervalArray(pd.arrays.IntervalArray):
    def __init__(self, data, dtype=None, copy=False):
        super().__init__(data, dtype, copy)
        if self.name == 'size':
            # Calculate geometric mean
            self.mean_values = gmean([self.right, self.left], axis=0)
        else:
            # Calculate arithmetic mean
            self.mean_values = (self.right + self.left) / 2

    @property
    def mean(self):
        if self.mean_values is not None:
            return self.mean_values
        elif self.name == 'size':
            # Calculate geometric mean
            return gmean([self.right, self.left], axis=0)
        else:
            # Calculate arithmetic mean
            return (self.right + self.left) / 2


def parse_vars_from_expr(expr: str) -> list[str]:
    """ Parse variables from a pandas query expression string.

    Args:
        expr: The expression string

    Returns:
        list[str]: The list of variables
    """
    variables = set()
    tokens = tokenize.generate_tokens(StringIO(expr).readline)
    logical_operators = {'and', 'or', '&', '|'}
    inside_backticks = False
    current_var = []

    for token in tokens:
        if token.string == '`':
            if inside_backticks:
                # End of backtick-enclosed variable
                variables.add(' '.join(current_var))
                current_var = []
            inside_backticks = not inside_backticks
        elif inside_backticks:
            if token.type in {tokenize.NAME, STRING}:
                current_var.append(token.string)
        elif token.type == tokenize.NAME and token.string not in logical_operators:
            variables.add(token.string)

    return list(variables)
