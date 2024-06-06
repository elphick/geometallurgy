import copy
import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union, Literal

import pandas as pd

from elphick.geomet.config import read_yaml
from elphick.geomet.utils.components import get_components, is_compositional
from elphick.geomet.utils.moisture import solve_mass_moisture
from elphick.geomet.utils.pandas import mass_to_composition, composition_to_mass, composition_factors
from elphick.geomet.utils.sampling import random_int
from elphick.geomet.utils.timer import log_timer


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
                 constraints: Optional[dict[str, list]] = None,
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
            constraints: The constraints, or bounds for the columns
            config_file: The configuration file
        """

        self._logger = logging.getLogger(name=self.__class__.__name__)

        if config_file is None:
            config_file = Path(__file__).parent / './config/mc_config.yml'
        self.config = read_yaml(config_file)

        # _nodes can preserve relationships from math operations, and can be used to build a network.
        self._nodes: list[Union[str, int]] = [random_int(), random_int()]

        self.name: str = name
        self.moisture_in_scope: bool = moisture_in_scope
        self.mass_wet_var: Optional[str] = mass_wet_var
        self.mass_dry_var: str = mass_dry_var
        self.moisture_var: Optional[str] = moisture_var
        self.component_vars: Optional[list[str]] = component_vars
        self.composition_units: Literal['%', 'ppm', 'ppb'] = composition_units
        self.composition_factor: int = composition_factors[composition_units]
        self.components_as_symbols: bool = components_as_symbols

        self._mass_data: Optional[pd.DataFrame] = None
        self._supplementary_data = None
        self._aggregate = None

        # set the data
        self.data = data

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

            self._mass_data = composition_to_mass(pd.concat([mass_totals, composition], axis=1),
                                                  mass_wet=self.mass_wet_var, mass_dry=self.mass_dry_var,
                                                  moisture_column_name=self.moisture_column,
                                                  component_columns=composition.columns,
                                                  composition_units=self.composition_units)
            self._logger.debug(f"Data has been set.")

            # Recalculate the aggregate whenever the data changes
            self.aggregate = self._weight_average()
        else:
            self._mass_data = None

    @property
    def aggregate(self):
        if self._aggregate is None:
            self._aggregate = self._weight_average()
        return self._aggregate

    @aggregate.setter
    def aggregate(self, value):
        self._aggregate = value

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
    def moisture_column(self) -> Optional[list[str]]:
        res = 'h2o'
        if self.moisture_in_scope:
            res = self.moisture_var
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

    def _weight_average(self):
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

    def split(self,
              fraction: float,
              name_1: Optional[str] = None,
              name_2: Optional[str] = None,
              include_supplementary_data: bool = False) -> tuple['MassComposition', 'MassComposition']:
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

        out: MassComposition = self.create_congruent_object(name=name_1, include_mc_data=True,
                                                            include_supp_data=include_supplementary_data)
        out._mass_data = self._mass_data * fraction

        comp: MassComposition = self.create_congruent_object(name=name_2, include_mc_data=True,
                                                             include_supp_data=include_supplementary_data)
        comp._mass_data = self._mass_data * (1 - fraction)

        return out, comp

    def add(self, other: 'MassComposition', name: Optional[str] = None,
            include_supplementary_data: bool = False) -> 'MassComposition':
        """Add two objects together

        Args:
            other: The other object
            name: The name of the new object
            include_supplementary_data: Whether to include the supplementary data

        Returns:
            The new object
        """
        new_obj = self.create_congruent_object(name=name, include_mc_data=True,
                                               include_supp_data=include_supplementary_data)
        new_obj._mass_data = self._mass_data + other._mass_data
        return new_obj

    def sub(self, other: 'MassComposition', name: Optional[str] = None,
            include_supplementary_data: bool = False) -> 'MassComposition':
        """Subtract other from self

        Args:
            other: The other object
            name: The name of the new object
            include_supplementary_data: Whether to include the supplementary data

        Returns:
            The new object
        """
        new_obj = self.create_congruent_object(name=name, include_mc_data=True,
                                               include_supp_data=include_supplementary_data)
        new_obj._mass_data = self._mass_data - other._mass_data
        return new_obj

    def div(self, other: 'MassComposition', name: Optional[str] = None,
            include_supplementary_data: bool = False) -> 'MassComposition':
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
        new_obj._mass_data = self._mass_data / other._mass_data
        return new_obj

    @abstractmethod
    def __str__(self):
        # return f"{self.name}\n{self.aggregate.to_dict()}"
        pass

    @abstractmethod
    def create_congruent_object(self, name: str,
                                include_mc_data: bool = False,
                                include_supp_data: bool = False) -> 'MassComposition':
        pass

    def __add__(self, other: 'MassComposition') -> 'MassComposition':
        """Add two objects

        Perform the addition with the mass-composition variables only and then append any attribute variables.
        Presently ignores any attribute vars in other
        Args:
            other: object to add to self

        Returns:

        """

        return self.add(other, include_supplementary_data=True)

    def __sub__(self, other: 'MassComposition') -> 'MassComposition':
        """Subtract the supplied object from self

        Perform the subtraction with the mass-composition variables only and then append any attribute variables.
        Args:
            other: object to subtract from self

        Returns:

        """

        return self.sub(other, include_supplementary_data=True)

    def __truediv__(self, other: 'MassComposition') -> 'MassComposition':
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
