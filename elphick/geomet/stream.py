import copy
from pathlib import Path
from typing import Optional, Literal

import pandas as pd

from elphick.geomet import MassComposition


class Stream(MassComposition):
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
        super().__init__(data=data, name=name, moisture_in_scope=moisture_in_scope,
                         mass_wet_var=mass_wet_var, mass_dry_var=mass_dry_var,
                         moisture_var=moisture_var, component_vars=component_vars,
                         composition_units=composition_units, components_as_symbols=components_as_symbols,
                         constraints=constraints, config_file=config_file)

    def create_congruent_object(self, name: str,
                                include_mc_data: bool = False,
                                include_supp_data: bool = False) -> 'Sample':
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

    def __str__(self):
        return f"Stream: {self.name}\n{self.aggregate.to_dict()}"
