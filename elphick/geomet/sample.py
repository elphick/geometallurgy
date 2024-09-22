import copy
from pathlib import Path
from typing import Optional, Literal

import pandas as pd

from elphick.geomet.base import MassComposition


class Sample(MassComposition):
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

