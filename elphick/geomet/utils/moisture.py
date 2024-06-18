import logging
import re
from copy import deepcopy
from typing import Optional, Dict, List

import numpy as np
import pandas as pd


def detect_moisture_column(columns: List[str]) -> Optional[str]:
    """Detects the moisture column in a list of columns

    Args:
        columns: List of column names

    Returns:

    """
    res: Optional[str] = None
    search_regex: str = '(h2o)|(moisture)|(moist)|(mc)|(moisture_content)'
    for col in columns:
        if re.search(search_regex, col, re.IGNORECASE):
            res = col
            break
    return res


def solve_mass_moisture(mass_wet: pd.Series = None,
                        mass_dry: pd.Series = None,
                        moisture: pd.Series = None,
                        moisture_column_name: str = 'h2o',
                        rtol: float = 1e-05,
                        atol: float = 1e-08) -> pd.Series:
    logger = logging.getLogger(name=__name__)
    _vars: Dict = {k: v for k, v in deepcopy(locals()).items()}
    key_columns = ['mass_wet', 'mass_dry', 'moisture']
    vars_supplied: List[str] = [k for k in key_columns if _vars.get(k) is not None]

    if len(vars_supplied) == 3:
        logger.info('Over-specified - checking for balance.')
        re_calc_moisture = (mass_wet - mass_dry) / mass_wet * 100
        if not np.isclose(re_calc_moisture, moisture, rtol=rtol, atol=atol).all():
            msg = f"Mass balance is not satisfied: {re_calc_moisture}"
            logger.error(msg)
            raise ValueError(msg)
    elif len(vars_supplied) == 1:
        raise ValueError('Insufficient arguments supplied - at least 2 required.')

    var_to_solve: str = next((k for k, v in _vars.items() if v is None), None)

    res: Optional[pd.Series] = None
    if var_to_solve:
        calculations = {
            'mass_wet': lambda: mass_dry / (1 - moisture / 100),
            'mass_dry': lambda: mass_wet - (mass_wet * moisture / 100),
            'moisture': lambda: (mass_wet - mass_dry) / mass_wet * 100
        }

        res = calculations[var_to_solve]()
        res.name = var_to_solve if var_to_solve != 'moisture' else moisture_column_name  # use the supplied column name

    return res