import logging

import pandas as pd
import pytest

from fixtures import sample_data
from elphick.geomet.utils.moisture import solve_mass_moisture, detect_moisture_column


def test_moisture_solver(sample_data):
    import numpy as np

    data = sample_data
    wet: pd.Series = data['wet_mass']
    dry: pd.Series = data['mass_dry']

    res_1: pd.Series = solve_mass_moisture(mass_wet=wet, mass_dry=dry, moisture=None)

    h2o: pd.Series = res_1.copy()

    dry_calc: pd.Series = solve_mass_moisture(mass_wet=wet, mass_dry=None, moisture=h2o)
    wet_calc: pd.Series = solve_mass_moisture(mass_wet=None, mass_dry=dry, moisture=h2o)

    assert all(np.isclose(wet, wet_calc))
    assert all(np.isclose(dry, dry_calc))

    with pytest.raises(ValueError, match='Insufficient arguments supplied - at least 2 required.'):
        res_4: pd.Series = solve_mass_moisture(mass_wet=None, mass_dry=None, moisture=h2o)

    res_5: pd.Series = solve_mass_moisture(mass_wet=wet, mass_dry=dry, moisture=h2o)


def test_detect_moisture_column(sample_data):
    data = sample_data
    columns = data.columns
    res = detect_moisture_column(columns)
    assert res is None

    columns = ['mass_wet', 'mass_dry', 'H2O', 'FE', 'SIO2', 'AL2O3', 'LOI']
    res = detect_moisture_column(columns)
    assert res == 'H2O'

    columns = ['mass_wet', 'mass_dry', 'h2o', 'FE', 'SIO2', 'AL2O3', 'LOI']
    res = detect_moisture_column(columns)
    assert res == 'h2o'

    columns = ['mass_wet', 'mass_dry', 'MC', 'FE', 'SIO2', 'AL2O3', 'LOI']
    res = detect_moisture_column(columns)
    assert res == 'MC'
