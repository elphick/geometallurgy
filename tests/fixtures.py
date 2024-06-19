import pandas as pd
import pytest


@pytest.fixture
def sample_data(include_wet_mass: bool = True, include_dry_mass: bool = True,
                include_moisture: bool = False) -> pd.DataFrame:
    """Creates synthetic data for testing

    Args:
        include_wet_mass: If True, wet mass is included.
        include_dry_mass: If True, dry mass is included.
        include_moisture: If True, moisture (H2O) is included.

    Returns:

    """

    # mass_wet: pd.Series = pd.Series([100, 90, 110], name='wet_mass')
    # mass_dry: pd.Series = pd.Series([90, 80, 100], name='dry_mass')
    mass_wet: pd.Series = pd.Series([100., 90., 110.], name='wet_mass')
    mass_dry: pd.Series = pd.Series([90., 80., 90.], name='mass_dry')
    chem: pd.DataFrame = pd.DataFrame.from_dict({'FE': [57., 59., 61.],
                                                 'SIO2': [5.2, 3.1, 2.2],
                                                 'al2o3': [3.0, 1.7, 0.9],
                                                 'LOI': [5.0, 4.0, 3.0]})
    attrs: pd.Series = pd.Series(['grp_1', 'grp_1', 'grp_2'], name='group')

    mass: pd.DataFrame = pd.concat([mass_wet, mass_dry], axis='columns')
    if include_wet_mass is True and mass_dry is False:
        mass = mass_wet
    elif include_dry_mass is False and mass_dry is True:
        mass = mass_dry
    elif include_dry_mass is False and mass_dry is False:
        raise AssertionError('Arguments provided result in no mass column')

    if include_moisture is True:
        moisture: pd.DataFrame = (mass_wet - mass_dry) / mass_wet * 100
        moisture.name = 'H2O'
        res: pd.DataFrame = pd.concat([mass, moisture, chem, attrs], axis='columns')
    else:
        res: pd.DataFrame = pd.concat([mass, chem, attrs], axis='columns')

    res.index.name = 'index'

    return res
