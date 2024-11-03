import pandas as pd
import pytest

from elphick.geomet import Sample
from elphick.geomet.base import MassComposition


@pytest.fixture
def sample_data():
    data = {
        'mass_wet': [100., 200., 300.],
        'mass_dry': [90., 180., 270.],
        'component_1': [10., 20., 30.],
        'component_2': [5., 10., 15.]
    }
    return pd.DataFrame(data)


@pytest.fixture
def excess_recovery_sample_data():
    data = {
        'mass_wet': [100., 200., 300.],
        'mass_dry': [90., 180., 290.],
        'component_1': [10., 20., 40.],
        'component_2': [5., 10., 15.]
    }
    return pd.DataFrame(data)


@pytest.fixture
def other_data():
    data = {
        'mass_wet': [110., 210., 310.],
        'mass_dry': [100., 190., 280.],
        'component_1': [12., 22., 32.],
        'component_2': [6., 11., 16.]
    }
    return pd.DataFrame(data)


def test_clip_recovery_no_clip(sample_data, other_data):
    mc = Sample(data=sample_data, name='sample', mass_wet_var='mass_wet', mass_dry_var='mass_dry',
                component_vars=['component_1', 'component_2'])
    other_mc = Sample(data=other_data, name='other', mass_wet_var='mass_wet', mass_dry_var='mass_dry',
                      component_vars=['component_1', 'component_2'])

    recovery_bounds = (0.01, 0.99)
    result = mc.clip_recovery(other_mc, recovery_bounds)

    assert isinstance(result, MassComposition)
    assert not result._mass_data.empty
    assert 'mass_wet' in result._mass_data.columns
    assert 'mass_dry' in result._mass_data.columns
    assert 'component_1' in result._mass_data.columns
    assert 'component_2' in result._mass_data.columns

    # no changes were made, since the recovery was within bounds
    pd.testing.assert_frame_equal(result.data.drop(columns='H2O'), sample_data.drop(columns='h2o'))


def test_clip_recovery_clipped(excess_recovery_sample_data, other_data):
    mc = Sample(data=excess_recovery_sample_data, name='sample', mass_wet_var='mass_wet', mass_dry_var='mass_dry',
                component_vars=['component_1', 'component_2'])
    original_moisture: pd.Series = mc.data['H2O']
    other_mc = Sample(data=other_data, name='other', mass_wet_var='mass_wet', mass_dry_var='mass_dry',
                      component_vars=['component_1', 'component_2'])

    recovery_bounds = (0.01, 0.99)
    result = mc.clip_recovery(other_mc, recovery_bounds)

    expected_result: pd.DataFrame = pd.DataFrame(
        {'mass_wet': {0: 100.0, 1: 200.0, 2: 286.75862}, 'mass_dry': {0: 90.0, 1: 180.0, 2: 277.2},
         'H2O': {0: 10.0, 1: 10.0, 2: 3.333333}, 'component_1': {0: 10.0, 1: 20.0, 2: 32.0},
         'component_2': {0: 5.0, 1: 10.0, 2: 15.692640692640694}})

    pd.testing.assert_frame_equal(result.data, expected_result)

    pd.testing.assert_series_equal(result.data['H2O'], original_moisture)
