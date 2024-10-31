import pandas as pd
import pytest
from elphick.geomet.base import MassComposition


@pytest.fixture
def sample_mass_composition():
    data = pd.DataFrame({
        'mass_wet': [100, 200, 300],
        'mass_dry': [90, 180, 270],
        'component_1': [10, 20, 30],
        'component_2': [5, 10, 15]
    })
    return MassComposition(data=data, mass_wet_var='mass_wet', mass_dry_var='mass_dry')


def test_set_moisture_with_float(sample_mass_composition):
    mc = sample_mass_composition
    mc.set_moisture(10, mass_to_adjust='wet')  # 10% moisture
    expected_mass_dry = mc.mass_data['mass_wet'] * (1 - 0.1)
    assert all(mc.mass_data['mass_dry'] == expected_mass_dry)


def test_set_moisture_with_series(sample_mass_composition):
    mc = sample_mass_composition
    moisture_series = pd.Series([10, 20, 30], index=mc.mass_data.index)  # 10%, 20%, 30% moisture
    mc.set_moisture(moisture_series, mass_to_adjust='wet')
    expected_mass_dry = mc.mass_data['mass_wet'] * (1 - moisture_series / 100)
    assert all(mc.mass_data['mass_dry'] == expected_mass_dry)


def test_set_moisture_adjust_dry(sample_mass_composition):
    mc = sample_mass_composition
    mc.set_moisture(10, mass_to_adjust='dry')  # 10% moisture
    expected_mass_wet = mc.mass_data['mass_dry'] / (1 - 0.1)
    assert all(mc.mass_data['mass_wet'] == expected_mass_wet)


def test_set_moisture_invalid_type(sample_mass_composition):
    mc = sample_mass_composition
    with pytest.raises(TypeError):
        mc.set_moisture([10, 20, 30], mass_to_adjust='wet')


def test_set_moisture_invalid_mass_to_adjust(sample_mass_composition):
    mc = sample_mass_composition
    with pytest.raises(ValueError):
        mc.set_moisture(10, mass_to_adjust='invalid')
