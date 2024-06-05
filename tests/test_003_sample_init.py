import copy

import pandas as pd
import pytest

from geomet import Sample
from geomet.utils.components import is_compositional
from geomet.utils.data import sample_data


@pytest.fixture
def expected_data() -> pd.DataFrame:
    expected_data = sample_data(include_wet_mass=True,
                                include_dry_mass=True,
                                include_moisture=True)
    return expected_data


@pytest.fixture
def expected_data_symbols() -> pd.DataFrame:
    expected_data = sample_data(include_wet_mass=True,
                                include_dry_mass=True,
                                include_moisture=True)
    expected_data.rename(columns=is_compositional(expected_data.columns, strict=False), inplace=True)
    return expected_data


def test_sample_init(expected_data):
    data = sample_data(include_moisture=True)
    smpl = Sample(data=data, name='sample', components_as_symbols=False)
    pd.testing.assert_frame_equal(smpl.data, expected_data)


def test_sample_init_symbols(expected_data_symbols):
    data = sample_data(include_moisture=True)
    smpl = Sample(data=data, name='sample', components_as_symbols=True)
    pd.testing.assert_frame_equal(smpl.data, expected_data_symbols)


def test_sample_init_no_moisture(expected_data_symbols):
    data = sample_data()
    smpl = Sample(data=data, name='sample')
    pd.testing.assert_frame_equal(smpl.data, expected_data_symbols)


def test_sample_init_no_wet_mass(expected_data_symbols):
    data = sample_data(include_moisture=True, include_wet_mass=False)
    smpl = Sample(data=data, name='sample')
    pd.testing.assert_frame_equal(smpl.data, expected_data_symbols.rename(columns={'wet_mass': 'mass_wet'}))


def test_sample_init_no_dry_mass(expected_data_symbols):
    data = sample_data(include_moisture=True, include_dry_mass=False)
    smpl = Sample(data=data, name='sample')
    pd.testing.assert_frame_equal(smpl.data, expected_data_symbols)


def test_sample_init_no_chem_vars(expected_data):
    data = sample_data(include_moisture=False, include_chem_vars=False)
    smpl = Sample(data=data, name='sample')

    expected_data = expected_data.drop(columns=['FE', 'SIO2', 'al2o3', 'LOI'])
    pd.testing.assert_frame_equal(smpl.data, expected_data)


def test_sample_init_moisture_naive(expected_data_symbols):
    name = 'sample'
    data = sample_data(include_moisture=False, include_wet_mass=False)
    smpl = Sample(data=data, name=name, moisture_in_scope=False)

    expected_data = expected_data_symbols.drop(columns=['wet_mass', 'H2O'])
    pd.testing.assert_frame_equal(smpl.data, expected_data)

    msg = (
        f"mass_wet_var is not provided and cannot be calculated from mass_dry_var and moisture_var.  "
        f"Consider specifying the mass_wet_var, mass_dry_var and moisture_var, or alternatively set "
        f"moisture_in_scope to False for sample")
    with pytest.raises(ValueError, match=msg):
        smpl = Sample(data=data, name=name, moisture_in_scope=True)


def test_deepcopy():
    # Create an instance of MassComposition
    smpl1 = Sample(data=sample_data())

    # Make a deep copy of mc1
    smpl2 = copy.deepcopy(smpl1)

    # Check that mc1 and mc2 are not the same object
    assert smpl1 is not smpl2

    # Check that mc1 and mc2 have the same data
    pd.testing.assert_frame_equal(smpl1.data, smpl2.data)
