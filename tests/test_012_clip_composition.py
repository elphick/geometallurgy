import pytest
import pandas as pd
import logging

from elphick.geomet import Sample

from fixtures import sample_data


@pytest.fixture
def test_sample(sample_data):
    return Sample(sample_data, name='sample')


@pytest.fixture
def oor_sample(sample_data):
    # modify the sample data to become out-of-range
    oor_data: pd.DataFrame = sample_data.copy()
    oor_data.loc[0, 'FE'] = 70.0
    oor_data.loc[1, 'SIO2'] = -10.0
    return Sample(oor_data, name='sample')


def test_dummy(sample_data):
    sample = Sample(sample_data)
    assert sample.data is not None


def test_clip_composition_default_ranges(oor_sample):
    oor_sample.clip_composition()
    expected_data = pd.DataFrame({
        'wet_mass': [100.0, 90.0, 110.0],
        'mass_dry': [90.0, 80.0, 90.0],
        'H2O': [10.0, 11.111111, 18.181818],
        'Fe': [69.97, 59.0, 61.0],  # Assuming H2O is clipped to 69.97
        'SiO2': [5.2, 0.0, 2.2],  # Assuming SiO2 is clipped to 0.0
        'Al2O3': [3.0, 1.7, 0.9],
        'LOI': [5.0, 4.0, 3.0],
        'group': ['grp_1', 'grp_1', 'grp_2']
    })
    expected_data.index.name = 'index'
    pd.testing.assert_frame_equal(oor_sample.data, expected_data)


def test_clip_composition_custom_ranges(oor_sample):
    custom_ranges = {
        'Fe': [60.0, 65.0],
        'SiO2': [0.0, 5.0],
    }
    oor_sample.clip_composition(custom_ranges)
    expected_data = pd.DataFrame({
        'wet_mass': [100.0, 90.0, 110.0],
        'mass_dry': [90.0, 80.0, 90.0],
        'H2O': [10.0, 11.111111, 18.181818],
        'Fe': [65.0, 60.0, 61.0],  # Fe clipped between 60 to 65.0
        'SiO2': [5.0, 0.0, 2.2],  # SiO2 clipped between 0 to 5.0
        'Al2O3': [3.0, 1.7, 0.9],
        'LOI': [5.0, 4.0, 3.0],
        'group': ['grp_1', 'grp_1', 'grp_2']})
    expected_data.index.name = 'index'

    pd.testing.assert_frame_equal(oor_sample.data, expected_data)


def test_clip_composition_log_message(oor_sample, caplog):
    with caplog.at_level(logging.INFO):
        oor_sample.clip_composition(None)
    assert "2 records where composition has been clipped to the range" in caplog.text
    assert "Affected indexes (first 50):" in caplog.text


def test_clip_composition_no_clipping_needed(test_sample):
    test_sample.clip_composition(None)
    expected_data = pd.DataFrame({
        'wet_mass': [100.0, 90.0, 110.0],
        'mass_dry': [90.0, 80.0, 90.0],
        'H2O': [10.0, 11.111111, 18.181818],
        'Fe': [57.0, 59.0, 61.0],
        'SiO2': [5.2, 3.1, 2.2],
        'Al2O3': [3.0, 1.7, 0.9],
        'LOI': [5.0, 4.0, 3.0],
        'group': ['grp_1', 'grp_1', 'grp_2']})
    expected_data.index.name = 'index'

    pd.testing.assert_frame_equal(test_sample.data, expected_data)


