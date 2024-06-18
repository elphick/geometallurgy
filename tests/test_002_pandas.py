import pytest
import pandas as pd
import numpy as np
from pandas import IntervalIndex
from scipy.stats.mstats import gmean

from elphick.geomet.utils.pandas import mass_to_composition, composition_to_mass, weight_average, MeanIntervalIndex, \
    MeanIntervalArray
from fixtures import sample_data as test_data


def test_composition_to_mass(test_data):
    result = composition_to_mass(test_data)

    expected_output = pd.DataFrame({'mass_dry': {0: 90.0, 1: 80.0, 2: 90.0}, 'FE': {0: 51.3, 1: 47.2, 2: 54.9},
                                    'SIO2': {0: 4.68, 1: 2.48, 2: 1.98},
                                    'al2o3': {0: 2.7, 1: 1.36, 2: 0.81}, 'LOI': {0: 4.5, 1: 3.2, 2: 2.7}},
                                   index=result.index)

    pd.testing.assert_frame_equal(result, expected_output)


def test_composition_to_mass_with_moisture(test_data):
    result = composition_to_mass(test_data, mass_wet='wet_mass', moisture_column_name='H2O', return_moisture=True)

    expected_output = pd.DataFrame({'wet_mass': {0: 100.0, 1: 90.0, 2: 110.0}, 'mass_dry': {0: 90.0, 1: 80.0, 2: 90.0},
                                    'H2O': {0: 10.0, 1: 10.0, 2: 20.0}, 'FE': {0: 51.3, 1: 47.2, 2: 54.9},
                                    'SIO2': {0: 4.68, 1: 2.48, 2: 1.98}, 'al2o3': {0: 2.7, 1: 1.36, 2: 0.81},
                                    'LOI': {0: 4.5, 1: 3.2, 2: 2.7}}, index=result.index)

    pd.testing.assert_frame_equal(result, expected_output)


def test_composition_to_mass_with_wet(test_data):
    result = composition_to_mass(test_data, mass_wet='wet_mass', return_moisture=False)

    expected_output = pd.DataFrame({'wet_mass': {0: 100.0, 1: 90.0, 2: 110.0}, 'mass_dry': {0: 90.0, 1: 80.0, 2: 90.0},
                                    'FE': {0: 51.3, 1: 47.2, 2: 54.9},
                                    'SIO2': {0: 4.68, 1: 2.48, 2: 1.98},
                                    'al2o3': {0: 2.7, 1: 1.36, 2: 0.81}, 'LOI': {0: 4.5, 1: 3.2, 2: 2.7}},
                                   index=result.index)
    pd.testing.assert_frame_equal(result, expected_output)


def test_composition_to_mass_with_wet_specific_comp_cols(test_data):
    result = composition_to_mass(test_data, mass_wet='wet_mass', component_columns=['FE', 'SIO2'])

    expected_output = pd.DataFrame({'wet_mass': {0: 100.0, 1: 90.0, 2: 110.0}, 'mass_dry': {0: 90.0, 1: 80.0, 2: 90.0},
                                    'FE': {0: 51.3, 1: 47.2, 2: 54.9},
                                    'SIO2': {0: 4.68, 1: 2.48, 2: 1.98}},
                                   index=result.index)
    pd.testing.assert_frame_equal(result, expected_output)


def test_mass_to_composition(test_data):
    df_mass: pd.DataFrame = composition_to_mass(test_data)
    df_comp: pd.DataFrame = mass_to_composition(df_mass)

    expected_output = test_data[[col for col in test_data.columns if col not in ['wet_mass', 'group']]]

    pd.testing.assert_frame_equal(df_comp, expected_output)


def test_mass_to_composition_with_wet(test_data):
    df_mass = composition_to_mass(test_data, mass_wet='wet_mass', moisture_column_name='h2o', return_moisture=True)
    df_comp: pd.DataFrame = mass_to_composition(df_mass, mass_wet='wet_mass')

    expected_output: pd.DataFrame = test_data[
        [col for col in test_data.columns if col not in ['group']]]
    expected_output.insert(loc=2, column='h2o', value=np.array([10.0, 11.1111111, 18.181818]))

    pd.testing.assert_frame_equal(df_comp, expected_output)


def test_weight_average(test_data):
    res = weight_average(test_data)

    expected_output: pd.Series = pd.Series(
        {'mass_dry': 260.0, 'FE': 59.0, 'SIO2': 3.5153846153846153, 'al2o3': 1.8730769230769235,
         'LOI': 4.0}, name='weight_average')

    pd.testing.assert_series_equal(res, expected_output)


def test_weight_average_with_wet(test_data):
    res = weight_average(test_data, mass_wet='wet_mass', moisture_column_name='h2o')

    expected_output: pd.Series = pd.Series(
        {'wet_mass': 300.0, 'mass_dry': 260.0, 'h2o': 13.333333333333334, 'FE': 59.0,
         'SIO2': 3.5153846153846153, 'al2o3': 1.8730769230769235, 'LOI': 4.0}, name='weight_average')

    pd.testing.assert_series_equal(res, expected_output)


def test_mean_interval_array():
    # Create a IntervalArray instance
    intervals = pd.arrays.IntervalArray.from_tuples([(1, 2), (2, 3), (3, 4)], closed='left')
    # create our custom object

    mean_values = [1.5, 2.5, 3.5]  # replace with your actual mean values
    intervals = MeanIntervalArray.from_tuples([(1, 2), (2, 3), (3, 4)], mean_values=mean_values)

    intervals = MeanIntervalArray.from_tuples([(1, 2), (2, 3), (3, 4)])

    # Check if the mean property returns the geometric mean
    expected_mean = np.mean([intervals.right, intervals.left], axis=0)
    assert np.allclose(intervals.mean, expected_mean)


def test_mean_interval_index():
    # Create a CustomIntervalIndex instance
    intervals = pd.arrays.IntervalArray.from_tuples([(1, 2), (2, 3), (3, 4)], closed='left')
    # check the intervals can instantiate a standard IntervalIndex
    index = IntervalIndex(intervals, name='size')
    # create our custom object
    index = MeanIntervalIndex(intervals)
    index.name = 'size'

    # Check if the mean property returns the geometric mean
    expected_mean = gmean([index.right, index.left], axis=0)
    assert np.allclose(index.mean, expected_mean)

    # Change the name and check if the mean property returns the arithmetic mean
    index.name = 'other'
    expected_mean = (index.right + index.left) / 2
    assert np.allclose(index.mean, expected_mean)


def test_mean_interval_index_with_input():
    # Create a CustomIntervalIndex instance
    intervals = pd.arrays.IntervalArray.from_tuples([(1, 2), (2, 3), (3, 4)])
    mean_values = [1.5, 2.5, 3.5]  # replace with your actual mean values
    index = MeanIntervalIndex(intervals, mean_values=mean_values)
    index.name = 'size'

    # Check if the mean property returns the geometric mean
    expected_mean = gmean([index.right, index.left], axis=0)
    assert np.allclose(index.mean, expected_mean)

    # Change the name and check if the mean property returns the arithmetic mean
    index.name = 'other'
    expected_mean = (index.right + index.left) / 2
    assert np.allclose(index.mean, expected_mean)
