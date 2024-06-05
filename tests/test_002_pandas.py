import numpy as np
import pandas as pd
import pytest

from geomet.utils.data import sample_data
from geomet.utils.pandas import mass_to_composition, composition_to_mass, weight_average
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

    expected_output: pd.DataFrame = pd.DataFrame(
        {'mass_dry': {0: 260.0}, 'FE': {0: 59.0}, 'SIO2': {0: 3.5153846153846153}, 'al2o3': {0: 1.8730769230769235},
         'LOI': {0: 4.0}}, index=res.index)

    pd.testing.assert_frame_equal(res, expected_output)


def test_weight_average_with_wet(test_data):
    res = weight_average(test_data, mass_wet='wet_mass', moisture_column_name='h2o')

    expected_output: pd.DataFrame = pd.DataFrame(
        {'wet_mass': {0: 300.0}, 'mass_dry': {0: 260.0}, 'h2o': {0: 13.333333333333334}, 'FE': {0: 59.0},
         'SIO2': {0: 3.5153846153846153}, 'al2o3': {0: 1.8730769230769235}, 'LOI': {0: 4.0}}, index=res.index)

    pd.testing.assert_frame_equal(res, expected_output)
