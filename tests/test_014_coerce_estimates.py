import pandas as pd

from elphick.geomet import Sample
from elphick.geomet.flowsheet.stream import Stream
from elphick.geomet.utils.estimates import coerce_input_estimates, coerce_output_estimates
from fixtures import sample_data as test_data


def test_coerce_output_estimate(test_data):
    data: pd.DataFrame = test_data

    obj_input: Stream = Stream(data=data, name='feed', moisture_in_scope=False, mass_dry_var='mass_dry')

    df_est: pd.DataFrame = data.copy()
    df_est['mass_dry'] = df_est['mass_dry'] * 0.95
    df_est['Fe'] = df_est['Fe'] * 1.3
    df_est[['SiO2', 'Al2O3', 'LOI']] = df_est[['SiO2', 'Al2O3', 'LOI']] * 0.8
    obj_est: Stream = Stream(data=df_est, name='estimate', moisture_in_scope=False, mass_dry_var='mass_dry')

    obj_coerced: Stream = coerce_output_estimates(estimate_stream=obj_est, input_stream=obj_input, show_plot=True)

    expected: pd.DataFrame = pd.DataFrame.from_dict(
        {'mass_dry': {0: 85.5, 1: 76.0, 2: 85.5}, 'Fe': {0: 59.4, 1: 61.4842105263158, 2: 63.56842105263157},
         'SiO2': {0: 4.16, 1: 2.4800000000000004, 2: 1.7599999999999998},
         'Al2O3': {0: 2.4, 1: 1.36, 2: 0.7200000000000002}, 'LOI': {0: 4.0, 1: 3.2000000000000006, 2: 2.4},
         'group': {0: 'grp_1', 1: 'grp_1', 2: 'grp_2'}})
    expected.index.name = 'index'

    pd.testing.assert_frame_equal(obj_coerced.data, expected)


def test_coerce_input_estimate(test_data):
    data: pd.DataFrame = test_data

    obj_input: Stream = Stream(data=data, name='input', moisture_in_scope=False, mass_dry_var='mass_dry')

    df_est: pd.DataFrame = data.copy().drop(columns='wet_mass')
    df_est['mass_dry'] = df_est['mass_dry'] * 0.15
    df_est['Fe'] = df_est['Fe'] * 0.8
    df_est[['SiO2', 'Al2O3', 'LOI']] = df_est[['SiO2', 'Al2O3', 'LOI']] * 10
    obj_est: Stream = Stream(data=df_est, name='estimate', moisture_in_scope=False, mass_dry_var='mass_dry')

    obj_coerced: Stream = coerce_input_estimates(estimate_stream=obj_est, input_stream=obj_input, show_plot=True)

    # we know we did not have to change anything (in this particular test)
    expected: pd.DataFrame = df_est.copy()

    pd.testing.assert_frame_equal(obj_coerced.data, expected)
