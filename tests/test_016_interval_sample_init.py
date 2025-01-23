import numpy as np
import pandas as pd
import pytest

from elphick.geomet import IntervalSample
from elphick.geomet.datasets import datasets, Downloader


@pytest.fixture
def interval_data_1d() -> pd.DataFrame:
    """A 1d interval dataset of iron ore data.

    Returns:
        A pd.Dataframe of the following form:

                                           index  mass_dry   H2O   MgO  ...   CaO  Na2O   K2O   DHID
        interval_from interval_to                               ...
        26.60         26.85            6      2.12  0.35  0.07  ...  0.04  0.01  0.03  CBS02
        26.85         27.10            7      2.06  0.23  0.06  ...  0.04  0.01  0.03  CBS02
        27.70         28.00            9      1.91  0.23  0.06  ...  0.03  0.01  0.02  CBS02
        28.00         28.30           10      1.96  0.36  0.06  ...  0.04  0.01  0.02  CBS02
        28.60         28.95           12      2.06  0.40  0.05  ...  0.03  0.01  0.01  CBS02
        ...                          ...       ...   ...   ...  ...   ...   ...   ...    ...


    """

    df: pd.DataFrame = Downloader().load_data(datafile='iron_ore_sample_A072391.zip', show_report=False)
    df.set_index(['interval_from', 'interval_to'], inplace=True)
    return df


@pytest.fixture
def interval_data_2d() -> pd.DataFrame:
    """A 2d interval dataset of iron ore data.

    Returns:
        A pd.DataFrame of the following form:


                                                          size_mass_pct  ...      V
        size_retained size_passing density_lo density_hi                 ...
        0.100         1.000        1.5        2.7                  67.1  ...  0.003
                                   2.7        3.3                  67.1  ...  0.010
                                   3.3        5.0                  67.1  ...  0.012
        0.063         0.100        1.5        2.7                  12.7  ...  0.007
                                   2.7        3.3                  12.7  ...  0.008
                                   3.3        5.0                  12.7  ...  0.002
        0.040         0.063        1.5        2.7                   8.2  ...  0.013
                                   2.7        3.3                   8.2  ...  0.110
                                   3.3        5.0                   8.2  ...  0.013


    """

    # The dataset contains size x assay, plus size x density x assay data.  We'll drop the size x assay data to leave the
    # sink / float data.

    df_data: pd.DataFrame = datasets.load_nordic_iron_ore_sink_float()

    df_sink_float: pd.DataFrame = df_data.dropna(subset=['density_lo', 'density_hi'], how='all').copy()
    # We will fill some nan values with assumptions
    df_sink_float['size_passing'].fillna(1.0, inplace=True)
    df_sink_float['density_lo'].fillna(1.5, inplace=True)
    df_sink_float['density_hi'].fillna(5.0, inplace=True)

    # This indicates that the mass_pct column is actually a density_mass_pct column.
    # We'll rename that but also need to get the size_mass_pct values for those sizes from the size dataset

    df_sink_float.rename(columns={'mass_pct': 'density_mass_pct'}, inplace=True)

    df_size: pd.DataFrame = df_data.loc[np.all(df_data[['density_lo', 'density_hi']].isna(), axis=1), :].copy()
    df_size.dropna(how='all', axis=1, inplace=True)

    size_pairs = set(list((round(r, 5), round(p, 5)) for r, p in
                          zip(df_sink_float['size_retained'].values, df_sink_float['size_passing'].values)))
    for r, p in size_pairs:
        df_sink_float.loc[
            (df_sink_float['size_retained'] == r) & (df_sink_float['size_passing'] == p), 'size_mass_pct'] = \
            df_size.loc[(df_size['size_retained'] == r) & (df_size['size_passing'] == p), 'mass_pct'].values[0]
    # relocate the size_mass_pct column to the correct position, after size_passing
    df_sink_float.insert(2, df_sink_float.columns[-1], df_sink_float.pop(df_sink_float.columns[-1]))
    # add the mass_pct column
    df_sink_float.insert(loc=6, column='mass_pct',
                         value=df_sink_float['density_mass_pct'] * df_sink_float['size_mass_pct'] / 100)

    df_sink_float.set_index(['size_retained', 'size_passing', 'density_lo', 'density_hi'], inplace=True)

    return df_sink_float


def test_interval_sample_init_1d(interval_data_1d):
    # confirm that a 1d interval dataset can be initialized
    df_1 = interval_data_1d
    obj: IntervalSample = IntervalSample(df_1, name='Drill program')
    assert isinstance(obj, IntervalSample)


def test_interval_sample_init_2d(interval_data_2d):
    # confirm that a 1d interval dataset can be initialized
    df_2 = interval_data_2d
    obj: IntervalSample = IntervalSample(df_2, name='Sink Float', moisture_in_scope=False, mass_dry_var='mass_pct')
    assert isinstance(obj, IntervalSample)
