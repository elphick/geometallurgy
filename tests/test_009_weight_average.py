import pandas as pd

from elphick.geomet import Sample
from fixtures import sample_data


def test_weight_average(sample_data):
    sample: Sample = Sample(sample_data)
    df_expected: pd.DataFrame = sample.aggregate
    df_avg: pd.DataFrame = sample.weight_average()
    pd.testing.assert_frame_equal(df_expected, df_avg)


def test_weight_average_grouped(sample_data):
    sample: Sample = Sample(sample_data)
    d_expected: dict = {'wet_mass': {'grp_1': 190.0, 'grp_2': 110.0}, 'mass_dry': {'grp_1': 170.0, 'grp_2': 90.0},
                        'H2O': {'grp_1': 10.526315789473683, 'grp_2': 18.181818181818183},
                        'Fe': {'grp_1': 57.94117647058824, 'grp_2': 61.0},
                        'SiO2': {'grp_1': 4.211764705882353, 'grp_2': 2.2},
                        'Al2O3': {'grp_1': 2.3882352941176475, 'grp_2': 0.9000000000000001},
                        'LOI': {'grp_1': 4.529411764705882, 'grp_2': 3.0000000000000004},
                        }
    df_expected: pd.DataFrame = pd.DataFrame.from_dict(d_expected)
    df_expected.index.name = 'group'
    df_avg: pd.DataFrame = sample.weight_average(group_by='group')
    pd.testing.assert_frame_equal(df_expected, df_avg)
