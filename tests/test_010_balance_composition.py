import numpy as np
import pandas as pd

from elphick.geomet import Sample
from fixtures import sample_data


def test_balance_composition(sample_data):
    sample: Sample = Sample(sample_data)

    df_expected: pd.DataFrame = sample.data

    sample.balance_composition()

    # since the sample has compliant records we expect the data to be unchanged
    pd.testing.assert_frame_equal(df_expected, sample.data)

    # modify to create an unbalanced record
    sample_data_unbalanced: pd.DataFrame = sample_data.copy()
    sample_data_unbalanced.loc[0, 'SiO2'] = 102.0

    sample2: Sample = Sample(sample_data_unbalanced)
    df_composition: pd.DataFrame = sample2.data[sample2.composition_columns]
    assert df_composition.loc[0, :].sum() > 100.0

    sample2.balance_composition()
    assert np.isclose(sample2.data[sample2.composition_columns].loc[0, :].sum(), 100)




