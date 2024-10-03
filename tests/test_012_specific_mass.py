import numpy as np
import pandas as pd

from elphick.geomet import IntervalSample
from elphick.geomet.datasets import Downloader


def test_specific_mass():
    data = Downloader().load_data(datafile='iron_ore_sample_A072391.zip', show_report=False)
    obj: IntervalSample = IntervalSample(data=data, name='interval_sample')
    df_specific_mass: pd.DataFrame = obj._specific_mass()
    assert np.isclose(df_specific_mass[obj.mass_dry_var].sum(), 1.0)
