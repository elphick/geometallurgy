import numpy as np
import pandas as pd
import pytest

from elphick.geomet import IntervalSample
from elphick.geomet.datasets import datasets
from elphick.geomet.utils.size import sizes_all


@pytest.fixture
def data_2d_regular() -> pd.DataFrame:
    data: pd.DataFrame = datasets.load_nordic_iron_ore_sink_float()
    # drop the fractions that do not include density
    data.dropna(subset=['density_lo', 'density_hi'], how='all', inplace=True)
    # add the boundary interval values
    data['density_lo'].fillna(1.5, inplace=True)
    data['density_hi'].fillna(5.0, inplace=True)
    data['size_passing'].fillna(0.5, inplace=True)
    # set the index to be the intervals
    data.set_index(['size_retained', 'size_passing', 'density_lo', 'density_hi'], inplace=True)
    return data


def test_upsample_2d(data_2d_regular):
    data: pd.DataFrame = data_2d_regular

    obj: IntervalSample = IntervalSample(data=data, name='sink_float',
                                         moisture_in_scope=False, mass_dry_var='mass_pct')

    density_range = [obj.mass_data.index.get_level_values('density').left.min(),
                     obj.mass_data.index.get_level_values('density').right.max()]
    interval_edges = {'size': sizes_all[::-1], 'density': list(np.arange(density_range[0], density_range[1]+0.1, step=0.1))}

    with pytest.raises(ValueError, match="The supplied size grid contains values lower than the minimum in the sample."):
        obj_upsampled: IntervalSample = obj.resample_2d(interval_edges=interval_edges)

    # limit the lower sizes to the minimum of the sample
    min_val = obj.mass_data.index.get_level_values('size').left.min()
    interval_edges['size'] = [s for s in  interval_edges['size'] if s > min_val]

    obj_upsampled: IntervalSample = obj.resample_2d(interval_edges=interval_edges)

    obj_upsampled.plot_heatmap(components=['mass_pct']).show()

    df_specific_mass: pd.DataFrame = obj_upsampled._specific_mass()

    # assert np.isclose(df_specific_mass[obj.mass_dry_var].sum(), 1.0)
