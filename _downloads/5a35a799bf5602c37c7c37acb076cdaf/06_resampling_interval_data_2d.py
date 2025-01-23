"""
Resampling 2D Interval Data
===========================

The Sink Float metallurgical test splits/fractionates samples by density.  The density fraction is often conducted by
size fraction, resulting in 2D fractionation (interval) data.

This example demonstrates how to resample 2D interval data using the IntervalSample object.

"""

import logging

# noinspection PyUnresolvedReferences
import numpy as np
import pandas as pd
import plotly.io

from elphick.geomet import IntervalSample
from elphick.geomet.datasets import datasets
from elphick.geomet.utils.pandas import MeanIntervalIndex
from elphick.geomet.utils.size import sizes_all

# %%
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S%z')

# %%
#
# Load Data
# ---------
#
# We load some real data.

df_data: pd.DataFrame = datasets.load_nordic_iron_ore_sink_float()
df_data

# %%
# The dataset contains size x assay, plus size x density x assay data.  We'll drop the size x assay data to leave the
# sink / float data.

df_sink_float: pd.DataFrame = df_data.dropna(subset=['density_lo', 'density_hi'], how='all').copy()
df_sink_float

# %%
# We will fill some nan values with assumptions
df_sink_float['size_passing'].fillna(1.0, inplace=True)
df_sink_float['density_lo'].fillna(1.5, inplace=True)
df_sink_float['density_hi'].fillna(5.0, inplace=True)

# %%
# Check the mass_pct by size

mass_check: pd.DataFrame = df_sink_float[['size_passing', 'size_retained', 'mass_pct']].groupby(
    ['size_passing', 'size_retained']).sum()
# check that all are 100
assert np.all(mass_check['mass_pct'] == 100)

mass_check

# %%
# This indicates that the mass_pct column is actually a density_mass_pct column.
# We'll rename that but also need to get the size_mass_pct values for those sizes from the size dataset

df_sink_float.rename(columns={'mass_pct': 'density_mass_pct'}, inplace=True)

df_size: pd.DataFrame = df_data.loc[np.all(df_data[['density_lo', 'density_hi']].isna(), axis=1), :].copy()
df_size.dropna(how='all', axis=1, inplace=True)
assert df_size['mass_pct'].sum() == 100

size_pairs = set(list((round(r, 5), round(p, 5)) for r, p in
                      zip(df_sink_float['size_retained'].values, df_sink_float['size_passing'].values)))
for r, p in size_pairs:
    df_sink_float.loc[(df_sink_float['size_retained'] == r) & (df_sink_float['size_passing'] == p), 'size_mass_pct'] = \
        df_size.loc[(df_size['size_retained'] == r) & (df_size['size_passing'] == p), 'mass_pct'].values[0]
# relocate the size_mass_pct column to the correct position, after size_passing
df_sink_float.insert(2, df_sink_float.columns[-1], df_sink_float.pop(df_sink_float.columns[-1]))
# add the mass_pct column
df_sink_float.insert(loc=6, column='mass_pct',
                     value=df_sink_float['density_mass_pct'] * df_sink_float['size_mass_pct'] / 100)
df_sink_float

# %%
# Create MeanIntervalIndexes
# --------------------------

size_intervals = pd.arrays.IntervalArray.from_arrays(df_sink_float['size_retained'], df_sink_float['size_passing'],
                                                     closed='left')
size_index = MeanIntervalIndex(size_intervals)
size_index.name = 'size'

density_intervals = pd.arrays.IntervalArray.from_arrays(df_sink_float['density_lo'], df_sink_float['density_hi'],
                                                        closed='left')
density_index = MeanIntervalIndex(density_intervals)
density_index.name = 'density'

df_sink_float.index = pd.MultiIndex.from_arrays([size_index, density_index])
df_sink_float.drop(columns=['size_retained', 'size_passing', 'density_lo', 'density_hi'], inplace=True)
df_sink_float

# %%
# Create a 2D IntervalSample
# --------------------------

interval_sample = IntervalSample(df_sink_float, name='SINK_FLOAT', moisture_in_scope=False, mass_dry_var='mass_pct')
print(interval_sample.is_2d_grid())
print(interval_sample.is_rectilinear_grid)

fig = interval_sample.plot_heatmap(components=['mass_pct'])
plotly.io.show(fig)

# %%
# Upsample
# --------
# We will upsample the data to a new grid

size_grid = sorted([s for s in sizes_all if s >= size_index.left.min() and s <= size_index.right.max()])
density_grid = np.arange(1.5, 5.1, 0.1)
new_grids: dict = {'size': size_grid, 'density': density_grid}

upsampled: IntervalSample = interval_sample.resample_2d(interval_edges=new_grids, precision=3)

pd.testing.assert_frame_equal(interval_sample.aggregate.reset_index(drop=True), upsampled.aggregate.reset_index(drop=True))

fig = upsampled.plot_heatmap(components=['mass_pct'])
plotly.io.show(fig)