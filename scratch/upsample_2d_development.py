"""
Upsample the 2D IntervalSample to a finer grid
==============================================

Replicating results from the paper by :cite:p:`2002,salama`.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from elphick.geomet import IntervalSample

# %%
# Load the data
# -------------
# The data as loaded corresponds to dataset `e` in the paper (experimental).

data_path = Path('../assets/canadian_coal_sink_float.csv')
df_e: pd.DataFrame = pd.read_csv(data_path)

df_e['mass_pct'] = df_e['size_mass_pct'] * df_e['density_mass_pct'] / 100
assert np.isclose(df_e['mass_pct'].sum(), 100.0)

# %%
# Downsample to create `el` in the paper (experimental labelled)

df_e['i'] = np.nan
df_e.loc[df_e['size_retained'] == 0.2, 'i'] = 1
df_e.loc[(df_e['size_retained'] >= 0.6) & (df_e['size_retained'] <= 2.0), 'i'] = 2
df_e.loc[df_e['size_retained'] > 2.0, 'i'] = 3
df_e['j'] = np.nan
df_e.loc[df_e['density_lo'] <= 1.35, 'j'] = 1
df_e.loc[(df_e['density_lo'] >= 1.40) & (df_e['density_lo'] <= 1.50), 'j'] = 2
df_e.loc[df_e['density_lo'] == 1.60, 'j'] = 3
df_e.loc[df_e['density_lo'] == 1.80, 'j'] = 4
df_e.loc[df_e['density_lo'] > 1.80, 'j'] = 5
# concatenate i and j to create the label
df_e['label'] = df_e['i'].astype(int).astype(str) + df_e['j'].astype(int).astype(str)
df_e.drop(columns=['i', 'j'], inplace=True)
# set the index
df_e.set_index(['size_retained', 'size_passing', 'density_lo', 'density_hi'], inplace=True)

# %%
# Create the Objects
# ------------------

sample_e: IntervalSample = IntervalSample(df_e, name='e', moisture_in_scope=False, mass_dry_var='mass_pct')
sample_e

# %%
df_el: pd.DataFrame = sample_e.weight_average(group_by='label')
# create the down-sampled index
df_first: pd.DataFrame = df_e.reset_index()[['size', 'density', 'label']].groupby('label').first()
df_last: pd.DataFrame = df_e.reset_index()[['size', 'density', 'label']].groupby('label').last()
df_intervals: pd.DataFrame = pd.merge(df_first, df_last, on='label', suffixes=('_from', '_to'))
df_intervals['size'] = pd.IntervalIndex.from_arrays(
    np.min([pd.IntervalIndex(df_intervals['size_from']).left, pd.IntervalIndex(df_intervals['size_to']).left], axis=0),
    np.max([pd.IntervalIndex(df_intervals['size_from']).right, pd.IntervalIndex(df_intervals['size_to']).right],
           axis=0),closed='left')
df_intervals['density'] = pd.IntervalIndex.from_arrays(
    np.min([pd.IntervalIndex(df_intervals['density_from']).left, pd.IntervalIndex(df_intervals['density_to']).left], axis=0),
    np.max([pd.IntervalIndex(df_intervals['density_from']).right, pd.IntervalIndex(df_intervals['density_to']).right],
           axis=0), closed='left')
df_el = df_el.join(df_intervals[['size', 'density']], on='label').set_index(['size', 'density'])
# sort by size descending, density increasing (for consistency with sample e)
df_el.sort_index(ascending=[False, True], inplace=True)

sample_el: IntervalSample = IntervalSample(df_el, name='el', moisture_in_scope=False, mass_dry_var='mass_pct')
sample_el

pass
