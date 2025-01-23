"""
Upsample the 2D IntervalSample to a finer grid
==============================================

Replicating results from the paper by :cite:p:`2002,salama`.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

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
           axis=0), closed='left')
df_intervals['density'] = pd.IntervalIndex.from_arrays(
    np.min([pd.IntervalIndex(df_intervals['density_from']).left, pd.IntervalIndex(df_intervals['density_to']).left],
           axis=0),
    np.max([pd.IntervalIndex(df_intervals['density_from']).right, pd.IntervalIndex(df_intervals['density_to']).right],
           axis=0), closed='left')
df_el = df_el.join(df_intervals[['size', 'density']], on='label').set_index(['size', 'density'])
# sort by size descending, density increasing (for consistency with sample e)
df_el.sort_index(ascending=[False, True], inplace=True)

sample_el: IntervalSample = IntervalSample(df_el, name='el', moisture_in_scope=False, mass_dry_var='mass_pct')
sample_el

# %%
# Upsample Mass Data
# ------------------
# At this stage we will focus only on the dry mass, and ignore components.
# The optimisation will be performed on specific_mass, where the mass is normalised to 1.0.
# The mass data will be upsampled to a finer grid.

# create the upsampled index (we can steal this from el)
upsampled_fraction_index = sample_e.data.index

# define the cost function
def fn_cost(specific_mass_array: np.ndarray) -> float:
    """The cost function

    Calculate the cost function for the specific mass array.
    The objective is to minimise the gradient between successive intervals in the direction of i, j, and ij.

    Args:
        specific_mass_array: A 2d array of specific mass values

    Returns:
        The cost function value (lower is better)
    """

    # calculate the gradient in the i direction
    grad_i = np.diff(specific_mass_array, axis=0)
    # calculate the gradient in the j direction
    grad_j = np.diff(specific_mass_array, axis=1)
    # calculate the gradient in the ij direction
    grad_ij = np.diff(specific_mass_array, axis=0) + np.diff(specific_mass_array, axis=1)
    # calculate the cost by summing the squared gradients
    cost = np.sum(grad_i ** 2) + np.sum(grad_j ** 2) + np.sum(grad_ij ** 2)
    return cost


def create_constraints(upsampled_fraction_index: pd.Index, measured_specific_mass: pd.DataFrame):
    """Create the optimisation constraints

    Args:
        upsampled_fraction_index: The pd.Index of the upsampled fractions
        measured_specific_mass: The measured (observed) specific mass data

    Returns:
        A list of constraints defined by ensuring the suitable subset of upsampled masses sum to the observed mass
        (for each 2d fraction)

    """
    constraints = []
    for i, row in measured_specific_mass.iterrows():
        # create the constraint
        constraint = {'type': 'eq',
                      'fun': lambda x: np.sum(x[upsampled_fraction_index] * upsampled_fraction_index) - row['mass_pct']}
        constraints.append(constraint)
    return constraints


def create_initial_guesses(upsampled_fraction_index: pd.Index, measured_specific_mass: pd.DataFrame):
    # create the initial guesses of the upsampled data, by leveraging the measured data
    initial_specific_mass = np.interp(upsampled_fraction_index, measured_specific_mass.index,
                                      measured_specific_mass['mass_pct'])
    return initial_specific_mass


# Example usage
measured_specific_mass = sample_el._specific_mass()[['mass_pct']]

initial_guess = create_initial_guesses(upsampled_fraction_index, measured_specific_mass)
constraints = create_constraints(upsampled_fraction_index, measured_specific_mass)

result = minimize(fn_cost, initial_guess, constraints=constraints, method='SLSQP')
optimized_specific_mass = result.x.reshape(upsampled_fraction_index.shape)
