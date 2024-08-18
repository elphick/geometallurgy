"""
Resampling Interval Data
========================

Interval (or fractional) data is common in metallurgy and mineral processing.  Samples are sized using sieves
in a laboratory and each resultant fraction is often assayed to determine chemical composition.
The typical nomenclature is of the interval edges is size_retained, size passing - any particle within an interval
or fraction was retained by the lower sieve size, but passed the sieve size above it.

"""
import logging

import numpy as np
import pandas as pd
import plotly

from elphick.geomet import IntervalSample
from elphick.geomet.datasets.sample_data import size_by_assay
from elphick.geomet.utils.size import sizes_all

# %%
logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S%z',
                    )

# %%
#
# Create a MassComposition object
# -------------------------------
#
# We get some demo data in the form of a pandas DataFrame
# We create this object as 1D based on the pandas index

df_data: pd.DataFrame = size_by_assay()
df_data

# %%
#
# The size index is of the Interval type, maintaining the fractional information.

size_fractions: IntervalSample = IntervalSample(df_data, name='Sample', moisture_in_scope=False)
size_fractions.data

# %%

size_fractions.aggregate

# %%
# Plot the original sample intervals
# ----------------------------------
# First we'll plot the intervals of the original sample

fig = size_fractions.plot_intervals(variables=['mass_dry', 'Fe', 'SiO2', 'Al2O3'],
                                    cumulative=False)
plotly.io.show(fig)

# %%
#
# Size distributions are often plotted in the cumulative form. Cumulative passing is achieved by setting the
# direction = ascending.

fig = size_fractions.plot_intervals(variables=['mass_dry', 'Fe', 'SiO2', 'Al2O3'],
                                    cumulative=True, direction='ascending')
fig

# %%
# Resample on a defined grid
# --------------------------
# Now we will resample on a defined grid (interval edges) and view the resampled fractions

new_edges = np.unique(np.geomspace(1.0e-03, size_fractions.data.index.right.max(), 50))
new_coords = np.insert(new_edges, 0, 0)

upsampled: IntervalSample = size_fractions.resample_1d(interval_edges=new_edges, precision=3, include_original_edges=True)

fig = upsampled.plot_intervals(variables=['mass_dry', 'Fe', 'SiO2', 'Al2O3'], cumulative=False)
# noinspection PyTypeChecker
plotly.io.show(fig)

# %%
# Close inspection of the plot above reals some sharp dips for some mass intervals.  This is caused by those intervals
# being narrower than the adjacent neighbours, hence they have less absolute mass.
# This is a visual artefact only, numerically it is correct, as shown by the cumulative plot.

fig = upsampled.plot_intervals(variables=['mass_dry', 'Fe', 'SiO2', 'Al2O3'], cumulative=True, direction='ascending')
fig

# %%
# Up-sample by a factor
# --------------------
# We can up-sample each of the original fraction by a factor.  Since adjacent fractions are similar, the fractional
# plot is reasonably smooth.  Note however, that fraction widths are still different, caused by the original sieve
# selection.

upsampled_2: IntervalSample = size_fractions.resample_1d(interval_edges=10, precision=3)
fig = upsampled_2.plot_intervals(variables=['mass_dry', 'Fe', 'SiO2', 'Al2O3'], cumulative=False)
fig

# %%
# Up-sample to a sieve series
# --------------------------
# Standard sieve series account for the log nature of a particle size distribution.  This results in reasonably
# equal width intervals on a log scale.  We will up-sample to a standard sieve series and plot the results.

new_sizes = [s for s in sizes_all if s <= size_fractions.data.index.right.max()]
new_sizes

# %%
upsampled_3: IntervalSample = size_fractions.resample_1d(interval_edges=new_sizes, precision=3)
fig = upsampled_3.plot_intervals(variables=['mass_dry', 'Fe', 'SiO2', 'Al2O3'], cumulative=False)
fig

# %%
# Validate the head grade against the original sample

pd.testing.assert_frame_equal(size_fractions.aggregate.reset_index(drop=True),
                              upsampled.aggregate.reset_index(drop=True))

pd.testing.assert_frame_equal(size_fractions.aggregate.reset_index(drop=True),
                              upsampled_2.aggregate.reset_index(drop=True))

# %%
# Complete a round trip by converting the up-sampled objects back to the original intervals and validate.

orig_index = size_fractions.data.index
original_edges: np.ndarray = np.sort(np.unique(list(orig_index.left) + list(orig_index.right)))

downsampled: IntervalSample = upsampled.resample_1d(interval_edges=original_edges, precision=3)
downsampled_2: IntervalSample = upsampled_2.resample_1d(interval_edges=original_edges, precision=3)

pd.testing.assert_frame_equal(size_fractions.data, downsampled.data)
pd.testing.assert_frame_equal(size_fractions.data, downsampled_2.data)

