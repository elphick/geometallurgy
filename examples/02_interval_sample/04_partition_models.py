"""
Partition Models
================

Partition models, (a.k.a. partition curves in the 1 dimensional case) define the separation of a unit operation /
process.

The Partition Number (K) is represents the probability that a particle will report to the concentrate stream.

.. math::
    K = \\frac{{m_{concentrate}}}{{m_{feed}}}

Note that the term “concentrate” in the numerator can be misleading, as it does not always accurately represent
the desired or obtained fraction. For instance, in a screening operation, the numerator might be the “oversize”
rather than the “concentrate.” To create a more universally applicable term for the numerator, we can redefine
it as the “preferred fraction.” This new terminology allows for a more inclusive and adaptable equation:

.. math::
    K = \\frac{{m_{preferred}}}{{m_{feed}}}

Consider a desliming cyclone that aims to separate a slurry at 150 micron.  The preferred stream is defined as
the Underflow (UF), since that is the "stream of interest" in our simple example.

..  Admonition:: TODO

    Add a reference to partition curves.

"""
from functools import partial

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
from scipy.interpolate import PchipInterpolator

from elphick.geomet import IntervalSample
from elphick.geomet.datasets.sample_data import size_by_assay
from elphick.geomet.flowsheet import Flowsheet
from elphick.geomet.utils.partition import napier_munn
from elphick.geomet.utils.pandas import calculate_partition

# sphinx_gallery_thumbnail_number = -1

# %%
#
# Create a mass-composition object
# --------------------------------
#
# We get some demo data in the form of a pandas DataFrame

df_data: pd.DataFrame = size_by_assay()
mc_feed: IntervalSample = IntervalSample(df_data, name='size sample', moisture_in_scope=False)
print(mc_feed)

# %%
# Define and Apply the Partition
# ------------------------------
#
# We partially initialise the partition function
# The dim argument is added to inform the split method which dimension to apply the function/split to

part_cyclone = partial(napier_munn, d50=0.150, ep=0.1, dim='size')

# %%
# Separate the object using the defined partitions.  UF = Underflow, OF = Overflow

mc_uf, mc_of = mc_feed.split_by_partition(partition_definition=part_cyclone, name_1='underflow', name_2='overflow')
fs: Flowsheet = Flowsheet().from_objects([mc_feed, mc_uf, mc_of])

fig = fs.table_plot(table_pos='left',
                    sankey_color_var='Fe', sankey_edge_colormap='copper_r', sankey_vmin=50, sankey_vmax=70)
fig

# %%
# We'll now get the partition data from the objects

df_partition: pd.DataFrame = mc_feed.calculate_partition(preferred=mc_uf)
df_partition

# %%
# Create an interpolator from the data.  As a Callable, the spline can be used to split a MassComposition object.

da = np.linspace(0.01, df_partition.index.right.max(), num=500)
spline_partition = PchipInterpolator(x=df_partition.sort_index()['da'], y=df_partition.sort_index()['K'])
pn_extracted = spline_partition(da)

# %%
# Plot the extracted data, and the spline on the input partition curve to visually validate.

pn_original = part_cyclone(da)

fig = go.Figure(go.Scatter(x=da, y=pn_original, name='Input Partition', line=dict(width=5, color='DarkSlateGrey')))
fig.add_trace(go.Scatter(x=df_partition['da'], y=df_partition['K'], name='Extracted Partition Data', mode='markers',
                         marker=dict(size=12, color='red', line=dict(width=2, color='DarkSlateGrey'))))
fig.add_trace(
    go.Scatter(x=da, y=pn_extracted, name='Extracted Partition Curve', line=dict(width=2, color='red', dash='dash')))

fig.update_xaxes(type="log")
fig.update_layout(title='Partition Round Trip Check', xaxis_title='da', yaxis_title='K', yaxis_range=[0, 1.05])

# noinspection PyTypeChecker
plotly.io.show(fig)

# %%
# There are minor differences between the interpolated spline curve and the Napier-Munn input partition.
# However, the data points all lie on both curves.

# %%
# Pandas Function
# ---------------
#
# The partition functionality is available as a pandas function also.

df_partition_2: pd.DataFrame = mc_feed.data.pipe(calculate_partition, df_preferred=mc_uf.data, col_mass_dry='mass_dry')
df_partition_2

# %%
pd.testing.assert_frame_equal(df_partition, df_partition_2)
