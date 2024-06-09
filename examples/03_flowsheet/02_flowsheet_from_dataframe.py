"""
Create Network
==============

Create a network from a DataFrame
"""

import pandas as pd
import plotly

from elphick.mass_composition.datasets.sample_data import size_by_assay_2
from elphick.mass_composition.flowsheet import Flowsheet

# %%
#
# Load a dataframe containing 3 streams
# -------------------------------------
#
# The dataframe is tall, indexed by size fractions and stream name

df_data: pd.DataFrame = size_by_assay_2()
df_data

# %%
# Create a network

fs: Flowsheet = Flowsheet.from_dataframe(df=df_data, mc_name_col='name')
fig = fs.table_plot(plot_type='sankey', table_pos='left', table_area=0.3)
fig

# %%
# The network has no knowledge of the stream relationships, so we need to create those relationships.

fs.set_stream_parent(stream='coarse', parent='feed')
fs.set_stream_parent(stream='fine', parent='feed')

fig = fs.table_plot(plot_type='sankey', table_pos='left', table_area=0.3)
# noinspection PyTypeChecker
plotly.io.show(fig)  # this call to show will set the thumbnail for use in the gallery
