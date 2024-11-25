"""
Filtering Flowsheets
====================

Flowsheets contain a network of streams and operations.  The streams contain data in the form of a pandas DataFrame.
This example demonstrates how to filter the data across a flowsheet.
The streams on a flowsheet are all congruent by definition - that is they have the same index.
Any applied filter is applied across all streams to ensure the streams remain congruent.

"""

import pandas as pd

from elphick.geomet.flowsheet import Flowsheet
from elphick.geomet.flowsheet.stream import Stream
from elphick.geomet.utils.data import sample_data

# %%
#
# Create some Sample objects
# --------------------------
#
# Create an object, and split it to create two more objects.

df_data: pd.DataFrame = sample_data()
obj_strm: Stream = Stream(df_data, name='Feed')
obj_strm_1, obj_strm_2 = obj_strm.split(0.4, name_1='stream 1', name_2='stream 2')

# %%
# View the Feed records
obj_strm.data

# %%
#
# Create a Flowsheet object
# -------------------------
#
# This requires passing an Iterable of Sample objects

fs: Flowsheet = Flowsheet.from_objects([obj_strm, obj_strm_1, obj_strm_2])
fs.report()

# %%
# Filter using `query`
# --------------------
# In this case, we wish to avoid mutating the original data, so we set `inplace=False`.

fs_query: Flowsheet = fs.query(expr='Fe>58', stream_name='Feed', inplace=False)
fs_query.report()

# %%
fs_query.get_edge_by_name('Feed').data

# %%
# Filter using `filter_by_index`
# ------------------------------
# When more complex filtering is required, we can filter by an index.

new_index: pd.Index = obj_strm.data.index[1:]
fs_filter: Flowsheet = fs.filter_by_index(index=new_index, inplace=False)
fs_filter.report()

# %%
fs_filter.get_edge_by_name('Feed').data
