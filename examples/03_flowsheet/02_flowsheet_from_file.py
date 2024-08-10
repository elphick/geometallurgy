"""
Flowsheet from File
===================

It is possible to create a Flowsheet object from a file.  This example demonstrates how to create a Flowsheet object
from a file.

"""
from copy import deepcopy
from pathlib import Path
from typing import Dict

import pandas as pd
from matplotlib import pyplot as plt

from elphick.geomet import Stream, Flowsheet
from elphick.geomet.operation import OP, Operation
from elphick.geomet.utils.data import sample_data

# %%
#
# YAML file
# ---------
#
# The yaml file needs to follow this structure.

yaml_filepath: Path = Path('./../../elphick/geomet/config/flowsheet_example.yaml')
yaml_config: str = yaml_filepath.read_text()
yaml_config

# %%
# Create Flowsheet
# ----------------

fs: Flowsheet = Flowsheet.from_yaml(yaml_filepath)
fs

# %%
# The Flowsheet object can be visualized as a network.
# The streams are not yet populated, so are shown in red.
fs.plot()
plt

# %%
# Load the data
# -------------
# Load data onto two of the three streams.

df_feed: pd.DataFrame = sample_data()
obj_feed: Stream = Stream(df_feed, name='Feed')

# assume the Coarse stream is 0.4 of the Feed stream mass
df_coarse: pd.DataFrame = df_feed.copy()
df_coarse['wet_mass'] = df_coarse['wet_mass'] * 0.4
df_coarse['mass_dry'] = df_coarse['mass_dry'] * 0.4
obj_coarse: Stream = Stream(df_coarse, name='Coarse')

fs.set_stream_data(stream_data={'Feed': obj_feed, 'Coarse': obj_coarse})

# %%
fs.plot()
plt

# %%
# Solve the flowsheet
# -------------------

fs.solve()
fs.plot()
plt

# %%
# Plot the mass balance
# ---------------------

fig = fs.table_plot(plot_type='network')
fig.show()
