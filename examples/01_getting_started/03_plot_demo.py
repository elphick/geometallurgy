"""
Plot Demo
=========

Demonstrating the plot methods.
"""

import pandas as pd
import plotly
from plotly.graph_objs import Figure
from elphick.geomet import Sample
from elphick.geomet.utils.data import sample_data

# %%
#
# Create a Sample object
# ----------------------
#
# We get some demo data in the form of a pandas DataFrame

df_data: pd.DataFrame = sample_data()
print(df_data.head())

# %%
#
# Construct a Sample object and standardise the chemistry variables

obj_smpl: Sample = Sample(df_data)
print(obj_smpl)

# %%
#
# Create an interactive parallel plot

fig: Figure = obj_smpl.plot_parallel()
fig

# %%
#
# Create an interactive parallel plot with only the components

fig2 = obj_smpl.plot_parallel(vars_include=['wet_mass', 'H2O', 'Fe'])
fig2

# %%
#
# Create a parallel plot with color

fig3 = obj_smpl.plot_parallel(color='group')
fig3

# %%
#
# Create a ternary diagram for 3 composition variables

fig4 = obj_smpl.plot_ternary(variables=['SiO2', 'Al2O3', 'LOI'], color='group')
# noinspection PyTypeChecker
plotly.io.show(fig4)  # this call to show will set the thumbnail for use in the gallery

