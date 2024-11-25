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
# Load Data
# ---------
#
# We get some demo data in the form of a pandas DataFrame

df_data: pd.DataFrame = sample_data()
df_data.head()

# %%
#
# Create Sample
# -------------

obj_smpl: Sample = Sample(df_data)
print(obj_smpl)

# %%
#
# Parallel Plots
# --------------
# Create an interactive parallel plot.  Great for visualising and interactively filtering mass-composition data.

fig: Figure = obj_smpl.plot_parallel()
fig

# %%
#
# Create a parallel plot with only selected components and color

fig2 = obj_smpl.plot_parallel(vars_include=['wet_mass', 'H2O', 'Fe', 'group'], color='group')
fig2

# %%
# Ternary Diagram
# ---------------
#
# Create a ternary diagram for any 3 composition variables.

fig3 = obj_smpl.plot_ternary(variables=['SiO2', 'Al2O3', 'LOI'], color='group')
# noinspection PyTypeChecker
plotly.io.show(fig3)  # this call to show will set the thumbnail for use in the gallery

