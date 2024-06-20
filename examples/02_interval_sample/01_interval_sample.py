"""
Interval Data
=============

This example adds a second dimension.  The second dimension is an interval, of the form interval_from, interval_to.
It is also known as binned data, where each 'bin' is bounded between and upper and lower limit.

An interval is relevant in geology, when analysing drill hole data.

Intervals are also encountered in metallurgy, but in that discipline they are often called fractions,
e.g. size fractions.  In that case the typical nomenclature is size_retained, size passing, since the data
originates from a sieve stack.

"""
import logging

import pandas as pd
import plotly.io
from matplotlib import pyplot as plt

from elphick.geomet import Sample, IntervalSample
from elphick.geomet.data.downloader import Downloader
from elphick.geomet.utils.pandas import weight_average
import plotly.graph_objects as go

# %%
logging.basicConfig(level=logging.INFO,
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

iron_ore_sample_data: pd.DataFrame = Downloader().load_data(datafile='iron_ore_sample_A072391.zip', show_report=False)
df_data: pd.DataFrame = iron_ore_sample_data
df_data.head()

# %%

obj_mc: Sample = Sample(df_data, name='Drill program')
obj_mc

# %%

obj_mc.aggregate

# %%
#
# Use the normal pandas groupby-apply as needed.  Here we leverage the weight_average function
# from utils.pandas

hole_average: pd.DataFrame = obj_mc.data.groupby('DHID').apply(weight_average)
hole_average

# %%
#
# We will now make a 2D dataset using DHID and the intervals.
df_data['DHID'] = df_data['DHID'].astype('category')
df_data = df_data.reset_index(drop=True).set_index(['DHID', 'interval_from', 'interval_to'])

obj_mc_2d: IntervalSample = IntervalSample(df_data, name='Drill program')
print(obj_mc_2d)

# %%
obj_mc_2d.aggregate

# %%
obj_mc_2d.data.groupby('DHID').apply(weight_average, **{'mass_wet': 'mass_wet', 'moisture_column_name': 'H2O'})

# %%
#
# View some plots
#

fig: go.Figure = obj_mc_2d.plot_parallel(color='DHID')
plotly.io.show(fig)
