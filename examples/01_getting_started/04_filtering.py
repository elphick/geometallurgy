"""
Filtering
=========

Filtering is often required to reduce the data of a MassComposition object to a specific subset of interest.

Both individual objects can be filtered, as can multiple objects contained within a Flowsheet object.

"""

import pandas as pd

from elphick.geomet import MassComposition, Sample
from elphick.geomet.flowsheet import Flowsheet
from elphick.geomet.datasets.sample_data import sample_data

# %%
#
# Create a Sample object
# ----------------------
#
# We get some demo data in the form of a pandas DataFrame

df_data: pd.DataFrame = sample_data()
df_data

# %%
#
# Construct the object

obj_mc: Sample = Sample(df_data, name='demo')
obj_mc.data

# %%
#
# Filtering Single Objects
# ------------------------
#
# One of the most common subsets is one that contains records above a particular grade.
# The method used to filter is called query, for consistency with the pandas method that execute the same.

obj_1: Sample = obj_mc.query('Fe>58')
obj_1.data

# %%
# Notice that the record with an Fe value below 58 has been removed.

# %%
#
# Filtering Multiple Objects
# --------------------------
#
# Multiple objects can be loaded into a Flowsheet.  We'll make a small network to demonstrate.

obj_one, obj_two = obj_mc.split(fraction=0.6, name_1='one', name_2='two')

fs: Flowsheet = Flowsheet.from_objects([obj_mc, obj_one, obj_two], name='Network')

# %%
# The weighted mean mass-composition of each object/edge/stream in the network can be reported out with the
# report method.

fs.report()

# %%
# Now we'll filter as we did before, though we must specify which object the query criteria is to be applied to.

fs.query(mc_name='demo', query_string='Fe>58').report()
