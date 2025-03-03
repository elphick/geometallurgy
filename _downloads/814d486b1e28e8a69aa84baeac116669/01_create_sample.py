"""
Create Sample
=============

The base object is a `Sample`, so let's create one
"""
import pandas as pd
from elphick.geomet.utils.data import sample_data
from elphick.geomet import Sample

# %%
# Load Data
# ---------
# First, let's load some toy data.  For demonstration this toy data has mixed case column names.

df: pd.DataFrame = sample_data(include_moisture=False)
df

# %%
# Create Sample
# -------------
# The `Sample` object has a `data` attribute that is a pandas DataFrame.  Where column names are recognised
# as components the case is converted to the represent the chemical symbols.

sample: Sample = Sample(data=df, name='sample')
sample.data

# %%
# Aggregation
# -----------
# The aggregate of the mass-composition is available via the `aggregate` property of the `Sample` object.

sample.aggregate

# %%
# For aggregation/weight-average by a grouping variable we can use the `weight_average` method.

sample.weight_average(group_by='group')
