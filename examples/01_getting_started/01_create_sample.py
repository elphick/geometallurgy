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

sample: Sample = Sample(data=df, name='sample')
sample.data

# %%
# The `Sample` object has a `data` attribute that is a pandas DataFrame.  The column names are standardized
# to lower case.

