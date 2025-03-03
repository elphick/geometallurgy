"""
Math Operations
===============

Demonstrate splitting and math operations that preserve the mass balance of components.
"""

# %%

import pandas as pd

from elphick.geomet import Sample
from elphick.geomet.utils.data import sample_data

# %%
#
# Load Data
# ---------
#
# We get some demo data in the form of a pandas DataFrame

df_data: pd.DataFrame = sample_data()
print(df_data.head())

# %%
#
# Create Sample
# -------------

obj_smpl: Sample = Sample(df_data, name='sample')
print(obj_smpl)

# %%
# Split the Sample
# ----------------
#
# Split the Sample and return the complement of the split fraction.
# Splitting does not modify the absolute grade of the input.

obj_smpl_split, obj_smpl_comp = obj_smpl.split(fraction=0.1, include_supplementary_data=True)
print(obj_smpl_split)

# %%
print(obj_smpl_comp)

# %%
#
# Operands
# --------
#
# The math operands +, -, / are supported for the Sample object.
# We'll add the split and complement parts.

obj_smpl_sum: Sample = obj_smpl_split + obj_smpl_comp
print(obj_smpl_sum)

# %%
#
# Notice the name of the resultant sample object is None.
# We'll confirm the sum of the splits is materially equivalent to the starting object.

pd.testing.assert_frame_equal(obj_smpl.data, obj_smpl_sum.data)

# %%
#
# Add finally add and then subtract the split portion to the original object, and check the output.

obj_smpl_sum: Sample = obj_smpl + obj_smpl_split
obj_smpl_minus: Sample = obj_smpl_sum - obj_smpl_split
pd.testing.assert_frame_equal(obj_smpl_minus.data, obj_smpl.data)
print(obj_smpl_minus)

# %%
#
# Demonstrate division.

obj_smpl_div: Sample = obj_smpl_split / obj_smpl
print(obj_smpl_div)

# %%
# Methods
# -------
#
# Performing math operations with methods allows the resultant objects to be renamed.

obj_smpl_sum_renamed: Sample = obj_smpl.add(obj_smpl_split, name='Summed object')
print(obj_smpl_sum_renamed)

# %%
obj_smpl_sub_renamed: Sample = obj_smpl.sub(obj_smpl_split, name='Subtracted object')
print(obj_smpl_sub_renamed)
