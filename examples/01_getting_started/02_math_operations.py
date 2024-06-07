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
# Create a mass-composition (mc) enabled Xarray Dataset
# -----------------------------------------------------
#
# We get some demo data in the form of a pandas DataFrame

df_data: pd.DataFrame = sample_data()
print(df_data.head())

# %%

# Construct a Sample object and standardise the chemistry variables

obj_smpl: Sample = Sample(df_data)
print(obj_smpl)

# %%
#
# Split the original Dataset and return the complement of the split fraction.
# Splitting does not modify the absolute grade of the input.

obj_smpl_split, obj_smpl_comp = obj_smpl.split(fraction=0.1, include_supplementary_data=True)
print(obj_smpl_split)

# %%
print(obj_smpl_comp)

# %%
#
# Add the split and complement parts using the mc.add method

obj_smpl_sum: Sample = obj_smpl_split + obj_smpl_comp
print(obj_smpl_sum)

# %%
#
# Confirm the sum of the splits is materially equivalent to the starting object.

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
#
# Math operations with rename
# The alternative syntax, methods rather than operands, allows renaming of the result object

obj_smpl_sum_renamed: Sample = obj_smpl.add(obj_smpl_split, name='Summed object')
print(obj_smpl_sum_renamed)

# %%
obj_smpl_sub_renamed: Sample = obj_smpl.sub(obj_smpl_split, name='Subtracted object')
print(obj_smpl_sum_renamed)

print('done')
