"""
Consuming OMF
=============

This example demonstrates how to consume an Open Mining Format file
"""
import tempfile
from pathlib import Path

import pandas as pd
import pooch
from omfpandas import OMFPandasReader

from elphick.geomet.block_model import BlockModel

# %%
# Load
# ----

# Cache an omf2 file locally
filepath: Path = Path(pooch.retrieve(
    url="https://raw.githubusercontent.com/elphick/omfpandas/main/assets/copper_deposit.omf",
    known_hash=None,
    path=Path(tempfile.gettempdir()) / "geometallurgy"))

omfpr: OMFPandasReader = OMFPandasReader(filepath=filepath)
blocks: pd.DataFrame = omfpr.read_blockmodel(blockmodel_name='Block Model')

# %%
# Create mass
# -----------
# The mass of each block is calculated from the volume and a density of 2.265 g/cm3.

blocks.reset_index(level=['dx', 'dy', 'dz'], inplace=True)
blocks['volume'] = blocks.dx * blocks.dy * blocks.dz
blocks.reset_index().set_index(keys=['x', 'y', 'z', 'dx', 'dy', 'dz'], inplace=True)
blocks.rename(columns={'CU_pct': 'Cu'}, inplace=True)
blocks['mass'] = blocks['volume'] * 2.265

# %%
# Create a BlockModel
# -------------------
# The BlockModel is created from the DataFrame and visualised.
# This model can be used in a Flowsheet model to make metallurgical predictions and preserve the spatial context.

bm: BlockModel = BlockModel(name=filepath.stem, data=blocks, mass_dry_var='mass', moisture_in_scope=False)
p = bm.plot(scalar='Cu')
p.title = filepath.stem
p.show()
