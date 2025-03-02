"""
Consuming OMF
=============

This example demonstrates how to consume an Open Mining Format file
"""
from pathlib import Path

import pandas as pd
import pooch
from omfpandas import OMFPandasReader

from elphick.geomet.block_model import BlockModel


# %%
# Load
# ----

# Cache an omf2 file locally
# filepath: Path = Path(pooch.retrieve(
#     url="https://github.com/elphick/omfpandas/blob/38-add-pyvista-plot-functionality/assets/v2/copper_deposit.omf",
#     known_hash=None,
#     path=Path(tempfile.gettempdir()) / "geometallurgy", ))

filepath: Path = Path('../../assets/copper_deposit.omf')

omfpr: OMFPandasReader = OMFPandasReader(filepath=filepath)
blocks: pd.DataFrame = omfpr.read_blockmodel(blockmodel_name='Block Model')

blocks.reset_index(level=['dx', 'dy', 'dz'], inplace=True)
blocks['volume'] = blocks.dx * blocks.dy * blocks.dz
blocks.reset_index().set_index(keys=['x', 'y', 'z', 'dx', 'dy', 'dz'], inplace=True)
blocks.rename(columns={'CU_pct': 'Cu'}, inplace=True)
blocks['mass'] = blocks['volume'] * 2.265

bm: BlockModel = BlockModel(name=filepath.stem, data=blocks, mass_dry_var='mass', moisture_in_scope=False)
p = bm.plot(scalar='Cu')
p.title = filepath.stem
p.show()
