from pathlib import Path

import numpy as np
import omfvista
import pandas as pd
import pooch
import pytest

from elphick.geomet.block_model import BlockModel


@pytest.fixture
def omf_model_path() -> Path:
    # Base URL and relative path
    base_url = "https://github.com/OpenGeoVis/omfvista/raw/master/assets/"
    relative_path = "test_file.omf"

    # Create a Pooch object
    p = pooch.create(
        path=pooch.os_cache("geometallurgy"),
        base_url=base_url,
        registry={relative_path: None}
    )

    # Use fetch method to download the file
    file_path = p.fetch(relative_path)

    return Path(file_path)


def test_load_from_omf(omf_model_path):
    msg = "mass_dry_var is not provided and cannot be calculated from mass_wet_var and moisture_var for Block Model"
    # with pytest.raises(ValueError, match=msg):
    #     bm: BlockModel = BlockModel.from_omf(omf_filepath=omf_model_path)

    msg = r"Column 'DMT' not found in the volume element"
    with pytest.raises(ValueError, match=msg):
        bm: BlockModel = BlockModel.from_omf(omf_filepath=omf_model_path, columns=['DMT'])

    bm: BlockModel = BlockModel.from_omf(omf_filepath=omf_model_path, columns=['CU_pct'])


    bm.plot('CU_pct').show(auto_close=False)
    print('done')


def test_to_omf(omf_model_path):
    block_model_filepath: Path = Path(__file__).parents[1] / "examples/04_block_model/block_model_copper.parquet"

    # Load the parquet file into a DataFrame
    df = pd.read_parquet(block_model_filepath)

    bm: BlockModel = BlockModel(data=df.rename(columns={'CU_pct': 'Cu'}).assign(**{'DMT': 2000}),
                                name='block_model', moisture_in_scope=False)
    bm._mass_data.head()
    bm.plot('Cu').show(auto_close=False)

    bm.to_omf(omf_filepath=Path('data/test_model.omf'))
    assert Path('data/test_model.omf').exists()

    # check some content using the OMFReader
    from omf import OMFReader
    reader = OMFReader('test_model.omf')
    omf_project = reader.get_project()
    assert omf_project.name == 'Block Model'
    assert len(omf_project.elements) == 1

    project = omfvista.load_project('data/test_model.omf')
    bm_loaded = project['Block Model']

    # check the variables in the model
    var_names = bm_loaded.array_names

    import pyvista as pv
    p = pv.Plotter()
    p.add_mesh_threshold(bm_loaded, 'Cu', show_edges=True, show_scalar_bar=True, cmap='viridis')
    p.show()

    print('done')



