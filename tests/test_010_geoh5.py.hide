from pathlib import Path

from geoh5py import Workspace
from geoh5py.data import Data
from geoh5py.groups import ContainerGroup
from geoh5py.objects import NoTypeObject


def test_project_load():
    # load an existing geoh5 workspace
    workspace_path = (Path(__file__).parents[1] / "Geoscience_ANALYST_demo_workspace_and_data" /
                      "GeoscienceANALYST_demo.geoh5")
    if not workspace_path.exists():
        raise FileNotFoundError(f"File not found: {workspace_path}")

    workspace = Workspace(workspace_path)
    print('done')

def test_create_new_project():
    # create a new geoh5 workspace
    if Path("data/test_workspace.geoh5").exists():
        Path("data/test_workspace.geoh5").unlink()
    workspace: Workspace = Workspace.create("data/test_workspace.geoh5")

    # create a pandas dataframe
    import pandas as pd
    df = pd.DataFrame({
        "column1": [5, 10, 20],
        "column2": ["a", "b", "c"],
        "column3": pd.to_datetime(["2010", "2011", "2012"]),
    })

    # create a group
    group = ContainerGroup.create(workspace, name='my group')

    # create an Object
    obj = NoTypeObject.create(workspace, name='my object', parent=group)

    # create some data
    data1 = Data.create(workspace, name='column1', values=[1, 2, 3], entity=obj)
    data2 = Data.create(workspace, name='column2', values=['a', 'b', 'c'], entity=obj)
    data3 = Data.create(workspace, name='column3', values=[10, 20, 30], entity=obj)

    # save the workspace
    workspace.save_as("data/test_workspace_2.geoh5")
    print('done')