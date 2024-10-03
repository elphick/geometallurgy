_blockmodel_imports = None


# Define the Extras class to encapsulate the imported modules
class BlockmodelExtras:
    def __init__(self, omf, omfvista, pv):
        self.omf = omf
        self.omfvista = omfvista
        self.pv = pv


def import_blockmodel_packages():
    """Helper method to safely import (only once) the blockmodel packages."""
    global _blockmodel_imports

    # Optional imports
    try:
        import omf
        import omfvista
        import pyvista as pv
        from pyvista import CellType
    except ImportError as e:
        raise ImportError("Optional packages omfpandas or omfvista is not installed."
                          "Please install it to use this feature.") from e

    if _blockmodel_imports is None:
        try:
            import omf
            import omfvista
            import pyvista as pv
            _blockmodel_imports = (omf, omfvista, pv)
        except ImportError:
            raise ImportError("Failed to import blockmodel related packages. "
                              "Consider executing: 'poetry install --extras blockmodel'")
    return _blockmodel_imports
