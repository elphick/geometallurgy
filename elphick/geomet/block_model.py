import logging
from functools import wraps
from pathlib import Path
from typing import Optional, Union, Literal, TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats

from elphick.geomet import extras
from elphick.geomet.base import MassComposition
from elphick.geomet.extras import BlockmodelExtras
from elphick.geomet.utils.timer import log_timer

if TYPE_CHECKING:
    import pyvista as pv


# Modify the import_extras decorator
def import_extras(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        omfpandas, omfvista, pv = extras.import_blockmodel_packages()
        extras_instance = BlockmodelExtras(omfpandas, omfvista, pv)
        return func(*args, imports=extras_instance, **kwargs)

    return wrapper


class BlockModel(MassComposition):
    def __init__(self,
                 data: Optional[pd.DataFrame] = None,
                 name: Optional[str] = None,
                 moisture_in_scope: bool = True,
                 mass_wet_var: Optional[str] = None,
                 mass_dry_var: Optional[str] = None,
                 moisture_var: Optional[str] = None,
                 component_vars: Optional[list[str]] = None,
                 composition_units: Literal['%', 'ppm', 'ppb'] = '%',
                 components_as_symbols: bool = True,
                 ranges: Optional[dict[str, list]] = None,
                 config_file: Optional[Path] = None):

        if data is not None:
            if isinstance(data.index, pd.MultiIndex):
                if all([n.lower() in data.index.names for n in ['x', 'y', 'z', 'dx', 'dy', 'dz']]):
                    self.is_irregular = True
                elif all([n.lower() in data.index.names for n in ['x', 'y', 'z']]):
                    self.is_irregular = False
                data.index.set_names([n.lower() for n in data.index.names], inplace=True)

            else:
                raise ValueError("The index must be a pd.MultiIndex with names ['x', 'y', 'z'] "
                                 "or [['x', 'y', 'z', 'dx', 'dy', 'dz'].")

            # sort the data to ensure consistent with pyvista
            data.sort_index(level=['z', 'y', 'x'], ascending=[True, True, True], inplace=True)

        super().__init__(data=data, name=name, moisture_in_scope=moisture_in_scope,
                         mass_wet_var=mass_wet_var, mass_dry_var=mass_dry_var,
                         moisture_var=moisture_var, component_vars=component_vars,
                         composition_units=composition_units, components_as_symbols=components_as_symbols,
                         ranges=ranges, config_file=config_file)

    @classmethod
    @import_extras
    def from_omf(cls, omf_filepath: Path, imports,
                 element_name: Optional[str] = None,
                 columns: Optional[list[str]] = None,
                 query: Optional[str] = None,
                 density: float = 2.5) -> 'BlockModel':
        """Create a BlockModel from an OMF file.

        Args:
            omf_filepath: Path to the OMF file.
            imports: internally used to import the necessary packages.
            element_name: The name of the element in the OMF file.
            columns: The columns to extract from the OMF file.
            query: The query to filter the DataFrame.
            density: The density of the material in g/cm3, used to calculate DMT (Dry Mass Tonnes).  A workaround.

        Returns:
            BlockModel: The BlockModel instance.

        """

        omfpr: imports.omfpandas.OMFPandasReader = imports.omfpandas.OMFPandasReader(filepath=omf_filepath)
        blocks: pd.DataFrame = omfpr.read_blockmodel(blockmodel_name=element_name, attributes=columns,
                                                     query=query)

        # get the block volume

        volume: Union[float, np.ndarray[float]]
        from omfpandas.blockmodel import OMFBlockModel
        from omfpandas.blockmodels.convert_blockmodel import df_to_blockmodel
        from omfpandas.blockmodels.geometry import Geometry
        geom: Geometry = OMFBlockModel(df_to_blockmodel(blocks, blockmodel_name=element_name)).geometry

        if geom.__class__.__name__ == 'RegularGeometry':
            volume = geom.block_size[0] * geom.block_size[1] * geom.block_size[2]
        elif geom.__class__.__name__ == 'TensorGeometry':
            # TODO: Implement the volume calculation for TensorGeometry - this is a placeholder.
            volume = geom.block_sizes[0][0] * geom.block_sizes[0][1] * geom.block_sizes[0][2]
        else:
            raise ValueError(f"Geometry type '{geom.__class__.__name__}' not supported.")

        if density is not None:
            blocks['mass_dry'] = volume * density
            moisture_in_scope = False

        return cls(data=blocks, name=element_name, mass_dry_var='mass_dry', moisture_in_scope=moisture_in_scope)

    @import_extras
    def to_omf(self, omf_filepath: Path, imports, element_name: str = 'Block Model',
               description: str = 'A block model'):
        """Write the BlockModel to an OMF file.

        Args:
            omf_filepath: Path to the OMF file.
            imports: internally used to import the necessary packages.
            element_name: The name of the element in the OMF file.
            description: Description of the block model.
        """
        # Create an OMFPandasWriter instance
        writer = imports.omfpandas.OMFPandasWriter(filepath=omf_filepath)

        # Write the block model to the OMF file
        writer.write_blockmodel(blockmodel_name=element_name, description=description, dataframe=self.data)

    @log_timer
    def get_blocks(self) -> Union['pv.StructuredGrid', 'pv.UnstructuredGrid']:

        try:
            # Attempt to create a regular grid
            grid = self.create_structured_grid()
            self._logger.debug("Created a pv.StructuredGrid.")
        except ValueError:
            # If it fails, create an irregular grid
            grid = self.create_unstructured_grid()
            self._logger.debug("Created a pv.UnstructuredGrid.")
        return grid

    @import_extras
    def plot(self, scalar: str, imports, show_edges: bool = True) -> 'pv.Plotter':

        if scalar not in self.data_columns:
            raise ValueError(f"Column '{scalar}' not found in the DataFrame.")

        # Create a PyVista plotter
        plotter = imports.pv.Plotter()

        mesh = self.get_blocks()

        # Add a thresholded mesh to the plotter
        plotter.add_mesh_threshold(mesh, scalars=scalar, show_edges=show_edges)

        return plotter

    def is_regular(self) -> bool:
        """
        Determine if the grid spacing is complete and regular
        If it is, a pv.StructuredGrid is suitable.
        If not, a pv.UnstructuredGrid is suitable.

        :return:
        """

        block_sizes = np.array(self._block_sizes())
        return np.all(np.isclose(np.mean(block_sizes, axis=1), 0))

    def _block_sizes(self):
        data = self.data
        x_unique = data.index.get_level_values('x').unique()
        y_unique = data.index.get_level_values('y').unique()
        z_unique = data.index.get_level_values('z').unique()

        x_spacing = np.diff(x_unique)
        y_spacing = np.diff(y_unique)
        z_spacing = np.diff(z_unique)

        return x_spacing, y_spacing, z_spacing

    def common_block_size(self):
        data = self.data
        x_unique = data.index.get_level_values('x').unique()
        y_unique = data.index.get_level_values('y').unique()
        z_unique = data.index.get_level_values('z').unique()

        x_spacing = np.abs(np.diff(x_unique))
        y_spacing = np.abs(np.diff(y_unique))
        z_spacing = np.abs(np.diff(z_unique))

        return stats.mode(x_spacing).mode, stats.mode(y_spacing).mode, stats.mode(z_spacing).mode

    @import_extras
    def create_structured_grid(self, imports) -> 'pv.StructuredGrid':

        # Get the unique x, y, z coordinates (centroids)
        data = self.data.sort_values(['z', 'y', 'x'])  # ensure the data is sorted F-style
        x_centroids = data.index.get_level_values('x').unique()
        y_centroids = data.index.get_level_values('y').unique()
        z_centroids = data.index.get_level_values('z').unique()

        # Calculate the cell size (assuming all cells are of equal size)
        dx = np.diff(x_centroids)[0]
        dy = np.diff(y_centroids)[0]
        dz = np.diff(z_centroids)[0]

        # Calculate the grid points
        x_points = np.concatenate([x_centroids - dx / 2, x_centroids[-1:] + dx / 2])
        y_points = np.concatenate([y_centroids - dy / 2, y_centroids[-1:] + dy / 2])
        z_points = np.concatenate([z_centroids - dz / 2, z_centroids[-1:] + dz / 2])

        # Create the 3D grid of points
        x, y, z = np.meshgrid(x_points, y_points, z_points, indexing='ij')

        # Create a StructuredGrid object
        grid = imports.pv.StructuredGrid(x, y, z)

        # Add the data from the DataFrame to the grid
        for column in data.columns:
            grid.cell_data[column] = data[column].values

        return grid

    def create_voxels(self) -> 'pv.UnstructuredGrid':
        grid = self.voxelise(self.data)
        return grid

    @import_extras
    def create_unstructured_grid(self, imports) -> 'pv.UnstructuredGrid':
        """
        Requires the index to be a pd.MultiIndex with names ['x', 'y', 'z', 'dx', 'dy', 'dz'].
        :return:
        """

        # Get the x, y, z coordinates and cell dimensions
        blocks = self.data.reset_index().sort_values(['z', 'y', 'x'])  # ensure the data is sorted F-style
        # if no dims are passed, estimate them
        if 'dx' not in blocks.columns:
            dx, dy, dz = self.common_block_size()
            blocks['dx'] = dx
            blocks['dy'] = dy
            blocks['dz'] = dz

        x, y, z, dx, dy, dz = (blocks[col].values for col in blocks.columns if col in ['x', 'y', 'z', 'dx', 'dy', 'dz'])
        blocks.set_index(['x', 'y', 'z', 'dx', 'dy', 'dz'], inplace=True)
        # Create the cell points/vertices
        # REF: https://github.com/OpenGeoVis/PVGeo/blob/main/PVGeo/filters/voxelize.py

        n_cells = len(x)

        # Generate cell nodes for all points in data set
        # - Bottom
        c_n1 = np.stack(((x - dx / 2), (y - dy / 2), (z - dz / 2)), axis=1)
        c_n2 = np.stack(((x + dx / 2), (y - dy / 2), (z - dz / 2)), axis=1)
        c_n3 = np.stack(((x - dx / 2), (y + dy / 2), (z - dz / 2)), axis=1)
        c_n4 = np.stack(((x + dx / 2), (y + dy / 2), (z - dz / 2)), axis=1)
        # - Top
        c_n5 = np.stack(((x - dx / 2), (y - dy / 2), (z + dz / 2)), axis=1)
        c_n6 = np.stack(((x + dx / 2), (y - dy / 2), (z + dz / 2)), axis=1)
        c_n7 = np.stack(((x - dx / 2), (y + dy / 2), (z + dz / 2)), axis=1)
        c_n8 = np.stack(((x + dx / 2), (y + dy / 2), (z + dz / 2)), axis=1)

        # - Concatenate
        # nodes = np.concatenate((c_n1, c_n2, c_n3, c_n4, c_n5, c_n6, c_n7, c_n8), axis=0)
        nodes = np.hstack((c_n1, c_n2, c_n3, c_n4, c_n5, c_n6, c_n7, c_n8)).ravel().reshape(n_cells * 8, 3)

        # create the cells
        # REF: https://docs/pyvista.org/examples/00-load/create-unstructured-surface.html
        cells_hex = np.arange(n_cells * 8).reshape(n_cells, 8)

        grid = imports.pv.UnstructuredGrid({imports.pv.CellType.VOXEL: cells_hex}, nodes)

        # add the attributes (column) data
        for col in blocks.columns:
            grid.cell_data[col] = blocks[col].values

        return grid

    @staticmethod
    @log_timer
    @import_extras
    def voxelise(blocks, imports) -> 'pv.UnstructuredGrid':

        logger = logging.getLogger(__name__)
        msg = "Voxelising blocks requires PVGeo package."
        logger.error(msg)
        raise NotImplementedError(msg)

        # vtkpoints = PVGeo.points_to_poly_data(centroid_data)

        x_values = blocks.index.get_level_values('x').values
        y_values = blocks.index.get_level_values('y').values
        z_values = blocks.index.get_level_values('z').values

        # Stack x, y, z values into a numpy array
        centroids = np.column_stack((x_values, y_values, z_values))

        # Create a PolyData object
        polydata = imports.pv.PolyData(centroids)

        # Add cell values as point data
        for column in blocks.columns:
            polydata[column] = blocks[[column]]

        # Create a Voxelizer filter
        voxelizer = PVGeo.filters.VoxelizePoints()
        # Apply the filter to the points
        grid = voxelizer.apply(polydata)

        logger.info(f"Voxelised {blocks.shape[0]} points.")
        logger.info("Recovered Angle (deg.): %.3f" % voxelizer.get_angle())
        logger.info("Recovered Cell Sizes: (%.2f, %.2f, %.2f)" % voxelizer.get_spacing())

        return grid

    def __str__(self):
        return f"BlockModel: {self.name}\n{self.aggregate.to_dict()}"
