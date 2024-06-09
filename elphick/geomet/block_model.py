import copy
import logging
from pathlib import Path
from typing import Optional, Union, Literal

import numpy as np
import pandas as pd
import pyvista as pv
from pyvista import CellType
from scipy import stats

from elphick.geomet import MassComposition
from elphick.geomet.utils.timer import log_timer


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

        if isinstance(data.index, pd.MultiIndex):
            if all([n.lower() in data.index.names for n in ['x', 'y', 'z', 'dx', 'dy', 'dz']]):
                self.is_irregular = True
            elif all([n.lower() in data.index.names for n in ['x', 'y', 'z']]):
                self.is_irregular = False
            data.index.set_names([n.lower() for n in data.index.names], inplace=True)

        else:
            raise ValueError("The index must be a pd.MultiIndex with names ['x', 'y', 'z'] "
                             "or [['x', 'y', 'z', 'dx', 'dy', 'dz'].")

        super().__init__(data=data, name=name, moisture_in_scope=moisture_in_scope,
                         mass_wet_var=mass_wet_var, mass_dry_var=mass_dry_var,
                         moisture_var=moisture_var, component_vars=component_vars,
                         composition_units=composition_units, components_as_symbols=components_as_symbols,
                         ranges=ranges, config_file=config_file)

    @log_timer
    def get_blocks(self) -> Union[pv.StructuredGrid, pv.UnstructuredGrid]:
        try:
            # Attempt to create a regular grid
            grid = self.create_structured_grid()
            self._logger.debug("Created a pv.StructuredGrid.")
        except ValueError:
            # If it fails, create an irregular grid
            grid = self.create_unstructured_grid()
            self._logger.debug("Created a pv.UnstructuredGrid.")
        return grid

    @log_timer
    def plot(self, scalar: str, show_edges: bool = True) -> pv.Plotter:

        # Create a PyVista plotter
        plotter = pv.Plotter()

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

    def create_structured_grid(self) -> pv.StructuredGrid:
        # Get the unique x, y, z coordinates (centroids)
        data = self.data
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
        grid = pv.StructuredGrid(x, y, z)

        # Add the data from the DataFrame to the grid
        for column in data.columns:
            grid.cell_data[column] = data[column].values

        return grid

    def create_voxels(self) -> pv.UnstructuredGrid:
        grid = self.voxelise(self.data)
        return grid

    @log_timer
    def create_unstructured_grid(self) -> pv.UnstructuredGrid:
        """
        Requires the index to be a pd.MultiIndex with names ['x', 'y', 'z', 'dx', 'dy', 'dz'].
        :return:
        """
        # Get the x, y, z coordinates and cell dimensions
        blocks = self.data.reset_index().sort_values(['z', 'y', 'x'])
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

        grid = pv.UnstructuredGrid({CellType.VOXEL: cells_hex}, nodes)

        # add the attributes (column) data
        for col in blocks.columns:
            grid.cell_data[col] = blocks[col].values

        return grid

    @staticmethod
    @log_timer
    def voxelise(blocks):

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
        polydata = pv.PolyData(centroids)

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
