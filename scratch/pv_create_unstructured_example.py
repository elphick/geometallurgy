"""
This example from : https://docs.pyvista.org/version/stable/examples/00-load/create-unstructured-surface#

"""

import numpy as np

import pyvista as pv
from pyvista import CellType

# %%

# Contains information on the points composing each cell.
# Each cell begins with the number of points in the cell and then the points
# composing the cell
cells = np.array([8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 10, 11, 12, 13, 14, 15])

# cell type array. Contains the cell type of each cell
cell_type = np.array([CellType.HEXAHEDRON, CellType.HEXAHEDRON])

# in this example, each cell uses separate points
cell1 = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ]
)

cell2 = np.array(
    [
        [0, 0, 2],
        [1, 0, 2],
        [1, 1, 2],
        [0, 1, 2],
        [0, 0, 3],
        [1, 0, 3],
        [1, 1, 3],
        [0, 1, 3],
    ]
)

# points of the cell array
points = np.vstack((cell1, cell2)).astype(float)

# create the unstructured grid directly from the numpy arrays
grid = pv.UnstructuredGrid(cells, cell_type, points)

# For cells of fixed sizes (like the mentioned Hexahedra), it is also possible to use the
# simplified dictionary interface. This automatically calculates the cell array.
# Note that for mixing with additional cell types, just the appropriate key needs to be
# added to the dictionary.
cells_hex = np.arange(16).reshape([2, 8])
# = np.array([[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]])
grid = pv.UnstructuredGrid({CellType.HEXAHEDRON: cells_hex}, points)

# plot the grid (and suppress the camera position output)
_ = grid.plot(show_edges=True)

# %%


# these points will all be shared between the cells
points = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.5, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.5, 0.5, 0.0],
        [1.0, 0.0, 0.5],
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 0.5],
        [0.0, 0.0, 1.0],
        [0.5, 0.0, 0.5],
        [0.5, 0.0, 1.0],
        [1.0, 1.0, 0.5],
        [1.0, 1.0, 1.0],
        [1.0, 0.5, 0.5],
        [1.0, 0.5, 1.0],
        [0.0, 1.0, 0.5],
        [0.0, 1.0, 1.0],
        [0.5, 1.0, 0.5],
        [0.5, 1.0, 1.0],
        [0.0, 0.5, 0.5],
        [0.0, 0.5, 1.0],
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 1.0],
    ]
)


# Each cell in the cell array needs to include the size of the cell
# and the points belonging to the cell.  In this example, there are 8
# hexahedral cells that have common points between them.
cells = np.array(
    [
        [8, 0, 2, 8, 7, 11, 13, 25, 23],
        [8, 2, 1, 4, 8, 13, 9, 17, 25],
        [8, 7, 8, 6, 5, 23, 25, 21, 19],
        [8, 8, 4, 3, 6, 25, 17, 15, 21],
        [8, 11, 13, 25, 23, 12, 14, 26, 24],
        [8, 13, 9, 17, 25, 14, 10, 18, 26],
        [8, 23, 25, 21, 19, 24, 26, 22, 20],
        [8, 25, 17, 15, 21, 26, 18, 16, 22],
    ]
).ravel()

print(f"cells shape: {cells.shape}")
print(f"cells type: {type(cells)}")
print(f"cells dtype: {cells.dtype}")

# each cell is a HEXAHEDRON
celltypes = np.full(8, CellType.HEXAHEDRON, dtype=np.uint8)

print(f"cell type shape: {celltypes.shape}")
print(f"cell type type: {type(celltypes)}")
print(f"cell type dtype: {celltypes.dtype}")

# plot
grid = pv.UnstructuredGrid(cells, celltypes, points)

# Alternate versions:
grid = pv.UnstructuredGrid({CellType.HEXAHEDRON: cells.reshape([-1, 9])[:, 1:]}, points)
grid = pv.UnstructuredGrid(
    {CellType.HEXAHEDRON: np.delete(cells, np.arange(0, cells.size, 9))}, points
)

# plot the grid (and suppress the camera position output)
_ = grid.plot(show_edges=True)