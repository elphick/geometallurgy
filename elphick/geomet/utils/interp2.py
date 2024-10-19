from typing import Optional, Iterable, Union

import numpy as np
import pandas as pd
from scipy.interpolate import pchip_interpolate, PchipInterpolator

from elphick.geomet.utils.pandas import composition_to_mass, mass_to_composition, weight_average


def mass_preserving_interp_2d(specific_mass_intervals: pd.DataFrame,
                              interval_edges: Optional[dict[str, Iterable[float]]] = None,
                              precision: Optional[int] = None,
                              mass_dry: str = 'mass_dry') -> pd.DataFrame:
    """Interpolate 2D interval data with zero mass loss using pchip

    This function applies mass-preserving up-sampling to 2D interval data.  The function will:
    For example if the first dimension is size and the second is density:

    1. Convert to specific mass
    2. Accumulate along both dimensions
    3. For every first (size) dimension value
        a. fit a monotonic spline for the values of the second (density) dimension
        b. interpolate the second (density) dimension to the supplied (density) grid
    4. For every value on the second (density) upsampled (supplied) grid:
        a. fit a monotonic spline to the values of the first (size) dimension
        b. interpolate the first (size) dimension to the supplied (size) grid
    5. Decumulate along both dimensions
    6. Convert to composition

    The above steps are repeated for every component, starting with dry mass and ending with the last component.


    Args:
        specific_mass_intervals: Dataframe with two pd.IntervalIndexes, representing specific mass intervals.
        interval_edges: Dict of the values of the new grid (interval edges) for each dimension, keyed by index
         name (dimension).  If None, the grid will be a rectilinear grid using unique values from the index.
        precision: Number of decimal places to round the index (edge) values.
        mass_dry: The dry mass column, not optional.  Consider solve_mass_moisture prior to this call if needed.

    Returns:

    """

    if precision is None:
        precision = 6

    dim_names = specific_mass_intervals.index.names
    original_dim1_vals: np.ndarray = np.unique(specific_mass_intervals.index.get_level_values(dim_names[0]).right)
    original_dim2_vals: np.ndarray = np.unique(specific_mass_intervals.index.get_level_values(dim_names[1]).right)

    specific_mass_intervals.sort_index(ascending=[True, True], inplace=True)

    if interval_edges is None:
        interval_edges: dict[str, list[float]] = {dim: specific_mass_intervals.index.get_level_values(dim).unique()
                                                  for dim in dim_names}

    # check that the supplied interval edges are bounded on the left by the sampled minimum
    for dim in dim_names:
        if np.min(interval_edges[dim]) < specific_mass_intervals.index.get_level_values(dim).left.min():
            raise ValueError(f"The supplied {dim} grid contains values lower than the minimum in the sample.")


    # Convert to a numpy 3d array
    # [i, j, k] where i is the dim1, j is the dim2, and k is the component

    # We will accumulate along both dimensions, but notably we cannot assume that the 2d data sits on a
    # rectilinear grid.

    # TODO: test this code block in the case of non-rectilinear input data.
    mass_data: np.ndarray = specific_mass_intervals.to_numpy().reshape(
        specific_mass_intervals.index.get_level_values(dim_names[0]).unique().size,
        specific_mass_intervals.index.get_level_values(dim_names[1]).unique().size, -1)

    mass_data_cum: np.ndarray = np.cumsum(np.cumsum(mass_data, axis=0), axis=1)

    # create a 3D array to store the upsampled dim2 data (for each original dim1 value)
    upsampled_dim2: np.ndarray = np.zeros(
        (len(original_dim1_vals), len(interval_edges[dim_names[1]]), mass_data_cum.shape[2]))

    # create a 3D array to store the full upsampled (dim1 and dim2) data
    mass_data_upsampled = np.zeros(
        (len(interval_edges[dim_names[0]]), len(interval_edges[dim_names[1]]), mass_data_cum.shape[2]))

    for k in range(mass_data_cum.shape[2]):  # for every component (c)

        for i in range(mass_data_cum.shape[0]):  # for every dim1 (i), upsample dim2 (j)
            x = original_dim2_vals  # dim2
            y = mass_data_cum[i, :, k]  # component

            dim2_spline = PchipInterpolator(x, y, extrapolate=True)
            # now we can upsample the spline to a finer grid
            y_new = dim2_spline(interval_edges[dim_names[1]])  # upsampled dim2
            # Store the upsampled data
            upsampled_dim2[i, :, k] = y_new

        # for every upsampled dim2 value (j), upsample dim1 (i)
        for j in range(len(interval_edges[dim_names[1]])):
            x = original_dim1_vals  # dim1
            y = upsampled_dim2[:, j, k]  # component

            dim1_spline = PchipInterpolator(x, y, extrapolate=True)
            # now we can upsample the spline to a finer grid
            y_new = dim1_spline(interval_edges[dim_names[0]])  # upsampled size
            # Store the upsampled data
            mass_data_upsampled[:, j, k] = y_new

    print(mass_data_upsampled.shape)
    # Now we can decumulate along both dimensions
    mass_data_decum: np.ndarray = np.diff(
        np.diff(mass_data_upsampled, axis=0,
                prepend=np.zeros((1, mass_data_upsampled.shape[1], mass_data_upsampled.shape[2]))),
        axis=1, prepend=np.zeros((mass_data_upsampled.shape[0], 1, mass_data_upsampled.shape[2])))

    # Reshape to a 2D array
    mass_data_decum = mass_data_decum.reshape(mass_data_upsampled.shape[0] * mass_data_upsampled.shape[1],
                                              mass_data_upsampled.shape[2])

    # Create the multiindex of IntervalIndex for the upsampled data, using the supplied interval edges

    # Generate the MultiIndex
    min_left_edges = np.array([i.left for i in specific_mass_intervals.index[0]])
    dim1_indexes = pd.IntervalIndex.from_breaks(
        np.round(np.hstack((min_left_edges[0], interval_edges[dim_names[0]])), precision),
        closed='left')
    dim2_indexes = pd.IntervalIndex.from_breaks(
        np.round(np.hstack((min_left_edges[1], interval_edges[dim_names[1]])), precision),
        closed='left')

    index = pd.MultiIndex.from_product([dim1_indexes, dim2_indexes], names=dim_names)

    # TODO: account for uniform density for some sizes, like cyclosize.
    # This can be by 3 splines, one for mid, and the two boundaries, by extrapolation.

    return pd.DataFrame(mass_data_decum, index=index, columns=specific_mass_intervals.columns)
