from typing import Optional, Iterable, Union

import numpy as np
import pandas as pd
from scipy.interpolate import pchip_interpolate

from elphick.geomet.utils.pandas import composition_to_mass, mass_to_composition, weight_average


def mass_preserving_interp(df_intervals: pd.DataFrame, interval_edges: Union[Iterable, int],
                           include_original_edges: bool = True, precision: Optional[int] = None,
                           mass_wet: str = 'mass_wet', mass_dry: str = 'mass_dry') -> pd.DataFrame:
    """Interpolate with zero mass loss using pchip

    This interpolates data_vars independently for a single dimension (coord) at a time.

    The function will:
    - convert from relative composition (%) to absolute (mass)
    - convert the index from interval to a float representing the right edge of the interval
    - cumsum to provide monotonic increasing data
    - interpolate with a pchip spline to preserve mass
    - diff to recover the original fractional data
    - reconstruct the interval index from the right edges
    - convert from absolute to relative composition

    Args:
        df_intervals: A pd.DataFrame with a single interval index, with mass, composition context.
        interval_edges: The values of the new grid (interval edges).  If an int, will up-sample by that factor, for
         example the value of 10 will automatically define edges that create 10 x the resolution (up-sampled).
        include_original_edges: If True include the original index edges in the result
        precision: Number of decimal places to round the index (edge) values.
        mass_wet: The wet mass column, not optional.  Consider solve_mass_moisture prior to this call if needed.
        mass_dry: The dry mass column, not optional.  Consider solve_mass_moisture prior to this call if needed.

    Returns:

    """

    if not isinstance(df_intervals.index, pd.IntervalIndex):
        raise NotImplementedError(f"The index `{df_intervals.index}` of the dataframe is not a pd.Interval. "
                                  f" Only 1D interval indexes are valid")

    composition_in: pd.DataFrame = df_intervals.copy()

    if isinstance(interval_edges, int):
        grid_vals = _upsample_grid_by_factor(indx=composition_in.sort_index().index, factor=interval_edges)
    else:
        grid_vals = np.sort(np.array(interval_edges))

    if precision is not None:
        composition_in.index = pd.IntervalIndex.from_arrays(np.round(df_intervals.index.left, precision),
                                                            np.round(df_intervals.index.right, precision),
                                                            closed=df_intervals.index.closed,
                                                            name=df_intervals.index.name)

        grid_vals = np.round(grid_vals, precision)

    if include_original_edges:
        original_edges = np.hstack([df_intervals.index.left, df_intervals.index.right])
        grid_vals = np.sort(np.unique(np.hstack([grid_vals, original_edges])))

    if not isinstance(grid_vals, np.ndarray):
        grid_vals = np.array(grid_vals)

    # convert from relative composition (%) to absolute (mass)
    mass_in: pd.DataFrame = composition_to_mass(composition_in, mass_wet=mass_wet, mass_dry=mass_dry)
    # convert the index from interval to a float representing the right edge of the interval
    mass_in.index = mass_in.index.right
    # add a row of zeros
    mass_in = pd.concat(
        [mass_in, pd.Series(0, index=mass_in.columns, name=composition_in.index.left.min()).to_frame().T],
        axis=0).sort_index(ascending=True)
    # cumsum to provide monotonic increasing data
    mass_cum: pd.DataFrame = mass_in.cumsum()
    # if the new grid extrapolates (on the coarse side), mass will be lost, so we assume that when extrapolating.
    # the mass in the extrapolated fractions is zero.  By inserting these records the spline will conform.
    x_extra = grid_vals[grid_vals > mass_cum.index.max()]
    if x_extra:
        cum_max: pd.Series = mass_cum.iloc[-1, :]
        mass_cum = mass_cum.reindex(index=mass_cum.index.append(pd.Index(x_extra)))  # reindex to enable insert
        mass_cum.loc[x_extra, :] = cum_max.values
    #  interpolate with a pchip spline to preserve mass
    chunks = []
    for col in mass_cum:
        tmp = mass_cum[col].dropna()  # drop any missing values
        new_vals = pchip_interpolate(tmp.index.values, tmp.values, grid_vals)
        chunks.append(new_vals)
    mass_cum_upsampled: pd.DataFrame = pd.DataFrame(chunks, index=mass_in.columns, columns=grid_vals).T
    # diff to recover the original fractional data
    mass_fractions_upsampled: pd.DataFrame = mass_cum_upsampled.diff().dropna(axis=0)
    # reconstruct the interval index from the grid
    mass_fractions_upsampled.index = pd.IntervalIndex.from_arrays(left=grid_vals[:-1],
                                                                  right=grid_vals[1:],
                                                                  closed=df_intervals.index.closed,
                                                                  name=df_intervals.index.name)
    interval_spans: dict[pd.Interval, float] = {interval: interval.right - interval.left for interval in
                                                mass_fractions_upsampled.index}
    zero_spans = [k for k, v in interval_spans.items() if v == 0]
    if len(zero_spans) > 1:
        raise ValueError(f"The interpolated index contains zero width intervals on left edges: {zero_spans}")

    # convert from absolute to relative composition
    res = mass_to_composition(mass_fractions_upsampled, mass_wet=mass_wet, mass_dry=mass_dry).sort_index(
        ascending=False)

    # confirm the weight average is preserved
    if not np.isclose(mass_in.sum().sum(), mass_fractions_upsampled.sum().sum(), rtol=1e-6):
        raise ValueError("The mass is not preserved in the interpolation")
    return res


def _upsample_grid_by_factor(indx: pd.IntervalIndex, factor):
    # TODO: must be a better way than this - vectorised?
    grid_vals: list = [indx.left.min()]
    for interval in indx:
        increment = (interval.right - interval.left) / factor
        for i in range(0, factor):
            grid_vals.append(interval.left + (i + 1) * increment)
    grid_vals.sort()
    return grid_vals


def mass_preserving_interp_2d(intervals: pd.DataFrame, interval_edges: dict[str, Iterable],
                              include_original_edges: bool = True, precision: Optional[int] = None,
                              mass_dry: str = 'mass_dry') -> pd.DataFrame:
    """Interpolate 2D interval data with zero mass loss using pchip

    This function applies mass-preserving up-sampling to 2D interval data.  The function will:
    - resample the first dimension using the mass_preserving_interp function
    - resample the second dimension using the mass_preserving_interp function at each of the new first
        dimension intervals
    - apply the upsampled mass proportions of the second dimension to the first dimension to create the final result

    Args:
        intervals: Dataframe with two pd.IntervalIndexes, in mass-composition space.
        interval_edges: Dict of the values of the new grid (interval edges) for each dimension, keyed by
         index name (dimension).
        include_original_edges: If True include the original index edges in the result
        precision: Number of decimal places to round the index (edge) values.
        mass_dry: The dry mass column, not optional.  Consider solve_mass_moisture prior to this call if needed.

    Returns:

    """

    # reduce the dataframe to the first dimension
    dim_1_name, dim_2_name = intervals.index.names[0], intervals.index.names[1]
    dim_1: pd.DataFrame = intervals.groupby(dim_1_name).apply(weight_average, **{'mass_dry': mass_dry})

    # interpolate the first dimension
    dim_1_interp: pd.DataFrame = mass_preserving_interp(dim_1, interval_edges=interval_edges[dim_1_name],
                                                        include_original_edges=include_original_edges,
                                                        precision=precision, mass_dry=mass_dry)

    chunks: list = []
    # iterate the original dim_1 fractions
    for dim_1_interval in dim_1.index:
        # reduce the dataframe to the second dimension for the current dim1 interval
        dim_2: pd.DataFrame = intervals.loc[dim_1_interval].copy()
        # interpolate the second dimension
        dim_2_interp: pd.DataFrame = mass_preserving_interp(dim_2, interval_edges=interval_edges[dim_2_name],
                                                            include_original_edges=include_original_edges,
                                                            precision=precision, mass_dry=mass_dry)
        # convert to recovery to enable proportioning of the interpolated dim_1 values
        dim_2_mass: pd.DataFrame = composition_to_mass(dim_2_interp, mass_dry=mass_dry)
        dim_2_deportment: pd.DataFrame = dim_2_mass.div(dim_2_mass.sum(axis=0), axis=1)

        # Filter the intervals from dim_1_interp that fall within the provided dim_1_interval, convert to mass
        filtered_intervals_mass = composition_to_mass(
            dim_1_interp.loc[dim_1_interp.index.map(lambda x: x.overlaps(dim_1_interval))].copy(), mass_dry=mass_dry)

        # expand the dimensions of the interpolated dim_1 data to include the second upsampled dimension
        for dim_1_interp_interval in filtered_intervals_mass.index:
            # proportion the dim_1 interpolated values by the dim_2 recovery
            new_vals: pd.DataFrame = dim_2_deportment.mul(filtered_intervals_mass.loc[dim_1_interp_interval].values)
            # create a multiindex by combining the dim_1 and dim_2 intervals
            new_index = pd.MultiIndex.from_arrays([pd.IntervalIndex([dim_1_interp_interval] * len(new_vals)), new_vals.index])
            new_vals.index = new_index
            chunks.append(new_vals)

    # concatenate the results
    res: pd.DataFrame = pd.concat(chunks, axis=0)
    res.index.names = [dim_1_name, dim_2_name]
    # convert to composition
    res = mass_to_composition(res, mass_dry=mass_dry)
    return res

