import numpy as np
from pandas.arrays import IntervalArray


def mean_size(size_intervals: IntervalArray) -> np.ndarray:
    """Geometric mean size

    Size calculations are performed using the geometric mean, not the arithmetic mean

    NOTE: If geometric mean is used for the pan fraction (0.0mm retained) it will return zero, which is an
    edge size not mean size.  So the mean ratio of the geometric mean to the arithmetic mean for all other
    fractions is used for the bottom fraction.


    Args:
        size_intervals: A pandas IntervalArray

    Returns:

    """

    intervals = size_intervals.copy()
    res = np.array((intervals.left * intervals.right) ** 0.5)

    geomean_mean_ratio: float = float(np.mean((res[0:-1] / intervals.mid[0:-1])))

    if np.isclose(size_intervals.min().left, 0.0):
        res[np.isclose(size_intervals.left, 0.0)] = size_intervals.min().mid * geomean_mean_ratio

    return res


# REF: https://www.globalgilson.com/blog/sieve-sizes

sizes_iso_565 = [63.0, 56.0, 53.0, 50.0, 45.0, 40.0, 37.5, 35.5, 31.5, 28.0, 26.5, 25.0, 22.4, 20.0,
                 19.0, 18.0, 16.0, 14.0, 13.2, 12.5, 11.2, 10.0, 9.5, 9.0, 8.0, 7.1, 6.7, 6.3, 5.6,
                 5.0, 4.75, 4.5, 4.0, 3.55, 3.35, 3.15, 2.8, 2.5, 2.36, 2.0, 1.8, 1.7, 1.6, 1.4, 1.25,
                 1.18, 1.12, 1.0, 0.900, 0.850, 0.800, 0.710, 0.630, 0.600, 0.560, 0.500, 0.450, 0.425,
                 0.400, 0.355, 0.315, 0.300, 0.280, 0.250, 0.224, 0.212, 0.200, 0.180, 0.160, 0.150, 0.140,
                 0.125, 0.112, 0.106, 0.100, 0.090, 0.080, 0.075, 0.071, 0.063, 0.056, 0.053, 0.050, 0.045,
                 0.040, 0.038, 0.036, 0.032, 0.025, 0.020]

sizes_astm_e11 = [100.0, 90.0, 75.0, 63.0, 53.0, 50.0, 45.0, 37.5, 31.5, 26.5, 25.0, 22.4, 19.0, 16.0,
                  13.2, 12.5, 11.2, 9.5, 8.0, 6.7, 6.3, 5.6, 4.75, 4.0, 3.35, 2.8, 2.36, 2.0, 1.7, 1.4,
                  1.18, 1.0, 0.850, 0.710, 0.600, 0.500, 0.425, 0.355, 0.300, 0.250, 0.212, 0.180, 0.150,
                  0.125, 0.106, 0.090, 0.075, 0.063, 0.053, 0.045, 0.038, 0.032, 0.025, 0.020]

sizes_all = sorted(list(set(sizes_astm_e11).union(set(sizes_iso_565))), reverse=True)
