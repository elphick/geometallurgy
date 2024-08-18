from typing import List
import pandas as pd
from itertools import tee


def amenability_index(df_recovery: pd.DataFrame,
                      col_target: str,
                      col_mass_recovery: str):
    """Calculate the Amenability Index

    Implementation of the Amenability Index as presented in the paper titled APPLICATIONS OF INDIVIDUAL PARTICLE
    PYKNOMETRY by G. Elphick and Dr. T.F. Mason at the DMS POWDERS’ 10th FERROSILICON CONFERENCE.

    The amenability Index for a particular gangue analyte is the complement of the relative recovery to the target
     analyte across the full sample.  It is process / operating point independent, hence characterises the ore,
     not the process.

    Args:
        df_recovery: DataFrame containing the ideal incremental recovery of a fractionated sample.
        col_target: The column name of the target analyte
        col_mass_recovery: The column name of the mass_recovery (yield) column

    Returns:
        A pd.Series containing the Amenability Indices for the gangue analytes
    """
    cols: List[str] = [col for col in df_recovery.columns if col not in [col_mass_recovery]]
    area_target = area_trapezoid(xs=df_recovery[col_mass_recovery], ys=df_recovery[col_target])
    results: List = []
    for analyte in cols:
        area = area_trapezoid(xs=df_recovery[col_mass_recovery], ys=df_recovery[analyte])
        results.append(1 - (area / area_target))
    return pd.Series(results, index=cols, name='amenability_index')


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ...
    For use in py39, after which itertools.pairwise can be used
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def area_trapezoid(xs, ys):
    area = 0
    for (ax, ay), (bx, by) in pairwise(zip(xs, ys)):
        h = bx - ax
        area += h * (ay + by) / 2
    return area
