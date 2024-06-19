"""
To provide sample data
"""
import random
from functools import partial
from pathlib import Path
from typing import Optional, Iterable, List

import numpy as np
import pandas as pd

from elphick.geomet import Sample, IntervalSample
from elphick.geomet.flowsheet import Flowsheet
from elphick.geomet.utils.components import is_compositional
from elphick.geomet.datasets import load_size_by_assay, load_iron_ore_sample_a072391, load_size_distribution, \
    load_a072391_met
from elphick.geomet.utils.partition import napier_munn, perfect


def sample_data(include_wet_mass: bool = True, include_dry_mass: bool = True,
                include_moisture: bool = False) -> pd.DataFrame:
    """Creates synthetic data for testing

    Args:
        include_wet_mass: If True, wet mass is included.
        include_dry_mass: If True, dry mass is included.
        include_moisture: If True, moisture (H2O) is included.

    Returns:

    """

    # mass_wet: pd.Series = pd.Series([100, 90, 110], name='wet_mass')
    # mass_dry: pd.Series = pd.Series([90, 80, 100], name='dry_mass')
    mass_wet: pd.Series = pd.Series([100., 90., 110.], name='wet_mass')
    mass_dry: pd.Series = pd.Series([90., 80., 90.], name='mass_dry')
    chem: pd.DataFrame = pd.DataFrame.from_dict({'FE': [57., 59., 61.],
                                                 'SIO2': [5.2, 3.1, 2.2],
                                                 'al2o3': [3.0, 1.7, 0.9],
                                                 'LOI': [5.0, 4.0, 3.0]})
    attrs: pd.Series = pd.Series(['grp_1', 'grp_1', 'grp_2'], name='group')

    mass: pd.DataFrame = pd.concat([mass_wet, mass_dry], axis='columns')
    if include_wet_mass is True and mass_dry is False:
        mass = mass_wet
    elif include_dry_mass is False and mass_dry is True:
        mass = mass_dry
    elif include_dry_mass is False and mass_dry is False:
        raise AssertionError('Arguments provided result in no mass column')

    if include_moisture is True:
        moisture: pd.DataFrame = (mass_wet - mass_dry) / mass_wet * 100
        moisture.name = 'H2O'
        res: pd.DataFrame = pd.concat([mass, moisture, chem, attrs], axis='columns')
    else:
        res: pd.DataFrame = pd.concat([mass, chem, attrs], axis='columns')

    res.index.name = 'index'

    return res


def dh_intervals(n: int = 5,
                 n_dh: int = 2,
                 analytes: Optional[Iterable[str]] = ('Fe', 'Al2O3')) -> pd.DataFrame:
    """Down-samples The drillhole data for testing

    Args:
        n: Number of samples
        n_dh: The number of drill-holes included
        analytes: the analytes to include
    Returns:

    """

    df_data: pd.DataFrame = load_iron_ore_sample_a072391()
    # df_data: pd.DataFrame = pd.read_csv('../sample_data/iron_ore_sample_data.csv', index_col='index')

    drillholes: List[str] = []
    for i in range(0, n_dh):
        drillholes.append(random.choice(list(df_data['DHID'].unique())))

    df_data = df_data.query('DHID in @drillholes').groupby('DHID').sample(5)

    cols_to_drop = [col for col in is_compositional(df_data.columns) if (col not in analytes) and (col != 'H2O')]
    df_data.drop(columns=cols_to_drop, inplace=True)

    df_data.index.name = 'index'

    return df_data


def size_by_assay() -> pd.DataFrame:
    """ Sample Size x Assay dataset
    """

    df_data: pd.DataFrame = load_size_by_assay()

    # df_data: pd.DataFrame = pd.DataFrame(data=[size_retained, size_passing, mass_pct, fe, sio2, al2o3],
    #                                      index=['size_retained', 'size_passing', 'mass_pct', 'Fe', 'SiO2', 'Al2O3']).T

    # # convert the sizes from micron to mm
    # df_data[['size_retained', 'size_passing']] = df_data[['size_retained', 'size_passing']] / 1000.0

    df_data.set_index(['size_retained', 'size_passing'], inplace=True)

    # ensure we meet the input column name requirements
    df_data.rename(columns={'mass_pct': 'mass_dry'}, inplace=True)

    return df_data


def size_by_assay_2() -> pd.DataFrame:
    """ 3 x Sample Size x Assay dataset (balanced)
    """
    mc_size: IntervalSample = IntervalSample(size_by_assay(), name='feed', moisture_in_scope=False)
    partition = partial(napier_munn, d50=0.150, ep=0.1, dim='size')
    mc_coarse, mc_fine = mc_size.split_by_partition(partition_definition=partition, name_1='coarse', name_2='fine')
    fs: Flowsheet = Flowsheet().from_streams([mc_size, mc_coarse, mc_fine])
    return fs.to_dataframe()


def size_by_assay_3() -> pd.DataFrame:
    """ 3 x Sample Size x Assay dataset (unbalanced)
    """
    mc_size: IntervalSample = IntervalSample(size_by_assay(), name='feed')
    partition = partial(napier_munn, d50=0.150, ep=0.1, dim='size')
    mc_coarse, mc_fine = mc_size.split_by_partition(partition_definition=partition, name_1='coarse', name_2='fine')
    # add error to the coarse stream to create an imbalance
    df_coarse_2 = mc_coarse.data.to_dataframe().apply(lambda x: np.random.normal(loc=x, scale=np.std(x)))
    mc_coarse_2: Sample = Sample(data=df_coarse_2, name='coarse')
    mc_coarse_2 = mc_coarse_2.set_parent_node(mc_size)
    fs_ub: Flowsheet = Flowsheet().from_streams([mc_size, mc_coarse_2, mc_fine])
    return fs_ub.to_dataframe()


def size_distribution() -> pd.DataFrame:
    return load_size_distribution()


def iron_ore_sample_data() -> pd.DataFrame:
    return load_iron_ore_sample_a072391().set_index('index')


def iron_ore_met_sample_data() -> pd.DataFrame:
    df_met: pd.DataFrame = load_a072391_met()
    df_met.dropna(subset=['Dry Weight Lump (kg)'], inplace=True)
    df_met['Dry Weight Lump (kg)'] = df_met['Dry Weight Lump (kg)'].apply(lambda x: x.replace('..', '.')).astype(
        'float64')
    df_met['Fe'] = df_met['Fe'].replace('MISSING', np.nan).astype('float64')
    df_met.dropna(subset=['Fe', 'Bulk_Hole_No', 'Dry Weight Fines (kg)'], inplace=True)
    df_met.columns = [col.replace('LOITotal', 'LOI') for col in df_met.columns]
    df_met.columns = [
        col.strip().lower().replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct').replace('__', '_')
        for
        col in df_met.columns]

    # clean up some values and types
    df_met = df_met.replace('-', np.nan).replace('#VALUE!', np.nan)
    head_cols: List[str] = [col for col in df_met.columns if 'head' in col]
    df_met[head_cols] = df_met[head_cols].astype('float64')
    df_met['bulk_hole_no'] = df_met['bulk_hole_no'].astype('category')
    df_met['sample_number'] = df_met['sample_number'].astype('int64')
    df_met.set_index('sample_number', inplace=True)

    # moves suffixes to prefix
    df_met = df_met.pipe(_move_suffix_to_prefix, '_head')
    df_met = df_met.pipe(_move_suffix_to_prefix, '_lump')
    return df_met


def demo_size_network() -> Flowsheet:
    mc_size: Sample = Sample(size_by_assay(), name='size sample')
    partition = partial(perfect, d50=0.150, dim='size')
    mc_coarse, mc_fine = mc_size.split_by_partition(partition_definition=partition)
    mc_coarse.name = 'coarse'
    mc_fine.name = 'fine'
    fs: Flowsheet = Flowsheet().from_streams([mc_size, mc_coarse, mc_fine])
    return fs


def _move_suffix_to_prefix(df, suffix):
    suffix_length = len(suffix)
    for col in df.columns:
        if col.endswith(suffix):
            new_col = suffix[1:] + '_' + col[:-suffix_length]  # Remove the suffix and prepend it to the start
            df.rename(columns={col: new_col}, inplace=True)
    return df


if __name__ == '__main__':
    df1: pd.DataFrame = size_by_assay()
    df2: pd.DataFrame = size_by_assay_2()
    df3: pd.DataFrame = size_by_assay_3()
    df4: pd.DataFrame = iron_ore_met_sample_data()
    print('done')
