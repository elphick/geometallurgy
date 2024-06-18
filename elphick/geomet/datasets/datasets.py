from elphick.geomet.datasets import Downloader
import pandas as pd


def load_a072391_assay(show_report: bool = False) -> pd.DataFrame:
    return Downloader().load_data(datafile='A072391_assay.zip', show_report=show_report)


def load_a072391_collars(show_report: bool = False) -> pd.DataFrame:
    return Downloader().load_data(datafile='A072391_collars.zip', show_report=show_report)


def load_a072391_geo(show_report: bool = False) -> pd.DataFrame:
    return Downloader().load_data(datafile='A072391_geo.zip', show_report=show_report)


def load_a072391_met(show_report: bool = False) -> pd.DataFrame:
    return Downloader().load_data(datafile='A072391_met.zip', show_report=show_report)


def load_a072391_wireline(show_report: bool = False) -> pd.DataFrame:
    return Downloader().load_data(datafile='A072391_wireline.zip', show_report=show_report)


def load_demo_data(show_report: bool = False) -> pd.DataFrame:
    return Downloader().load_data(datafile='demo_data.zip', show_report=show_report)


def load_iron_ore_sample_a072391(show_report: bool = False) -> pd.DataFrame:
    return Downloader().load_data(datafile='iron_ore_sample_A072391.zip', show_report=show_report)


def load_iron_ore_sample_xyz_a072391(show_report: bool = False) -> pd.DataFrame:
    return Downloader().load_data(datafile='iron_ore_sample_xyz_A072391.zip', show_report=show_report)


def load_nordic_iron_ore_sink_float(show_report: bool = False) -> pd.DataFrame:
    return Downloader().load_data(datafile='nordic_iron_ore_sink_float.zip', show_report=show_report)


def load_size_by_assay(show_report: bool = False) -> pd.DataFrame:
    return Downloader().load_data(datafile='size_by_assay.zip', show_report=show_report)


def load_size_distribution(show_report: bool = False) -> pd.DataFrame:
    return Downloader().load_data(datafile='size_distribution.zip', show_report=show_report)

