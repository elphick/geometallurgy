import webbrowser
from pathlib import Path
from typing import Dict

import pandas as pd
import platformdirs
import pooch
from pooch import Unzip, Pooch


class Downloader:
    def __init__(self):
        """Instantiate a Downloader
        """

        self.register: pd.DataFrame = pd.read_csv(Path(__file__).parent / 'register.csv', index_col=False)

        self.dataset_hashes: Dict = self.register[['target', 'target_sha256']].set_index('target').to_dict()[
            'target_sha256']

        self.downloader: Pooch = pooch.create(path=Path(platformdirs.user_cache_dir('mass_composition', 'elphick')),
                                              base_url="https://github.com/elphick/mass-composition/raw/main/docs"
                                                       "/source/_static/",
                                              version=None,
                                              version_dev=None,
                                              registry={**self.dataset_hashes})

    def load_data(self, datafile: str = 'size_by_assay.zip', show_report: bool = False) -> pd.DataFrame:
        """
        Load the 231575341_size_by_assay data as a pandas.DataFrame.
        """
        if datafile not in self.dataset_hashes.keys():
            raise KeyError(f"The file {datafile} is not in the registry containing: {self.dataset_hashes.keys()}")

        fnames = self.downloader.fetch(datafile, processor=Unzip())
        if show_report:
            webbrowser.open(str(Path(fnames[0]).with_suffix('.html')))
        data = pd.read_csv(Path(fnames[0]).with_suffix('.csv'))
        return data

