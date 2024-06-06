import pandas as pd

from elphick.geomet.sample import Sample


class IntervalSample(Sample):
    """
    A class to represent a sample of data with an interval index.
    This exposes methods to split the sample by a partition definition.
    """

    def __init__(self, data: pd.DataFrame, name: str):
        super().__init__(data, name)
        self._data = data
        self._name = name

    def split_by_partition(self, partition_definition, name_1: str, name_2: str):
        """
        Split the sample into two samples based on the partition definition.
        :param partition_definition: A function that takes a data frame and returns a boolean series.
        :param name_1: The name of the first sample.
        :param name_2: The name of the second sample.
        :return: A tuple of two IntervalSamples.
        """
        raise NotImplementedError('Not yet ready...')
        mask = partition_definition(self._data)
        sample_1 = self._data[mask]
        sample_2 = self._data[~mask]
        return IntervalSample(sample_1, name_1), IntervalSample(sample_2, name_2)
