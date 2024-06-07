from functools import reduce
from typing import Optional

import numpy as np
import pandas as pd


class Operation:
    def __init__(self, name):
        self.name = name
        self._input_streams = []
        self._output_streams = []
        self._is_balanced: Optional[bool] = None
        self._unbalanced_records: Optional[pd.DataFrame] = None

    @property
    def input_streams(self):
        return self._input_streams

    @input_streams.setter
    def input_streams(self, streams):
        self._input_streams = streams
        self._is_balanced = self.check_balance()

    @property
    def output_streams(self):
        return self._output_streams

    @output_streams.setter
    def output_streams(self, streams):
        self._output_streams = streams
        self._is_balanced = self.check_balance()

    def check_balance(self) -> Optional[bool]:
        """Checks if the mass and chemistry of the input and output streams are balanced"""
        if not self.input_streams or not self.output_streams:
            return None

        # Calculate the mass of the inputs and outputs
        if len(self.input_streams) == 1:
            input_mass = self.input_streams[0]._mass_data
        else:
            input_mass = reduce(lambda a, b: a.add(b, fill_value=0),
                                [stream._mass_data for stream in self.input_streams])

        if len(self.output_streams) == 1:
            output_mass = self.output_streams[0]._mass_data
        else:
            output_mass = reduce(lambda a, b: a.add(b, fill_value=0),
                                 [stream._mass_data for stream in self.output_streams])

        is_balanced = np.all(np.isclose(input_mass, output_mass))
        self._unbalanced_records = (input_mass - output_mass).iloc[np.where(~np.isclose(input_mass, output_mass))[0]]

        return is_balanced

    @property
    def is_balanced(self) -> Optional[bool]:
        return self._is_balanced

    @property
    def unbalanced_records(self) -> Optional[pd.DataFrame]:
        return self._unbalanced_records


class InputOperation(Operation):
    def __init__(self, name):
        super().__init__(name)


class OutputOperation(Operation):
    def __init__(self, name):
        super().__init__(name)


class PassthroughOperation(Operation):
    def __init__(self, name):
        super().__init__(name)


class UnitOperation(Operation):
    def __init__(self, name, num_inputs, num_outputs):
        super().__init__(name)
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
