from typing import Optional

import numpy as np
import pandas as pd


class Operation:
    def __init__(self, name):
        self.name = name
        self._input_streams = []
        self._output_streams = []
        self._is_balanced = None
        self._unbalanced_records = None

    @property
    def input_streams(self):
        return self._input_streams

    @input_streams.setter
    def input_streams(self, streams):
        self._input_streams = streams
        self._is_balanced = None  # Reset balance status

    @property
    def output_streams(self):
        return self._output_streams

    @output_streams.setter
    def output_streams(self, streams):
        self._output_streams = streams
        self._is_balanced = None  # Reset balance status

    def is_balanced(self) -> Optional[bool]:
        """Checks if the mass and chemistry of the input and output streams are balanced"""
        if not self.input_streams or not self.output_streams:
            return None

        # Update the total mass of the input and output streams
        total_input_mass: pd.Series = pd.concat([stream._mass_data for stream in self.input_streams]).sum()
        total_output_mass: pd.Series = pd.concat([stream._mass_data for stream in self.output_streams]).sum()

        self._mass_diff = total_input_mass - total_output_mass
        self._is_balanced = np.all(np.isclose(total_input_mass, total_output_mass))
        self._unbalanced_records = np.where(~np.isclose(total_input_mass, total_output_mass))[0]

        return self._is_balanced

    def get_failed_records(self):
        """Returns the dataframe of the records that failed the balance check"""
        if self._is_balanced is None:
            self.is_balanced()
        unbalanced_records = pd.Index(self._unbalanced_records)
        failed_records = self._mass_diff[self._mass_diff.index.isin(unbalanced_records)]
        return failed_records.to_frame(name='mass_difference')


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
