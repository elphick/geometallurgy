from copy import copy
from enum import Enum
from functools import reduce
from typing import Optional, TypeVar, TYPE_CHECKING

import numpy as np
import pandas as pd

from elphick.geomet.base import MC
from elphick.geomet.flowsheet.stream import Stream
from elphick.geomet.utils.pandas import MeanIntervalIndex
from elphick.geomet.utils.partition import load_partition_function

# generic type variable, used for type hinting that play nicely with subclasses
OP = TypeVar('OP', bound='Operation')

if TYPE_CHECKING:
    from elphick.geomet import IntervalSample


class NodeType(Enum):
    SOURCE = 'input'
    SINK = 'output'
    BALANCE = 'degree 2+'


class Operation:
    def __init__(self, name):
        self.name = name
        self._inputs = []
        self._outputs = []
        self._is_balanced: Optional[bool] = None
        self._unbalanced_records: Optional[pd.DataFrame] = None

    @property
    def has_empty_input(self) -> bool:
        return None in self.inputs

    @property
    def has_empty_output(self) -> bool:
        return None in self.outputs

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, value: list[MC]):
        self._inputs = value
        self.check_balance()

    @property
    def outputs(self):
        return self._outputs

    @outputs.setter
    def outputs(self, value: list[MC]):
        self._outputs = value
        self.check_balance()

    @property
    def node_type(self) -> Optional[NodeType]:
        if self.inputs and not self.outputs:
            res = NodeType.SINK
        elif self.outputs and not self.inputs:
            res = NodeType.SOURCE
        elif self.inputs and self.outputs:
            res = NodeType.BALANCE
        else:
            res = None
        return res

    def get_input_mass(self) -> pd.DataFrame:
        inputs = [i for i in self.inputs if i is not None]

        if not inputs:
            try:
                return self._create_zero_mass()
            except:
                return pd.DataFrame()
        elif len(inputs) == 1:
            return inputs[0].mass_data
        else:
            return reduce(lambda a, b: a.add(b, fill_value=0), [stream.mass_data for stream in inputs])

    def get_output_mass(self) -> pd.DataFrame:
        outputs = [o for o in self.outputs if o is not None]

        if not outputs:
            try:
                return self._create_zero_mass()
            except:
                return pd.DataFrame()
        elif len(outputs) == 1:
            return outputs[0].mass_data
        else:
            return reduce(lambda a, b: a.add(b, fill_value=0), [output.mass_data for output in outputs])

    def check_balance(self):
        """Checks if the mass and chemistry of the input and output are balanced"""
        if not self.inputs or not self.outputs:
            return None

        input_mass, output_mass = self.get_input_mass(), self.get_output_mass()
        is_balanced = np.all(np.isclose(input_mass.fillna(0.0), output_mass.fillna(0.0)))
        self._unbalanced_records = (input_mass - output_mass).loc[~np.isclose(input_mass, output_mass).any(axis=1)]
        self._is_balanced = is_balanced

    @property
    def is_balanced(self) -> Optional[bool]:
        return self._is_balanced

    @property
    def unbalanced_records(self) -> Optional[pd.DataFrame]:
        return self._unbalanced_records

    def solve(self) -> Optional[MC]:
        """Solves the operation

        Missing data is represented by None in the input and output streams.
        Solve will replace None with an object that balances the mass and chemistry of the input and output streams.
        Returns
        The back-calculated mc object
        """

        # Check the number of missing inputs and outputs
        missing_count: int = self.inputs.count(None) + self.outputs.count(None)
        if missing_count > 1:
            raise ValueError("The operation cannot be solved - too many degrees of freedom")
        mc = None
        if missing_count == 0 and self.is_balanced:
            return mc
        else:
            if None in self.inputs:
                ref_object = self.outputs[0]
                # Find the index of None in inputs
                none_index = self.inputs.index(None)

                # Calculate the None object
                new_input_mass: pd.DataFrame = self.get_output_mass() - self.get_input_mass()
                # Create a new object from the mass dataframe
                mc = type(ref_object).from_mass_dataframe(new_input_mass, mass_wet=ref_object.mass_wet_var,
                                                          mass_dry=ref_object.mass_dry_var,
                                                          moisture_column_name=ref_object.moisture_column,
                                                          component_columns=ref_object.composition_columns,
                                                          composition_units=ref_object.composition_units,
                                                          moisture_in_scope=ref_object.moisture_in_scope)
                # Replace None with the new input
                self.inputs[none_index] = mc

            elif None in self.outputs:
                ref_object = self.inputs[0]
                # Find the index of None in outputs
                none_index = self.outputs.index(None)

                # Calculate the None object
                if len(self.outputs) == 1 and len(self.inputs) == 1:
                    # passthrough, no need to calculate.  Shallow copy to minimise memory.
                    mc = copy(self.inputs[0])
                    mc.name = None
                else:
                    new_output_mass: pd.DataFrame = self.get_input_mass() - self.get_output_mass()
                    # Create a new object from the mass dataframe
                    mc = type(ref_object).from_mass_dataframe(new_output_mass, mass_wet=ref_object.mass_wet_var,
                                                              mass_dry=ref_object.mass_dry_var,
                                                              moisture_column_name=ref_object.moisture_column,
                                                              component_columns=ref_object.composition_columns,
                                                              composition_units=ref_object.composition_units,
                                                              moisture_in_scope=ref_object.moisture_in_scope)

                # Replace None with the new output
                self.outputs[none_index] = mc

            # update the balance related attributes
            self.check_balance()
            return mc

    def _create_zero_mass(self) -> pd.DataFrame:
        """Creates a zero mass dataframe with the same columns and index as the mass data"""
        # get the firstan object with the mass data
        obj = self._get_object()
        return pd.DataFrame(data=0, columns=obj.mass_data.columns, index=obj.mass_data.index)

    def _get_object(self, name: Optional[str] = None) -> MC:
        """Returns an object from inputs or outputs"""
        candidates = [mc for mc in self.outputs + self.inputs if mc is not None]
        if len(candidates) == 0:
            raise ValueError("No object found")
        if name:
            for obj in candidates:
                if obj is not None and obj.name == name:
                    return obj
            raise ValueError(f"No object found with name {name}")
        else:
            return candidates[0]

    @classmethod
    def from_dict(cls, config: dict) -> 'Operation':
        name = config.get('name')

        return cls(name=name)


class Input(Operation):
    def __init__(self, name):
        super().__init__(name)


class Output(Operation):
    def __init__(self, name):
        super().__init__(name)


class Passthrough(Operation):
    def __init__(self, name):
        super().__init__(name)


class PartitionOperation(Operation):
    """An operation that partitions the input stream into multiple output streams based on a partition function

    The partition input is the mean of the fractions or the geomean if the fractions are in the size dimension
    The partition function is typically a partial function so that the partition is defined for all arguments
    other than the input mean fraction values in one or two dimensions.  The argument names must match the
    index names in the IntervalSample.

    """

    def __init__(self, name, partition=None):
        super().__init__(name)
        self.partition = partition
        self.partition_function = None
        if self.partition and 'module' in self.partition and 'function' in self.partition:
            self.partition_function = load_partition_function(self.partition['module'], self.partition['function'])

    def solve(self) -> [MC, MC]:
        if self.partition_function:
            self.apply_partition()
            # update the balance related attributes
            self.check_balance()
        return self.outputs

    def apply_partition(self):
        if len(self.inputs) != 1:
            raise ValueError("PartitionOperation must have exactly one input")
        for input_sample in self.inputs:
            input_sample: 'IntervalSample'
            if input_sample is not None:
                output, complement = input_sample.split_by_partition(self.partition_function)
                self.outputs = [output, complement]

    @classmethod
    def from_dict(cls, config: dict) -> 'PartitionOperation':
        name = config.get('name')
        partition = config.get('partition')
        return cls(name=name, partition=partition)
