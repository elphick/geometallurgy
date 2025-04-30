import pandas as pd
import pytest
import yaml

from elphick.geomet import Sample, IntervalSample
from elphick.geomet.flowsheet.operation import Operation, PartitionOperation
from elphick.geomet.utils.data import sample_data
from elphick.geomet.utils.pandas import MeanIntervalIndex
from elphick.geomet.utils.partition import napier_munn_size_1mm


@pytest.fixture
def expected_data() -> pd.DataFrame:
    expected_data = sample_data(include_wet_mass=True,
                                include_dry_mass=True,
                                include_moisture=True)
    expected_data.columns = [col.lower() for col in expected_data.columns]
    expected_data.rename(columns={'wet_mass': 'mass_wet'}, inplace=True)
    return expected_data


@pytest.fixture
def sample_split() -> tuple[Sample, Sample, Sample]:
    data = sample_data()
    smpl = Sample(data=data, name='sample')
    ref, comp = smpl.split(fraction=0.5, include_supplementary_data=False)
    return ref, comp, smpl


@pytest.fixture
def sample_split_with_supp() -> tuple[Sample, Sample, Sample]:
    data = sample_data()
    smpl = Sample(data=data, name='sample')
    ref, comp = smpl.split(fraction=0.5, include_supplementary_data=True)
    return ref, comp, smpl


def test_operation_split(sample_split, expected_data):
    ref, comp, smpl = sample_split
    pd.testing.assert_frame_equal(ref.data, comp.data)

    op_node: Operation = Operation(name='split')
    op_node.inputs = [smpl]
    op_node.outputs = [ref, comp]
    assert op_node.is_balanced


def test_operation_add(sample_split_with_supp, expected_data):
    comp, ref, smpl = sample_split_with_supp
    smpl_new = ref.add(comp, name='sample_new', include_supplementary_data=True)
    pd.testing.assert_frame_equal(smpl.data, smpl_new.data)

    op_node: Operation = Operation(name='add')
    op_node.inputs = [smpl]
    op_node.outputs = [smpl_new]
    assert op_node.is_balanced


def test_operation_sub(sample_split_with_supp, expected_data):
    ref, comp, smpl = sample_split_with_supp
    ref_new = smpl.sub(comp, name='ref_new', include_supplementary_data=True)
    pd.testing.assert_frame_equal(ref.data, ref_new.data)

    op_node: Operation = Operation(name='add')
    op_node.inputs = [ref]
    op_node.outputs = [ref_new]
    assert op_node.is_balanced


def test_operation_imbalance_split(sample_split, expected_data):
    ref, comp, smpl = sample_split

    # introduce imbalance
    new_data: pd.DataFrame = comp.data.copy()
    new_data.loc[0, 'wet_mass'] = 1000
    comp.data = new_data

    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(ref.data, comp.data)

    op_node: Operation = Operation(name='split')
    op_node.inputs = [smpl]
    op_node.outputs = [ref, comp]
    with pytest.raises(AssertionError):
        assert op_node.is_balanced

    expected: pd.DataFrame = pd.DataFrame(
        {'wet_mass': {0: -950.0}, 'mass_dry': {0: 0.0}, 'Fe': {0: 0.0}, 'SiO2': {0: 0.0}, 'Al2O3': {0: 0.0},
         'LOI': {0: 0.0}}, index=op_node.unbalanced_records.index)
    pd.testing.assert_frame_equal(op_node.unbalanced_records, expected)


def test_operation_solve_simo(sample_split, expected_data):
    # SIMO: Single Input Multiple Output

    ref, comp, smpl = sample_split

    # create an operation
    op_node: Operation = Operation(name='split')

    # set an output stream to None
    op_node.inputs = [smpl]
    op_node.outputs = [ref, None]
    with pytest.raises(AssertionError):
        assert op_node.is_balanced

    # solve the operation to back-calculate an object materially equivalent to comp (name will be different)
    op_node.solve()

    assert op_node.is_balanced
    pd.testing.assert_frame_equal(comp.data, op_node.outputs[1].data)

    # set the input stream to None
    op_node.inputs = [None]
    op_node.outputs = [ref, comp]

    with pytest.raises(AssertionError):
        assert op_node.is_balanced

    op_node.solve()
    assert op_node.is_balanced


def test_operation_solve_miso(sample_split, expected_data):
    # MISO: Multiple Input Single Output

    ref, comp, smpl = sample_split

    # create an operation
    op_node: Operation = Operation(name='add')

    # set an input stream to None
    op_node.inputs = [ref, None]
    op_node.outputs = [smpl]
    with pytest.raises(AssertionError):
        assert op_node.is_balanced

    # solve the operation to back-calculate an object materially equivalent to comp (name will be different)
    op_node.solve()

    assert op_node.is_balanced
    pd.testing.assert_frame_equal(comp.data, op_node.inputs[1].data)

    # set the output stream to None
    op_node.inputs = [ref, comp]
    op_node.outputs = [None]

    with pytest.raises(AssertionError):
        assert op_node.is_balanced

    op_node.solve()
    assert op_node.is_balanced


def test_get_object():
    # Create some MassComposition objects
    data = pd.DataFrame({'wet_mass': [1000, 2000], 'mass_dry': [800, 1600]})
    input1 = Sample(data=data, name='input1')
    input2 = Sample(data=data, name='input2')
    output = Sample(data=data, name='output')

    # Create an Operation object and set its inputs and outputs
    op = Operation(name='test_operation')
    op.inputs = [input1, input2]
    op.outputs = [output]

    # Test getting an object by its name
    assert op._get_object('input1') == input1
    assert op._get_object('input2') == input2
    assert op._get_object('output') == output

    # Test getting an object without specifying a name
    # This should return the first non-None output if it exists (why the output and not input?)
    assert op._get_object() == output

    # Set the outputs to None and test getting an object without specifying a name again
    # This should return the first non-None input
    op.outputs = [None]
    assert op._get_object() == input1

    # Test getting an object with a name that doesn't exist
    # This should raise a ValueError
    with pytest.raises(ValueError):
        op._get_object('non_existent_name')


def test_solve_missing_count():
    # Create some MassComposition objects
    data = pd.DataFrame({'wet_mass': [1000, 2000], 'mass_dry': [800, 1600]})
    input1 = Sample(data=data, name='input1')
    output1 = Sample(data=data, name='output1')
    output2 = Sample(data=data, name='output2')

    # Create an Operation object and set its inputs and outputs
    op = Operation(name='test_operation')

    # Test with more than one missing inputs or outputs
    op.inputs = [input1, None]
    op.outputs = [None, None]
    with pytest.raises(ValueError):
        op.solve()

    # Test with no missing inputs or outputs and the operation is balanced
    op.inputs = [input1]
    op.outputs = [output1, output2]
    op.solve()  # This should not raise any exceptions


def test_solve_by_partition():
    """Test solving an Operation object by partitioning the input data using napier_munn_size_1mm."""
    # Create some MassComposition objects
    data = pd.DataFrame({'wet_mass': [1000., 2000.], 'mass_dry': [800., 1600.], 'Fe': [55.0, 60.0]})
    data.index = pd.IntervalIndex.from_arrays([1., 2.], [2., 3.], closed='left', name='size')
    input1 = IntervalSample(data=data, name='input1')

    # Create a PartitionOperation object and set its inputs and outputs
    partition_op = PartitionOperation(name='partition_operation', partition={'module': 'elphick.geomet.utils.partition',
                                                                             'function': 'napier_munn_size_1mm',
                                                                             'output_stream': 'Lump',
                                                                             'complement_stream': 'Fines'})
    partition_op.inputs = [input1]
    partition_op.outputs = [None, None]  # Output will be calculated

    # Solve the operation to apply the partition function
    partition_op.solve()

    # Check if the operation is balanced
    assert partition_op.is_balanced

    # Verify the output data
    expected_output_data = input1.mass_data.copy()[['mass_dry']]
    expected_output_data['mass_dry'] *= napier_munn_size_1mm(MeanIntervalIndex(expected_output_data.index).mean)
    pd.testing.assert_frame_equal(partition_op.outputs[0].mass_data[['mass_dry']], expected_output_data)


def test_operation_config_loading(tmp_path):
    # Create a temporary YAML configuration file
    config_data = {
        'flowsheet': {
            'operation': {
                'tol_abs': 1e-3,
                'tol_rel': 1e-5
            }
        }
    }
    config_file = tmp_path / "mc_config.yml"
    with open(config_file, 'w') as file:
        yaml.dump(config_data, file)

    # Initialize the Operation with the temporary config file
    op = Operation(name="test_operation", config_file=config_file)

    # Assert that the tolerances are loaded correctly
    assert op.tol_abs == 1e-3
    assert op.tol_rel == 1e-5


def test_operation_tolerance_adjustment():
    # Create input and output data with a subtle difference exceeding default tolerances
    input_data = pd.DataFrame({'mass_wet': [1000.001, 2000.001], 'mass_dry': [800.001, 1600.001]})
    output_data = pd.DataFrame({'mass_wet': [1000.0, 2000.0], 'mass_dry': [800.0, 1600.0]})

    # Create an Operation instance with default tolerances
    op = Operation(name="test_operation")
    op.inputs = [Sample(data=input_data, name="input")]
    op.outputs = [Sample(data=output_data, name="output")]

    # Assert that the operation is not balanced with default tolerances
    assert not op.is_balanced
    assert not op.unbalanced_records.empty

    # Adjust tolerances to make the operation pass
    op.tol_abs = 1.0e-03
    op.check_balance()

    # Assert that the operation is now balanced
    assert op.is_balanced
    assert op.unbalanced_records.empty