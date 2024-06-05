import pandas as pd
import pytest

from geomet import Sample, Operation
from geomet.utils.data import sample_data


@pytest.fixture
def expected_data() -> pd.DataFrame:
    expected_data = sample_data(include_wet_mass=True,
                                include_dry_mass=True,
                                include_moisture=True)
    expected_data.columns = [col.lower() for col in expected_data.columns]
    expected_data.rename(columns={'wet_mass': 'mass_wet'}, inplace=True)
    return expected_data


def test_operation_split(expected_data):
    data = sample_data(include_moisture=True)
    smpl = Sample(data=data, name='sample')
    ref, comp = smpl.split(fraction=0.5)
    pd.testing.assert_frame_equal(ref.data, comp.data)

    op_node: Operation = Operation(name='split')
    op_node.input_streams = [smpl]
    op_node.output_streams = [ref, comp]
    assert op_node.is_balanced()


def test_operation_add(expected_data):
    data = sample_data()
    smpl = Sample(data=data, name='sample')
    ref, comp = smpl.split(fraction=0.5, include_supplementary_data=True)
    smpl_new = ref.add(comp, name='sample_new', include_supplementary_data=True)
    pd.testing.assert_frame_equal(smpl.data, smpl_new.data)

    op_node: Operation = Operation(name='add')
    op_node.input_streams = [smpl]
    op_node.output_streams = [smpl_new]
    assert op_node.is_balanced()


def test_operation_sub(expected_data):
    data = sample_data()
    smpl = Sample(data=data, name='sample')
    ref, comp = smpl.split(fraction=0.5, include_supplementary_data=True)
    ref_new = smpl.sub(comp, name='ref_new', include_supplementary_data=True)
    pd.testing.assert_frame_equal(ref.data, ref_new.data)

    op_node: Operation = Operation(name='add')
    op_node.input_streams = [ref]
    op_node.output_streams = [ref_new]
    assert op_node.is_balanced()


def test_operation_imbalance_split(expected_data):
    data = sample_data(include_moisture=True)
    smpl = Sample(data=data, name='sample')
    ref, comp = smpl.split(fraction=0.5)

    # introduce imbalance
    new_data: pd.DataFrame = comp.data.copy()
    new_data.loc[0, 'wet_mass'] = 1000
    comp.data = new_data

    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(ref.data, comp.data)

    op_node: Operation = Operation(name='split')
    op_node.input_streams = [smpl]
    op_node.output_streams = [ref, comp]
    with pytest.raises(AssertionError):
        assert op_node.is_balanced()

    df_imbalance: pd.DataFrame = op_node.get_failed_records()
    print(df_imbalance)


def test_operation_solve(expected_data):
    data = sample_data(include_moisture=True)
    smpl = Sample(data=data, name='sample')
    ref, comp = smpl.split(fraction=0.5)

    # set a stream to empty
    comp.data = None

    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(ref.data, comp.data)

    op_node: Operation = Operation(name='split')
    op_node.input_streams = [smpl]
    op_node.output_streams = [ref, comp]
    with pytest.raises(AssertionError):
        assert op_node.is_balanced()

    df_imbalance: pd.DataFrame = op_node.get_failed_records()
    print(df_imbalance)
