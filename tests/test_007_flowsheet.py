from pathlib import Path

import pandas as pd
import pytest
import yaml

from elphick.geomet import Sample, IntervalSample
from elphick.geomet.flowsheet import Flowsheet
from elphick.geomet.flowsheet.operation import Operation
from elphick.geomet.flowsheet.stream import Stream
from fixtures import sample_data, interval_sample_data


def test_flowsheet_init(sample_data):
    obj_in: Sample = Sample(sample_data, name='Feed')
    obj_out_1, obj_out_2 = obj_in.split(0.4, name_1='stream 1', name_2='stream 2')
    fs: Flowsheet = Flowsheet.from_objects([obj_in, obj_out_1, obj_out_2])

    # Check that the Flowsheet object has been created
    assert isinstance(fs, Flowsheet), "Flowsheet object has not been created"

    # Check that the Flowsheet object contains the correct number of nodes
    assert len(fs.graph.nodes) == 4, "Flowsheet object does not contain the correct number of nodes"

    # Check that the Flowsheet object contains the correct number of edges
    assert len(fs.graph.edges) == 3, "Flowsheet object does not contain the correct number of edges"

    # Check that the nodes have the correct OP objects
    for node in fs.graph.nodes:
        assert isinstance(fs.graph.nodes[node]['mc'], Operation), f"Node {node} does not have a OP object"

    # Check that the edges have the correct MC objects
    for u, v, data in fs.graph.edges(data=True):
        assert isinstance(data['mc'], Stream), f"Edge ({u}, {v}) does not have a MC object"


def test_solve_output(sample_data):
    # Create a new Flowsheet object
    obj_in: Sample = Sample(sample_data, name='Feed')
    obj_out_1, obj_out_2 = obj_in.split(0.4, name_1='stream 1', name_2='stream 2')
    fs: Flowsheet = Flowsheet.from_objects([obj_in, obj_out_1, obj_out_2])

    # set one output edge to None
    fs.set_stream_data(stream_data={'stream 2': None})

    # Call the solve method
    fs.solve()

    # Check that the solve method has filled in the missing MC object
    for u, v, data in fs.graph.edges(data=True):
        assert data['mc'] is not None, f"Edge ({u}, {v}) has not been filled in by solve method"

    # Check that the missing_count is zero
    missing_count = sum([1 for u, v, d in fs.graph.edges(data=True) if d['mc'] is None])
    assert missing_count == 0, "There are still missing MC objects after calling solve method"


def test_solve_input(sample_data):
    # Create a new Flowsheet object
    obj_in: Sample = Sample(sample_data, name='Feed')
    obj_out_1, obj_out_2 = obj_in.split(0.4, name_1='stream 1', name_2='stream 2')
    fs: Flowsheet = Flowsheet.from_objects([obj_in, obj_out_1, obj_out_2])

    # set the input edge to None
    fs.set_stream_data(stream_data={'Feed': None})

    # Call the solve method
    fs.solve()

    # Check that the solve method has filled in the missing MC object
    for u, v, data in fs.graph.edges(data=True):
        assert data['mc'] is not None, f"Edge ({u}, {v}) has not been filled in by solve method"

    # Check that the missing_count is zero
    missing_count = sum([1 for u, v, d in fs.graph.edges(data=True) if d['mc'] is None])
    assert missing_count == 0, "There are still missing MC objects after calling solve method"


def test_report_with_missing(sample_data):
    # Create a new Flowsheet object
    obj_in: Sample = Sample(sample_data, name='Feed')
    obj_out_1, obj_out_2 = obj_in.split(0.4, name_1='stream 1', name_2='stream 2')
    fs: Flowsheet = Flowsheet.from_objects([obj_in, obj_out_1, obj_out_2])

    # set the input edge to None
    fs.set_stream_data(stream_data={'Feed': None})

    with pytest.raises(KeyError):
        fs.report()


def test_query(sample_data):
    # Create a new Flowsheet object
    obj_in: Sample = Sample(sample_data, name='Feed')
    obj_out_1, obj_out_2 = obj_in.split(0.4, name_1='stream 1', name_2='stream 2')
    fs: Flowsheet = Flowsheet.from_objects([obj_in, obj_out_1, obj_out_2])

    # Call the query method with inplace=False
    fs_reduced: Flowsheet = fs.query(expr='Fe>58', inplace=False)

    # Check that the original flowsheet remains unmutated
    assert fs.get_input_streams()[0].data.equals(obj_in.data)
    assert fs.get_output_streams()[0].data.equals(obj_out_1.data)
    assert fs.get_output_streams()[1].data.equals(obj_out_2.data)

    # Check that the filtered flowsheet has the correct data
    assert not fs_reduced.get_input_streams()[0].data.equals(obj_in.data)
    assert not fs_reduced.get_output_streams()[0].data.equals(obj_out_1.data)
    assert not fs_reduced.get_output_streams()[1].data.equals(obj_out_2.data)

    # Call the query method with inplace=True
    fs.query(expr='Fe>58', inplace=True)

    # Check that the original flowsheet is now mutated, matching the fs_reduced
    assert fs.get_input_streams()[0].data.equals(fs_reduced.get_input_streams()[0].data)
    assert fs.get_output_streams()[0].data.equals(fs_reduced.get_output_streams()[0].data)
    assert fs.get_output_streams()[1].data.equals(fs_reduced.get_output_streams()[1].data)


def test_filter_by_index(sample_data):
    # Create a new Flowsheet object
    obj_strm: Stream = Stream(sample_data, name='Feed')
    obj_strm_1, obj_strm_2 = obj_strm.split(0.4, name_1='stream 1', name_2='stream 2')
    fs: Flowsheet = Flowsheet.from_objects([obj_strm, obj_strm_1, obj_strm_2])

    # Call the filter method with inplace=False
    index = obj_strm.data.index[1:]
    fs_reduced: Flowsheet = fs.filter_by_index(index=index, inplace=False)

    # Check that the original flowsheet remains unmutated
    assert fs.get_input_streams()[0].data.equals(obj_strm.data)
    assert fs.get_output_streams()[0].data.equals(obj_strm_1.data)
    assert fs.get_output_streams()[1].data.equals(obj_strm_2.data)

    # Check that the filtered flowsheet has the correct data
    assert not fs_reduced.get_input_streams()[0].data.equals(obj_strm.data)
    assert not fs_reduced.get_output_streams()[0].data.equals(obj_strm_1.data)
    assert not fs_reduced.get_output_streams()[1].data.equals(obj_strm_2.data)

    # Call the filter method with inplace=True
    fs.filter_by_index(index=index, inplace=True)

    # Check that the original flowsheet is now mutated, matching the fs_reduced
    assert fs.get_input_streams()[0].data.equals(fs_reduced.get_input_streams()[0].data)
    assert fs.get_output_streams()[0].data.equals(fs_reduced.get_output_streams()[0].data)
    assert fs.get_output_streams()[1].data.equals(fs_reduced.get_output_streams()[1].data)


def test_flowsheet_from_yaml():
    # Create the flowsheet object
    flowsheet = Flowsheet.from_yaml(Path(__file__).parents[1] / 'elphick/geomet/config'
                                                                '/flowsheet_example_simple.yaml')

    # Verify the flowsheet object has been created
    assert isinstance(flowsheet, Flowsheet), "Flowsheet object has not been created"


def test_solve_flowsheet_simple(interval_sample_data):
    # Create the flowsheet object
    flowsheet = Flowsheet.from_yaml(Path(__file__).parents[1] / 'elphick/geomet/config'
                                                                '/flowsheet_example_simple.yaml')

    feed_sample: IntervalSample = IntervalSample(interval_sample_data, name='Feed')

    df_coarse: pd.DataFrame = interval_sample_data.copy()
    df_coarse['wet_mass'] = df_coarse['wet_mass'] * 0.4
    df_coarse['mass_dry'] = df_coarse['mass_dry'] * 0.4
    obj_coarse: Stream = Stream(df_coarse, name='Coarse')

    flowsheet.set_stream_data(stream_data={'Feed': feed_sample, 'Coarse': obj_coarse})

    flowsheet.plot().show()

    # Solve the flowsheet
    flowsheet.solve()

    flowsheet.plot().show()


def test_solve_flowsheet_partition(interval_sample_data):
    # Create the flowsheet object
    flowsheet = Flowsheet.from_yaml(Path(__file__).parents[1] / 'elphick/geomet/config'
                                                                '/flowsheet_example_partition.yaml')

    # add the feed sample to the flowsheet
    feed_sample: IntervalSample = IntervalSample(interval_sample_data, name='Feed')
    flowsheet.set_stream_data({'Feed': feed_sample})
    flowsheet.plot().show()

    # Solve the flowsheet
    flowsheet.solve()

    flowsheet.plot().show()
