import pandas as pd

from elphick.geomet import Stream
from elphick.geomet.flowsheet import Flowsheet
from elphick.geomet.operation import Operation
from fixtures import sample_data


def test_flowsheet_init(sample_data):
    obj_strm: Stream = Stream(sample_data, name='Feed')
    obj_strm_1, obj_strm_2 = obj_strm.split(0.4, name_1='stream 1', name_2='stream 2')
    fs: Flowsheet = Flowsheet.from_objects([obj_strm, obj_strm_1, obj_strm_2])

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


def test_solve(sample_data):
    # Create a new Flowsheet object
    obj_strm: Stream = Stream(sample_data, name='Feed')
    obj_strm_1, obj_strm_2 = obj_strm.split(0.4, name_1='stream 1', name_2='stream 2')
    fs: Flowsheet = Flowsheet.from_objects([obj_strm, obj_strm_1, obj_strm_2])

    # set one edge to None
    fs.set_stream_data(stream_data={'stream 2': None})

    # Call the solve method
    fs.solve()

    # Check that the solve method has filled in the missing MC object
    for u, v, data in fs.graph.edges(data=True):
        assert data['mc'] is not None, f"Edge ({u}, {v}) has not been filled in by solve method"

    # Check that the missing_count is zero
    missing_count = sum([1 for u, v, d in fs.graph.edges(data=True) if d['mc'] is None])
    assert missing_count == 0, "There are still missing MC objects after calling solve method"


def test_query(sample_data):
    # Create a new Flowsheet object
    obj_strm: Stream = Stream(sample_data, name='Feed')
    obj_strm_1, obj_strm_2 = obj_strm.split(0.4, name_1='stream 1', name_2='stream 2')
    fs: Flowsheet = Flowsheet.from_objects([obj_strm, obj_strm_1, obj_strm_2])

    # Call the query method with inplace=False
    fs_reduced: Flowsheet = fs.query(expr='Fe>58', inplace=False)

    # Check that the original flowsheet remains unmutated
    assert fs.get_input_streams()[0].data.equals(obj_strm.data)
    assert fs.get_output_streams()[0].data.equals(obj_strm_1.data)
    assert fs.get_output_streams()[1].data.equals(obj_strm_2.data)

    # Check that the filtered flowsheet has the correct data
    assert not fs_reduced.get_input_streams()[0].data.equals(obj_strm.data)
    assert not fs_reduced.get_output_streams()[0].data.equals(obj_strm_1.data)
    assert not fs_reduced.get_output_streams()[1].data.equals(obj_strm_2.data)

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
