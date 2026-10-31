from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from elphick.geomet import Sample, IntervalSample
from elphick.geomet.flowsheet import Flowsheet
from elphick.geomet.flowsheet.operation import Operation
from elphick.geomet.flowsheet.stream import Stream
from elphick.geomet.utils.partition import register_partition_model
from fixtures import sample_data, interval_sample_data

import logging

logging.basicConfig(level=logging.DEBUG)


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


def _constant_partition(size, fraction):
    return np.full_like(size, fraction, dtype=float)


def _two_feed_config():
    register_partition_model('test_constant_partition', _constant_partition, replace=True)
    path = Path(__file__).parents[1] / 'examples/03_flowsheet/two_feed_partition.yaml'
    config = yaml.safe_load(path.read_text())
    for node in ('screen', 'screen_2'):
        config['FLOWSHEET']['operations'][node]['partition'] = {
            'model': 'test_constant_partition',
            'args': {'fraction': 0.4 if node == 'screen' else 0.5},
            'partition_stream': 'Coarse' if node == 'screen' else 'Coarse 2',
        }
    return config


def _make_feed(data, name, scale=1.0):
    sample_data = data.copy()
    sample_data[['wet_mass', 'mass_dry']] *= scale
    return IntervalSample(sample_data, name=name)


def test_yaml_partition_circuit_registered_model_and_feed_replacement(interval_sample_data):
    register_partition_model('test_constant_partition', _constant_partition, replace=True)
    flowsheet = Flowsheet.from_dict(_two_feed_config())
    feed_a = _make_feed(interval_sample_data, 'Feed A')
    feed_b = _make_feed(interval_sample_data, 'Feed B', scale=2.0)
    flowsheet.set_stream_data({'Feed A': feed_a, 'Feed B': feed_b})
    flowsheet.solve()

    streams = flowsheet.streams_to_dict()
    combined_mass = feed_a.mass_data + feed_b.mass_data
    pd.testing.assert_frame_equal(
        streams['Coarse'].mass_data,
        combined_mass.mul(0.4, axis=0),
        check_names=False,
    )
    pd.testing.assert_frame_equal(
        streams['Coarse 2'].mass_data,
        combined_mass.mul(0.6 * 0.5, axis=0),
        check_names=False,
    )
    pd.testing.assert_frame_equal(
        streams['Fine 2'].mass_data,
        combined_mass.mul(0.6 * 0.5, axis=0),
        check_names=False,
    )
    for node in flowsheet.graph.nodes:
        operation = flowsheet.graph.nodes[node]['mc']
        if operation.inputs and operation.outputs:
            assert operation.is_balanced

    old_product_mass = streams['Coarse Product'].mass_data['mass_dry'].sum()
    replacement = _make_feed(interval_sample_data, 'Feed A', scale=3.0)
    flowsheet.set_stream_data({'Feed A': replacement})
    flowsheet.solve()
    assert flowsheet.streams_to_dict()['Coarse Product'].mass_data['mass_dry'].sum() > old_product_mass

    filtered = flowsheet.filter_by_index(feed_a.mass_data.index[:2], inplace=False)
    old_filtered_product_mass = filtered.streams_to_dict()['Coarse Product'].mass_data['mass_dry'].sum()
    filtered.set_stream_data({
        'Feed A': _make_feed(interval_sample_data.iloc[:2], 'Feed A', scale=5.0),
    })
    filtered.solve()
    assert filtered.streams_to_dict()['Coarse Product'].mass_data['mass_dry'].sum() \
        > old_filtered_product_mass

    changed_config = _two_feed_config()
    changed_config['FLOWSHEET']['operations']['screen']['partition']['args']['fraction'] = 0.25
    changed_flowsheet = Flowsheet.from_dict(changed_config)
    changed_flowsheet.set_stream_data({
        'Feed A': _make_feed(interval_sample_data, 'Feed A'),
        'Feed B': _make_feed(interval_sample_data, 'Feed B', scale=2.0),
    })
    changed_flowsheet.solve()
    expected = (feed_a.mass_data + feed_b.mass_data).mul(0.25, axis=0)
    pd.testing.assert_frame_equal(
        changed_flowsheet.streams_to_dict()['Coarse'].mass_data, expected, check_names=False)


def test_yaml_partition_circuit_rejects_missing_and_unknown_feeds(interval_sample_data):
    flowsheet = Flowsheet.from_dict(_two_feed_config())
    with pytest.raises(KeyError, match='Unknown stream'):
        flowsheet.set_stream_data({'Typo': _make_feed(interval_sample_data, 'Typo')})
    flowsheet.set_stream_data({'Feed A': _make_feed(interval_sample_data, 'Feed A')})
    with pytest.raises(ValueError, match='missing required feeds.*Feed B'):
        flowsheet.solve()


def test_yaml_partition_circuit_rejects_incompatible_fraction_grids(interval_sample_data):
    flowsheet = Flowsheet.from_dict(_two_feed_config())
    incompatible_data = interval_sample_data.copy()
    incompatible_data.index = pd.IntervalIndex.from_breaks([1, 2, 3, 5], closed='left', name='size')
    with pytest.raises(ValueError, match="Operation 'Combine feeds'.*indexes"):
        flowsheet.set_stream_data({
            'Feed A': _make_feed(interval_sample_data, 'Feed A'),
            'Feed B': _make_feed(incompatible_data, 'Feed B'),
        })


def test_yaml_partition_circuit_requires_fractionated_feeds(interval_sample_data):
    flowsheet = Flowsheet.from_dict(_two_feed_config())
    with pytest.raises(TypeError, match="Feed stream 'Feed A'.*IntervalSample"):
        flowsheet.set_stream_data({
            'Feed A': Sample(interval_sample_data, name='Feed A'),
            'Feed B': _make_feed(interval_sample_data, 'Feed B'),
        })


@pytest.mark.parametrize(
    ('change', 'message'),
    [
        (lambda config: config['FLOWSHEET']['operations']['screen'].update({'type': 'Unknown'}),
         "unknown type"),
        (lambda config: config['FLOWSHEET']['operations']['screen']['partition'].update(
            {'partition_stream': 'Unknown'}), "partition_stream"),
        (lambda config: config['FLOWSHEET']['operations'].pop('screen_2'), "Operation definitions"),
        (lambda config: config['FLOWSHEET']['operations']['screen']['partition'].update(
            {'model': 'not_registered'}), "Unknown partition model"),
    ],
)
def test_yaml_partition_configuration_validation(change, message):
    config = _two_feed_config()
    register_partition_model('test_constant_partition', _constant_partition, replace=True)
    change(config)
    with pytest.raises((ValueError, KeyError), match=message):
        Flowsheet.from_dict(config)


def test_yaml_partition_configuration_rejects_edge_collisions_and_cycles():
    config = _two_feed_config()
    config['FLOWSHEET']['streams']['Feed B'].update({'node_in': 'feed_a', 'node_out': 'mixer'})
    with pytest.raises(ValueError, match='share source/target edge'):
        Flowsheet.from_dict(config)

    config = _two_feed_config()
    config['FLOWSHEET']['streams']['Combined Feed']['node_out'] = 'mixer'
    with pytest.raises(ValueError, match='acyclic'):
        Flowsheet.from_dict(config)


def test_yaml_partition_configuration_rejects_invalid_args_and_ambiguous_resolver():
    config = _two_feed_config()
    register_partition_model('test_constant_partition', _constant_partition, replace=True)
    config['FLOWSHEET']['operations']['screen']['partition']['args'] = {'not_a_parameter': 1}
    with pytest.raises(ValueError, match="Invalid arguments.*not_a_parameter"):
        Flowsheet.from_dict(config)

    config = _two_feed_config()
    config['FLOWSHEET']['operations']['screen']['partition'].update({
        'module': 'elphick.geomet.utils.partition',
        'function': 'napier_munn_size',
    })
    with pytest.raises(ValueError, match='both.*model'):
        Flowsheet.from_dict(config)


def test_partition_results_must_be_finite_and_in_range(interval_sample_data):
    register_partition_model('test_invalid_partition', lambda size: np.full_like(size, 1.1), replace=True)
    config = _two_feed_config()
    config['FLOWSHEET']['operations']['screen']['partition'] = {
        'model': 'test_invalid_partition',
        'partition_stream': 'Coarse',
    }
    flowsheet = Flowsheet.from_dict(config)
    flowsheet.set_stream_data({
        'Feed A': _make_feed(interval_sample_data, 'Feed A'),
        'Feed B': _make_feed(interval_sample_data, 'Feed B'),
    })
    with pytest.raises(ValueError, match="model 'test_invalid_partition'.*within \\[0, 1\\]"):
        flowsheet.solve()


def test_partition_configuration_reports_missing_model_parameters(interval_sample_data):
    path = Path(__file__).parents[1] / 'elphick/geomet/config/flowsheet_example_partition.yaml'
    config = yaml.safe_load(path.read_text())
    config['FLOWSHEET']['operations']['screen']['partition']['args'].pop('ep')
    flowsheet = Flowsheet.from_dict(config)
    flowsheet.set_stream_data({'Feed': IntervalSample(interval_sample_data, name='Feed')})
    with pytest.raises(ValueError, match="model 'napier_munn_size'.*configure all non-dimension parameters"):
        flowsheet.solve()


def test_ordinary_operation_does_not_guess_multiple_products(interval_sample_data):
    path = Path(__file__).parents[1] / 'elphick/geomet/config/flowsheet_example_simple.yaml'
    flowsheet = Flowsheet.from_yaml(path)
    flowsheet.set_stream_data({'Feed': IntervalSample(interval_sample_data, name='Feed')})
    with pytest.raises(ValueError, match="unresolved streams:.*Coarse.*Fine.*blocking operations"):
        flowsheet.solve()


@pytest.mark.skip("This test is slow, and for a solved problem. #52")
def test_flowsheet_multiple_runs(sample_data):
    success_count = 0
    num_runs = 500

    for i in range(num_runs):
        try:
            # Create a new Flowsheet object
            obj_in = Sample(sample_data, name='Feed')
            obj_out_1, obj_out_2 = obj_in.split(0.4, name_1='stream 1', name_2='stream 2')
            fs = Flowsheet.from_objects([obj_in, obj_out_1, obj_out_2])

            # set the input edge to None
            fs.set_stream_data(stream_data={'Feed': None})

            # Call the solve method
            fs.solve()

            # Check that the solve method has filled in the missing MC object
            for u, v, data in fs.graph.edges(data=True):
                assert data[
                           'mc'] is not None, f"Edge ({u}, {v}) has not been filled in by solve method for iteration {i}"

            # Check that the missing_count is zero
            missing_count = sum([1 for u, v, d in fs.graph.edges(data=True) if d['mc'] is None])
            assert missing_count == 0, f"There are still missing MC objects after calling solve method for iteration {i}"

            success_count += 1
        except Exception as e:
            print(f"Run failed with exception: {e} for iteration {i}")

    print(f"Number of successful runs: {success_count}/{num_runs}")
    assert success_count == num_runs, f"Only {success_count} out of {num_runs} runs were successful"


@pytest.mark.skip("This test is slow, and for a solved problem. #52")
def test_no_self_loop_edges(sample_data):
    num_runs = 500
    for _ in range(num_runs):
        obj_in = Sample(sample_data, name='Feed')
        obj_out_1, obj_out_2 = obj_in.split(0.4, name_1='stream 1', name_2='stream 2')
        fs = Flowsheet.from_objects([obj_in, obj_out_1, obj_out_2])

        # Check for self-loop edges
        for u, v in fs.graph.edges():
            assert u != v, f"Self-loop edge ({u}, {u}) found"


if __name__ == "__main__":
    pytest.main([__file__])
