import pandas as pd
import pytest

from elphick.geomet import Sample
from elphick.geomet.flowsheet import Flowsheet
from elphick.geomet.flowsheet.operation import Operation
from elphick.geomet.flowsheet.stream import Stream
from fixtures import sample_data


@pytest.fixture
def unhealthy_network(sample_data: pd.DataFrame) -> Flowsheet:
    obj_in: Sample = Sample(sample_data, name='Feed')
    obj_out_1, obj_out_2 = obj_in.split(0.4, name_1='stream 1', name_2='stream 2')
    fs: Flowsheet = Flowsheet.from_objects([obj_in, obj_out_1, obj_out_2])

    # unbalance the network
    df_stream_2: pd.DataFrame = obj_out_2.data.copy().drop(columns=['H2O'])
    df_stream_2 = df_stream_2 * 3.0
    obj_out_2: Stream = Stream(df_stream_2, name='stream 2')
    fs.set_stream_data({'stream 2': obj_out_2})

    return fs


def test_flowsheet_nodes_unhealthy_report(unhealthy_network):
    fs = unhealthy_network
    fs.table_plot(plot_type='network').show()

    df_nodes: pd.DataFrame = fs.unhealthy_node_records()
    assert df_nodes.shape == (3, 7)


def test_flowsheet_streams_unhealthy_report(unhealthy_network):
    fs = unhealthy_network
    fs.table_plot(plot_type='network').show()

    df_streams: pd.DataFrame = fs.unhealthy_stream_records()
    assert df_streams.shape == (3, 32)
