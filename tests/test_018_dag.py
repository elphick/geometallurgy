import pytest

from elphick.geomet import Sample
from elphick.geomet.flowsheet.dag import DAG
from fixtures import sample_data


def test_dag_runs_chained_split_operations(sample_data):
    feed = Sample(sample_data, name='Feed')
    dag = (DAG()
           .add_input('Feed')
           .add_step('First split', Sample.split, ['Feed'],
                     {'fraction': 0.4, 'name_1': 'Product', 'name_2': 'Tailings'})
           .add_step('Product split', Sample.split, ['Product'],
                     {'fraction': 0.5, 'name_1': 'Final product', 'name_2': 'Middlings'})
           .add_output('Final product output', 'Final product')
           .add_output('Middlings output', 'Middlings')
           .add_output('Tailings output', 'Tailings'))

    dag.run({'Feed': feed}, progress_bar=False)

    assert set(dag.streams) == {'Feed', 'Product', 'Tailings', 'Final product', 'Middlings'}
    assert dag.streams['Final product'].name == 'Final product'
    assert dag.streams['Final product'].mass_data.equals(feed.mass_data * 0.2)
    assert dag.node_executed and all(dag.node_executed.values())


def test_dag_rejects_unknown_stream_without_mutating_graph():
    dag = DAG().add_input('Feed')

    with pytest.raises(ValueError, match='No parent node'):
        dag.add_step('Split', Sample.split, ['Missing'], {'fraction': 0.5})

    assert dag.all_nodes_ == ['Feed']


def test_dag_rejects_duplicate_input_streams():
    dag = DAG().add_input('Feed')

    with pytest.raises(ValueError, match='same stream more than once'):
        dag.add_step('Merge', lambda *streams: streams[0], ['Feed', 'Feed'])

    assert dag.all_nodes_ == ['Feed']


def test_dag_rejects_cycles_before_execution(sample_data):
    dag = DAG().add_input('Feed').add_step(
        'Split', Sample.split, ['Feed'],
        {'fraction': 0.5, 'name_1': 'Product', 'name_2': 'Tailings'}
    )
    dag.add_output('Product output', 'Product')
    dag.graph.add_edge('Product output', 'Split')

    with pytest.raises(ValueError, match='Directed Acyclic Graph'):
        dag.run({'Feed': Sample(sample_data, name='Feed')}, progress_bar=False)
