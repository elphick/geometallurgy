"""
Define and Run a DAG
===================

A directed acyclic graph (DAG) is useful for describing a processing circuit
whose streams are created by operations such as splitting and combining.
This example builds and runs a small circuit from named streams and operations.
"""
import matplotlib.pyplot as plt

from elphick.geomet import Sample
from elphick.geomet.flowsheet import DAG
from elphick.geomet.utils.data import sample_data

# %%
# Define the DAG
# --------------
#
# The graph combines two feed streams, splits the combined feed twice, and
# recombines the resulting streams into two products.

feed_data = sample_data()
feed_1 = Sample(feed_data, name='Feed 1')
feed_2 = Sample(feed_data.copy(), name='Feed 2')

dag = DAG(name='Beneficiation circuit')
dag.add_input('feed_1')
dag.add_input('feed_2')
dag.add_step('combine feeds', Sample.add, ['feed_1', 'feed_2'], {'name': 'feed'})
dag.add_step('primary split', Sample.split, ['feed'],
             {'fraction': 0.3, 'name_1': 'lump', 'name_2': 'fines'})
dag.add_step('lump split', Sample.split, ['lump'],
             {'fraction': 0.3, 'name_1': 'coarse product', 'name_2': 'lump middlings'})
dag.add_step('fines split', Sample.split, ['fines'],
             {'fraction': 0.3, 'name_1': 'fine product', 'name_2': 'fine middlings'})
dag.add_step('combine coarse streams', Sample.add, ['coarse product', 'fine product'],
             {'name': 'coarse concentrate'})
dag.add_step('combine middlings', Sample.add, ['lump middlings', 'fine middlings'],
             {'name': 'middlings'})
dag.add_output('coarse concentrate output', 'coarse concentrate')
dag.add_output('middlings output', 'middlings')

# %%
# Run the DAG
# -----------
#
# Input streams must have compatible indexes. The outputs are available from
# the DAG's stream mapping after execution.

dag.run({'feed_1': feed_1, 'feed_2': feed_2}, progress_bar=False)
dag.streams['coarse concentrate'].aggregate

# %%
# Visualize the graph
# -------------------

dag.plot()
plt
