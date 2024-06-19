"""
Flowsheet Basics
================

Related Sample objects can be managed as a network.  In the Process Engineering/Metallurgy
disciplines the network will often be called a `flowsheet`.

"""
from copy import deepcopy
from typing import Dict

import pandas as pd
from matplotlib import pyplot as plt

from elphick.geomet import Stream, Flowsheet
from elphick.geomet.operation import OP, Operation
from elphick.geomet.utils.data import sample_data

# %%
#
# Create some Sample objects
# --------------------------
#
# Create an object, and split it to create two more objects.

df_data: pd.DataFrame = sample_data()
obj_strm: Stream = Stream(df_data, name='Feed')
obj_strm_1, obj_strm_2 = obj_strm.split(0.4, name_1='stream 1', name_2='stream 2')

# %%
# Placeholder random nodes are created for each Sample object.
# This is done to capture the relationships implicitly defined by any math operations performed on the objects.

for obj in [obj_strm, obj_strm_1, obj_strm_2]:
    print(obj.name, obj._nodes)

# %%
#
# Create a Flowsheet object
# -------------------------
#
# This requires passing an Iterable of Sample objects

fs: Flowsheet = Flowsheet.from_objects([obj_strm, obj_strm_1, obj_strm_2])

# %%
# Print the node object detail

for node in fs.graph.nodes:
    print(fs.graph.nodes[node]['mc'])

# %%
# Note that the random node placeholder integers have been renumbered for readability.

for obj in [obj_strm, obj_strm_1, obj_strm_2]:
    print(obj.name, obj._nodes)

# %%
# Print the overall network balanced status
#
# NOTE: presently this only includes node balance status
# edge balance status will assure the mass-moisture balance is satisfied

print(fs.balanced)

# %%
# Plot the network.
# Imbalanced Nodes will appear red.  Later, Imbalanced Edges will also appear red.

fs.plot()
plt

# %%
# Display the weight averages for all edges (streams) in the network (flowsheet)

df_report: pd.DataFrame = fs.report()
df_report

# %%

df_report: pd.DataFrame = fs.report(apply_formats=True)
df_report

# %%
# Plot the interactive network using plotly

fig = fs.plot_network()
fig

# %%
# Plot the Sankey

fig = fs.plot_sankey()
fig

# %%
# Demonstrate the table-plot

fig = fs.table_plot(plot_type='sankey', table_pos='top', table_area=0.3).update_layout(height=700)
fig

# %%

fig = fs.table_plot(plot_type='network', table_pos='bottom', table_area=0.3).update_layout(height=700)
fig

# %%
#
# Expand the Network with Math Operators
# --------------------------------------
#

obj_strm_3, obj_strm_4 = obj_strm_2.split(0.8, name_1='stream 3', name_2='stream 4')
obj_strm_5 = obj_strm_1.add(obj_strm_3, name='stream 5')

fs2: Flowsheet = Flowsheet.from_objects([obj_strm, obj_strm_1, obj_strm_2, obj_strm_3, obj_strm_4, obj_strm_5])

fig = fs2.table_plot(plot_type='sankey', table_pos='left')
fig

# %%
#
# Setting Node names
# ------------------

nodes_before: Dict[int, Operation] = fs.nodes_to_dict()
print({n: o.name for n, o in nodes_before.items()})

# %%
fs.set_node_names(node_names={0: 'node_0', 1: 'node_1', 2: 'node_2', 3: 'node_3'})
nodes_after: Dict[int, Operation] = fs.nodes_to_dict()
print({n: o.name for n, o in nodes_after.items()})

# %%
#
# Setting Stream data
# -------------------
#
# First we show how to easily access the stream data as a dictionary

stream_data: Dict[str, Stream] = fs.streams_to_dict()
print(stream_data.keys())

# %%
# We will replace stream 2 with the same data as stream 1.

new_stream: Stream = deepcopy(fs.get_edge_by_name('stream 1'))
# we need to rename to avoid a creating a duplicate stream name
new_stream.name = 'stream 1 copy'
fs.set_stream_data({'stream 2': new_stream})
print(fs.streams_to_dict().keys())

# %%
# Of course the network is now unbalanced as highlighted in the Sankey

fig = fs.table_plot()
fig

# %%
#
# Methods to modify relationships
# -------------------------------
#
# Sometimes the network that is automatically created may not be what you are after - for example flow may be in
# the wrong direction.  We'll learn how to modify an existing network, by picking up the network above.
#
# Let's break the links for the stream 1.

fs.reset_stream_nodes(stream="stream 1")
fig = fs.table_plot()
fig

# %%
# We'll now break all remaining connections (we could have done this from the start).

fs.reset_stream_nodes()
fig = fs.table_plot()
fig

# %%
# Now we'll create some linkages - of course they will be completely rubbish and not balance.

fs.set_stream_parent(stream="stream 1", parent="Feed")
fs.set_stream_child(stream="stream 1", child="stream 1 copy")
fig = fs.table_plot()
fig

# %%
# Perhaps less useful, but possible, we can build relationships by setting nodes directly.

fs.reset_stream_nodes()
fs.set_nodes(stream="stream 1", nodes=(1, 2))
fs.set_nodes(stream="stream 1 copy", nodes=(2, 3))
fig = fs.table_plot()
fig
