"""
YAML-Driven Flowsheet Simulation
================================

Define a multi-feed circuit, partition models, and product routing in YAML,
then solve it by binding fractionated ``IntervalSample`` feeds to named streams.
This uses the ``Flowsheet`` API directly; a separate DAG or Python ``run``
callable is not needed.
"""
from pathlib import Path

import pandas as pd

from elphick.geomet import IntervalSample
from elphick.geomet.flowsheet import Flowsheet

# %%
# Load the YAML circuit
# ---------------------

yaml_path = Path(__file__).with_name('two_feed_partition.yaml')
print(yaml_path.read_text())
flowsheet = Flowsheet.from_yaml(yaml_path)

# %%
# Bind fractionated feeds by their YAML stream names
# --------------------------------------------------

size_index = pd.IntervalIndex.from_breaks([0.1, 0.5, 1.0, 2.0, 4.0], closed='left', name='size')
feed_data = pd.DataFrame(
    {
        'mass_dry': [20.0, 30.0, 40.0, 10.0],
        'Fe': [58.0, 60.0, 62.0, 64.0],
        'SiO2': [5.0, 4.5, 3.5, 2.0],
    },
    index=size_index,
)
feed_a = IntervalSample(feed_data.copy(), name='Feed A', moisture_in_scope=False)
feed_b_data = feed_data.copy()
feed_b_data['mass_dry'] *= 0.75
feed_b = IntervalSample(feed_b_data, name='Feed B', moisture_in_scope=False)

flowsheet.set_stream_data({'Feed A': feed_a, 'Feed B': feed_b})
flowsheet.solve()

# %%
# Inspect the solved network and stream report
# --------------------------------------------

network = flowsheet.plot_network()
network

# %%
report_and_sankey = flowsheet.table_plot(plot_type='sankey')
report_and_sankey

# %%
stream_report = flowsheet.report()
stream_report
