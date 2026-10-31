YAML-Defined Flowsheets
=======================

``Flowsheet.from_yaml()`` loads stream connections and operation nodes from the
existing ``FLOWSHEET`` schema. Supply the measured feed data separately using
the configured source-stream names, then call ``solve()``. A complete example,
including two feeds, chained partitions, and recombined products, is available
in :doc:`the flowsheet gallery <../auto_examples/examples/03_flowsheet/04_yaml_partition_flowsheet>`.

Partition operations use a registered model name and keyword arguments:

.. code-block:: yaml

   type: PartitionOperation
   partition:
     model: napier_munn_size
     args:
       d50: 1.0
       ep: 0.1
     partition_stream: Coarse

The fraction-coordinate argument (``size`` in this example) is taken from the
input ``IntervalSample`` dimensions. Other required parameters are supplied in
``args``. The registered model must accept keyword arguments named for its
fraction dimensions; built-in names include ``perfect``, ``napier_munn``,
``napier_munn_size``, and ``napier_munn_density``.

Applications can register callables before loading a circuit:

.. code-block:: python

   from elphick.geomet.utils.partition import register_partition_model

   register_partition_model("my_partition", my_partition_function)

The callable is held in an explicit in-process registry. Model names are not
interpreted as Python expressions or imported modules. The older explicit
``module`` / ``function`` configuration is also supported for trusted
configurations; because it imports the named module, do not use it with
untrusted YAML. Do not combine that form with ``model``.

Use ``set_stream_data({"Feed A": feed_a, "Feed B": feed_b})`` to bind data.
Names must match configured stream names; unknown names are rejected and
``solve()`` reports missing required feeds. Feeds used by partition operations
must be ``IntervalSample`` objects. Streams combined at an operation must have
identical fraction indexes and dimension names. The solver does not resample
incompatible grids.

The solver handles acyclic flowsheets, mixing populated inputs, one missing
input or output at a time, chained partitions, and product recombination. It
does not guess splits when multiple ordinary-operation outputs are unknown.
Recycle loops and iterative convergence are not supported. Replacing feed data
invalidates streams previously derived by the solver before the next solve.

YAML-driven simulation is performed by ``Flowsheet.solve()`` using the
configured graph and supplied stream data. It does not construct or require a
separate DAG object. The optional Python-callable ``DAG.run()`` style is a
different interface for programmatically defined workflows.
