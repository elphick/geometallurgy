Project Scope
==============

Context
-------

Geoscientific disciples, like Metallurgy, Geometallurgy, Geology, and Mining Engineering, rely on the analysis of
data based on mass, moisture and chemistry.  The data is collected from drill-holes, samples, and process streams.
The data is used to model the behaviour of the material in the ground, and the material as it is processed.

Purpose
---------

To provide a package that supports the geometallurgical workflow from drill-hole data to sample fractionation
and mass balanced process simulation.  The package should be able to handle large datasets and provide the
necessary visualisations to support the workflow.  Plots should be interactive to maximise context and insight.
Assurance of data integrity is a key requirement.

Output
------

The package should be developed in a test-driven manner, with tests written in pytest.

The package provides an api that supports the following requirements:

Sample Object
~~~~~~~~~~~~~

- the fundamental object is a `Sample` object containing mass (wet, dry, h2o) and assay data
- the `Sample` object is created from a `pandas.DataFrame` object, and underlying data is stored as a `pandas.DataFrame`
- the records in a `Sample` object can represent:

  - time-series samples
  - drill-hole data
  - a sample fraction (e.g. a sieve size fraction)
  - block in a block model

- mass-weighted math operations on `Sample` objects
- `Sample` objects can represent drill-hole data, sample fractions, or process streams
- `Sample` objects can be combined to form composite samples
- `Sample` objects can be split by the following:

  - mass
  - partition model
  - machine learning model

- the mass-moisture of a `Sample` must always balance
- moisture is always calculated on a wet basis
- the chemistry of a `Sample` is always based on the dry mass
- the concrete data of a sample will be in mass units to simplify math operations
- the `Sample` object will have a `name` attribute to identify the sample
- when math operations on `Sample` objects, the relationships are preserved using hidden src_node and dst_node
  attributes.  This allows conversion to a flowsheet object without mapping the relationships again.
- an `IntervalSample` object is a subclass of `Sample` that represents a sample with an interval index.  It is used
  to represent a drill-hole intervals, or samples fractionated by size (sieved samples), etc.

Stream and Flowsheet Objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `Stream` objects represent a `Sample` assigned to the edge of a Directional Acyclic Graph (DAG) a.k.a a Flowsheet
- `Stream` is a subclass of `Sample` with additional attributes for the `src_node` and `dst_node`
- nodes in the `Flowsheet` are (unit) `Operation` objects that report the mass balance status across that node.
- a special `Stream` object is the `WaterStream` object that represents a water ony flow in a flowsheet.
  It has no chemistry.  It is a subclass of `Stream`.
- flowsheet visualisations include network and sankey plots, with tabular summaries of mass and chemistry for each
  stream
- and empty `Stream` is a `Stream` object with no data, but with a name.  It is used to represent a stream that is
  expected to have data, but does not yet.
- the `solve` method on a `Node` object will back-calculate any empty streams.

BlockModel Object
~~~~~~~~~~~~~~~~~

- subclasses Sample.  Requires a pd.MultiIndex with x, y, z.
- provides 3D plotting of the block model by leveraging the pyvista package.

Operation Object
~~~~~~~~~~~~~~~~

- `Operation` objects are nodes in a `Flowsheet` object
- `Operation` objects have a `name` attribute
- `Operation` objects have a `solve` method that back-calculates any missing data in the input streams
- `Operation` objects have a `summary` method that provides a tabular summary of the mass and chemistry of the input
  and output streams
- `Operation` objects have a `plot` method that provides a visualisation of the mass and chemistry of the input and
  output streams

Resources
---------

Expect the dependencies to include the following packages:

- pandas
- dask
- periodictable
- plotly
- omf
- omfvista, pyvista

Timing
------

This is a non-funded project, with no timeline.  Progress should be reasonably rapid, by re-using code from the
mass-composition package.

To Do
-----

.. todo::
   Add tests for the pandas utilities, which provide the mass-composition transforms and weight averaging

.. todo::
   Modify the composition module to be more intuitive.  For example you would expect is_element to return a bool,
   but it returns a reduced list of matches.
   Additionally, is_compositional with strict=True the returned list order may vary due to the use of sets in the
   method.  This is not ideal for testing.

.. todo::
   Cleanup the flowsheet module, locating static methods to utils where appropriate

.. todo::
   sankey_width_var - default to none but resolve to mass_dry using var_map.

.. todo::
   Create new repo open-geomet-data that contains the data for examples and case studies.