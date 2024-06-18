Functionality
=============

In part, this page is used to document the planned functionality of the package.  It is also used to document the
progress of the package development.

The package provides an api that supports the following requirements:

Sample Object
-------------

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
----------------------------

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
-----------------

- subclasses Sample.  Requires a pd.MultiIndex with x, y, z.
- provides 3D plotting of the block model by leveraging the pyvista package.

Operation Object
----------------

- `Operation` objects are nodes in a `Flowsheet` object
- `Operation` objects have a `name` attribute
- `Operation` objects have a `solve` method that back-calculates any missing data in the input streams
- `Operation` objects have a `summary` method that provides a tabular summary of the mass and chemistry of the input
  and output streams
- `Operation` objects have a `plot` method that provides a visualisation of the mass and chemistry of the input and
  output streams
