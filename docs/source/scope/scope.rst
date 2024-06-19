Project Scope
=============

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

The package provides an api that supports the following objects:

- Sample: a container for mass, moisture, and chemistry data
- Stream: a container for a Sample object that is part of a flowsheet
- Flowsheet: a container for a network of Stream objects
- BlockModel: a container for a 3D array of mass, moisture, and chemistry data
- Operation: a node in a Flowsheet object that reports the mass balance status across that node
- WaterStream: a subclass of Stream that represents a water only flow in a flowsheet
- EmptyStream: a Stream object with no data, but with a name.  It is used to represent a stream that is expected to
  have data, but does not yet.
- IntervalSample: a subclass of Sample that represents a sample with an interval index.  It is used to represent a
  drill-hole intervals, or samples fractionated by size (sieved samples), etc.
- utils: a module that provides utility functions for the package

For more information on the objects, see the functionality and api reference:

- `Functionality <functionality.html>`_
- `API Reference <../api/modules.html>`_

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

