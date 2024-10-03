Geomet 0.4.2 (2024-10-03)
=========================

Other Tasks
-----------

- Improved management of dependencies for the blockmodel extra (#32)


Geomet 0.4.1 (2024-09-29)
=========================

Feature
-------

- Exposed weight_average method with group_by argument. (#29)


Geomet 0.4.0 (2024-09-22)
=========================

Other Tasks
-----------

- Breaking change - refactored stream inheritance. Objects now mutate to Stream with nodes property after math operations (#21)

Geomet 0.3.3 (2024-09-21)
=========================

Bugfix
------

- Fixed incorrect label on plot_network nodes (#15)


Geomet 0.3.2 (2024-09-03)
=========================

Other Tasks
-----------

- Reduced ydata-profiling = ^4.6.0 to enable pydantic < 2.0 if required. (#25)


Geomet 0.3.1 (2024-08-19)
=========================

Bugfix
------

- Fixed Flowsheet.solve missing input bug. Improved error message for Flowsheet.report when a stream is empty. (#23)


Geomet 0.3.0 (2024-08-18)
=========================

Feature
-------

- Interval sample functionality added. Standard sieve sizes added. (#13)


Geomet 0.2.1 (2024-08-17)
=========================

Feature
-------

- Added query and filter_by_index methods to Flowsheet.  Added example. (#18)


Geomet 0.2.0 (2024-08-10)
=========================

Feature
-------

- Added ability to create a Flowsheet from file, from_yaml, from_json (#14)


Geomet 0.1.1 (2024-06-19)
=========================

Other Tasks
-----------

- Cleaned up tests that are incomplete (#11)
