# Objective

The aim is to consider the following 4 files that are from the mass-composition project
and to migrate that content to this project.  It is believed that with emerging clarity 
of use cases a better design can be achieved

## Use Cases

1. A collection of Samples or Streams, or Block Models (all MassComposition subclasses in this new package)
   resulting from math operations can be easily converted into a flowsheet object.  The flowsheet visualisation
   will show via the status that the network balances.
2. A flowsheet already defined (somehow) can have objects loaded onto edges that align with the MassComposition
   object name.  The flowsheet can then be used to calculate the mass balance of the network and report status as in 1.
3. A flowsheet can be used for simulation.  This case is managed in the legacy code by DAG.  
   To `run` or `execute` or `simulate` requires the user to provide the mc objects that are require inputs
   defined by the out-edges of input nodes on the network.  Each node had a definition of what operation/function
   to apply to the incoming node to calculate outputs.  It is likely that subclassing `Flowsheet` may make sense with that
   class being called Simulator?
4. Later - a Flowsheet can be used to balance data that does not balance.  This is a common problem in the mining
   industry where data is collected from different sources and the data does not balance.  The flowsheet can be used
   to balance the data and report the status of the balance.  This may alter the decision whether a custom node object
   is used as the actual node on the nx.graph or if the node object is placed inside the node as an attribute
   (to cater for the two states, measured and balanced).  It is expected that a MassBalance object would subclass Flowsheet?

## Considerations

1. The legacy code used xarray as the underlying data structure, though this new project simply uses pandas, which so far seems ok.
2. Rename of MCNode from the legacy code.  In the new code so far, this is called Operation.  But debating this name choice since
   s node may or may not have a math operation e.g. use case 1 and 2.