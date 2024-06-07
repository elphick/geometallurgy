import logging
import webbrowser
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.cm as cm
import seaborn as sns
from networkx import cytoscape_data

from plotly.subplots import make_subplots

from elphick.geomet import Stream, Sample


class Flowsheet:
    def __init__(self, name: str = 'Flowsheet'):
        self.name: str = name
        self.graph: nx.DiGraph = nx.DiGraph()
        self._logger: logging.Logger = logging.getLogger(__class__.__name__)

    @classmethod
    def from_streams(cls, streams: List[Union[Stream, Sample]],
                     name: Optional[str] = 'Flowsheet') -> 'Flowsheet':
        """Instantiate from a list of objects

        Args:
            streams: List of MassComposition objects
            name: name of the network

        Returns:

        """

        streams: List[Union[Stream, Sample]] = cls._check_indexes(streams)
        bunch_of_edges: List = []
        for stream in streams:
            if stream._nodes is None:
                raise KeyError(f'Stream {stream.name} does not have the node property set')
            nodes = stream._nodes

            # add the objects to the edges
            bunch_of_edges.append((nodes[0], nodes[1], {'mc': stream}))

        graph = nx.DiGraph(name=name)
        graph.add_edges_from(bunch_of_edges)
        d_node_objects: Dict = {}
        for node in graph.nodes:
            d_node_objects[node] = MCNode(node_id=int(node))
        nx.set_node_attributes(graph, d_node_objects, 'mc')

        for node in graph.nodes:
            d_node_objects[node].inputs = [graph.get_edge_data(e[0], e[1])['mc'] for e in graph.in_edges(node)]
            d_node_objects[node].outputs = [graph.get_edge_data(e[0], e[1])['mc'] for e in graph.out_edges(node)]

        graph = nx.convert_node_labels_to_integers(graph)
        # update the temporary nodes on the mc object property to match the renumbered integers
        for node1, node2, data in graph.edges(data=True):
            data['mc'].nodes = [node1, node2]
        obj = cls()
        obj.graph = graph
        return obj

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame,
                       name: Optional[str] = 'Flowsheet',
                       mc_name_col: Optional[str] = None,
                       n_jobs: int = 1) -> 'Flowsheet':
        """Instantiate from a DataFrame

        Args:
            df: The DataFrame
            name: name of the network
            mc_name_col: The column specified contains the names of objects to create.
              If None the DataFrame is assumed to be wide and the mc objects will be extracted from column prefixes.
            n_jobs: The number of parallel jobs to run.  If -1, will use all available cores.

        Returns:
            Flowsheet: An instance of the Flowsheet class initialized from the provided DataFrame.

        """
        streams: Dict[Union[int, str], Sample] = streams_from_dataframe(df=df, mc_name_col=mc_name_col,
                                                                        n_jobs=n_jobs)
        return cls().from_streams(streams=list(streams.values()), name=name)
