import logging
from collections.abc import Callable, Iterable
from typing import Optional, Union

import networkx as nx

from elphick.geomet.base import MassComposition
from elphick.geomet.flowsheet.stream import Stream

logger = logging.getLogger(__name__)


class Node:
    def __init__(self, operation: Callable, dependencies: list[str], kwargs: Optional[dict] = None):
        self.operation = operation
        self.dependencies = dependencies
        self.kwargs = kwargs if kwargs is not None else {}


class DAG:
    """A directed acyclic graph for composing mass-composition operations."""

    def __init__(self, name: str = 'DAG', n_jobs: int = -1):
        self.name = name
        self.stream_parent_node: dict[str, str] = {}
        self.n_jobs = n_jobs
        self.graph = nx.DiGraph(name=name)
        self.node_executed: dict[str, bool] = {}

    @property
    def streams(self) -> dict[str, Stream]:
        """Return streams produced on the graph's edges, keyed by stream name."""
        streams = {}
        for _, _, data in self.graph.edges(data=True):
            stream = data.get('mc')
            if stream is not None:
                streams[stream.name] = stream
        return streams

    @property
    def all_nodes_(self) -> list[str]:
        """Return the names of all graph nodes."""
        return list(self.graph.nodes)

    @staticmethod
    def input(stream: Stream) -> Stream:
        return stream

    @staticmethod
    def output(stream: Stream) -> Stream:
        return stream

    def add_input(self, name: str) -> 'DAG':
        if name in self.graph:
            raise ValueError(f"A step with the name '{name}' already exists.")
        self.graph.add_node(name, operation=self.input, kwargs=None, defined=True, name=name,
                            strm=None, dependencies=[])
        self.stream_parent_node[name] = name
        return self

    def add_step(self, name: str, operation: Callable, streams: list[str], kwargs: Optional[dict] = None,
                 defined: bool = True) -> 'DAG':
        if name in self.graph:
            raise ValueError(f"A step with the name '{name}' already exists.")
        if not callable(operation):
            raise TypeError("operation must be callable.")
        if not streams:
            raise ValueError("A step must depend on at least one stream.")
        if len(streams) != len(set(streams)):
            raise ValueError("A step cannot use the same stream more than once.")
        unknown_streams = [stream for stream in streams if stream not in self.stream_parent_node]
        if unknown_streams:
            raise ValueError(f"No parent node found for stream(s): {unknown_streams}")
        duplicate_streams = [stream for stream in streams if stream in self._edge_names()]
        if duplicate_streams:
            raise KeyError(f"Edge with name {duplicate_streams[0]} already exists in the graph.")
        output_names = [kwargs[key] for key in ('name', 'name_1', 'name_2')
                        if kwargs is not None and isinstance(kwargs.get(key), str)]
        if len(output_names) != len(set(output_names)):
            raise ValueError("Output stream names must be unique.")
        for stream_name in output_names:
            if stream_name in self.stream_parent_node or stream_name in self.graph:
                raise ValueError(f"Stream with the name '{stream_name}' already exists.")

        dependencies = [self.stream_parent_node[stream] for stream in streams]
        self.graph.add_node(name, operation=operation, dependencies=dependencies, kwargs=kwargs,
                            defined=defined, name=name)
        for stream, parent_node in zip(streams, dependencies):
            self.graph.add_edge(parent_node, name, name=stream, mc=None)

        if kwargs is not None:
            for stream_name in output_names:
                self.stream_parent_node[stream_name] = name
        return self

    def add_output(self, name: str, stream: str) -> 'DAG':
        if name in self.graph:
            raise ValueError(f"A step with the name '{name}' already exists.")
        parent_node = self.stream_parent_node.get(stream)
        if parent_node is None:
            raise ValueError(f"No parent node found for stream {stream}")
        if stream in self._edge_names():
            raise KeyError(f"Edge with name {stream} already exists in the graph.")
        self.graph.add_node(name, operation=self.output, dependencies=[stream], kwargs=None, defined=True, name=name)
        self.graph.add_edge(parent_node, name, name=stream, mc=None)
        return self

    def _edge_names(self) -> set[str]:
        return {data.get('name') for _, _, data in self.graph.edges(data=True)}

    def _topological_sort(self) -> list[str]:
        return list(nx.topological_sort(self.graph))

    def run(self, input_streams: dict[str, Union[Stream, MassComposition]], progress_bar: bool = True) -> None:
        """Execute all graph nodes in topological order using the supplied input streams."""
        del progress_bar
        self._finalize()
        self.node_executed = {node: False for node in self.graph.nodes}

        for node in self._topological_sort():
            _, _, updated_edges = self.execute_node(node, input_streams)
            for edge, stream in updated_edges.items():
                self.graph.edges[edge]['mc'] = stream
            self.node_executed[node] = True
            logger.debug("Executed DAG node %s", node)

    def execute_node(self, node: str, strms: dict[str, Union[Stream, MassComposition]]):
        """Execute one node and return its result and any edge values to update."""
        node_data = self.graph.nodes[node]
        operation = node_data['operation']
        kwargs = node_data.get('kwargs') or {}

        if operation == self.input:
            stream = strms[node]
            if isinstance(stream, MassComposition) and not isinstance(stream, Stream):
                stream = stream.to_stream()
            if not isinstance(stream, Stream):
                raise TypeError(f"Input '{node}' must be a MassComposition or Stream.")
            result = operation(stream)
        elif operation == self.output:
            predecessor = next(self.graph.predecessors(node))
            result = operation(self.graph.edges[predecessor, node].get('mc'))
            if result is None:
                raise ValueError(f"Input stream for output '{node}' has not been produced.")
        else:
            inputs = [self.graph.edges[edge].get('mc') for edge in self.graph.in_edges(node)]
            if any(stream is None for stream in inputs):
                raise ValueError(f"Input streams for step '{node}' have not been produced.")
            if operation == Stream.add and len(inputs) > 2:
                result = inputs[0]
                for stream in inputs[1:-1]:
                    result = operation(result, stream)
                result = operation(result, inputs[-1], **kwargs)
            else:
                result = operation(*inputs, **kwargs)

        updated_edges = {}
        successors = list(self.graph.successors(node))
        if successors and node_data.get('defined', True):
            if isinstance(result, MassComposition):
                for successor in successors:
                    updated_edges[(node, successor)] = result
            elif isinstance(result, Iterable) and not isinstance(result, (str, bytes, dict)):
                for stream in result:
                    if not isinstance(stream, MassComposition):
                        raise TypeError(f"Step '{node}' returned a value that is not a MassComposition.")
                    edge = next(
                        ((source, target) for source, target, data in self.graph.edges(data=True)
                         if data.get('name') == stream.name and source == node),
                        None
                    )
                    if edge is None:
                        raise ValueError(f"Edge not found for stream {stream.name}")
                    updated_edges[edge] = stream
            else:
                raise TypeError(f"Step '{node}' must return a MassComposition or an iterable of them.")
        return node, result, updated_edges

    def _finalize(self) -> None:
        if not self.graph:
            raise ValueError("The DAG is empty.")
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("Graph is not a Directed Acyclic Graph (DAG), so cannot be executed.")
        orphan_nodes = [node for node, degree in self.graph.degree() if degree == 0]
        if orphan_nodes:
            raise ValueError(f"Orphan nodes: {orphan_nodes} exist. Please check your configuration.")
        self._validate_unique_names()

    def _validate_unique_names(self) -> None:
        names = [data.get('name') for _, _, data in self.graph.edges(data=True)]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ValueError(f"Stream names are not unique within the DAG. Duplicates: {duplicates}")

    def plot(self):
        """Display a simple network diagram of the DAG."""
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots()
        nx.draw(self.graph, with_labels=True, ax=axes)
        return figure
