import uuid

from elphick.geomet.base import MassComposition


class Stream(MassComposition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nodes = [uuid.uuid4(), uuid.uuid4()]

    def set_parent_node(self, parent: 'Stream') -> 'Stream':
        self.nodes = [parent.nodes[1], self.nodes[1]]
        return self

    def set_child_node(self, child: 'Stream') -> 'Stream':
        self.nodes = [self.nodes[0], child.nodes[0]]
        return self

    def set_nodes(self, nodes: list) -> 'Stream':
        if len(nodes) != 2:
            raise ValueError('Nodes must be a list of length 2')
        if nodes[0] == nodes[1]:
            raise ValueError('Nodes must be different')
        self.nodes = nodes
        return self

    # @classmethod
    # def from_mass_composition(cls, obj: MassComposition) -> 'Stream':
    #     filtered_kwargs = filter_kwargs(obj, **obj.__dict__)
    #     filtered_kwargs['data'] = obj.data
    #     stream = cls(**filtered_kwargs)
    #     stream.__class__ = type(obj.__class__.__name__, (obj.__class__, cls), {})
    #     return stream

    @classmethod
    def from_dict(cls, config: dict) -> 'Stream':
        name = config.get('name')
        node_in = config.get('node_in')
        node_out = config.get('node_out')
        return cls(name=name).set_nodes([node_in, node_out])