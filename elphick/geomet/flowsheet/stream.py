from typing import Union

from elphick.geomet.base import MassComposition, filter_kwargs
from elphick.geomet.block_model import BlockModel
from elphick.geomet.interval_sample import IntervalSample
from elphick.geomet.sample import Sample
from elphick.geomet.utils.sampling import random_int


class Stream(MassComposition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nodes = [self.random_int(), self.random_int()]

    def set_parent_node(self, parent: 'Stream') -> 'Stream':
        self.nodes = [parent.nodes[1], self.nodes[1]]
        return self

    def set_child_node(self, child: 'Stream') -> 'Stream':
        self.nodes = [self.nodes[0], child.nodes[0]]
        return self

    def set_nodes(self, nodes: list) -> 'Stream':
        self.nodes = nodes
        return self

    @staticmethod
    def random_int():
        import random
        return random.randint(0, 100)

    # @classmethod
    # def from_mass_composition(cls, obj: MassComposition) -> 'Stream':
    #     filtered_kwargs = filter_kwargs(obj, **obj.__dict__)
    #     filtered_kwargs['data'] = obj.data
    #     stream = cls(**filtered_kwargs)
    #     stream.__class__ = type(obj.__class__.__name__, (obj.__class__, cls), {})
    #     return stream
