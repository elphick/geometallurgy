from importlib import metadata

from .base import MassComposition
from .sample import Sample
from .interval_sample import IntervalSample
from .stream import Stream
from .operation import Operation
from .flowsheet import Flowsheet

try:
    __version__ = metadata.version('geomet')
except metadata.PackageNotFoundError:
    # Package is not installed
    pass
