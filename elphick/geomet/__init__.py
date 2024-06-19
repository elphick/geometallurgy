from __future__ import annotations
from importlib import metadata

from .base import MassComposition
from .sample import Sample
from .interval_sample import IntervalSample
from .stream import Stream
from .operation import Operation
from .flowsheet import Flowsheet

try:
    __version__ = metadata.version('geometallurgy')
except metadata.PackageNotFoundError:
    # Package is not installed
    pass
