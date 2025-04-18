from __future__ import annotations
from importlib import metadata

from .sample import Sample
from .interval_sample import IntervalSample

try:
    __version__ = metadata.version('geometallurgy')
except metadata.PackageNotFoundError:
    # Package is not installed
    pass
