from .model import *
from .data import *
from .nn import *
from .utils import *
from .config import *

from importlib.metadata import PackageNotFoundError, version as _pkg_version
try:
    __version__ = _pkg_version('scmidas')
except PackageNotFoundError:
    __version__ = '0.0.0+unknown'
del PackageNotFoundError, _pkg_version
