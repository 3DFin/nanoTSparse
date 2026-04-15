from . import _nanots

from . import backends

from .operators import *
from .tensor import *
from .version import __version__

backends.init()
