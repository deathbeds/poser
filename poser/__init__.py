"Symbollic, extensible function composition in python."

from . import poser, methodz
__all__ = poser.__all__ + methodz.__all__

from .poser import *
from .methodz import *
