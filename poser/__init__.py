"Symbollic, extensible function composition in python."

from .poser import *
from .methodz import *
from . import poser, methodz
__all__ = poser.__all__ + methodz.__all__