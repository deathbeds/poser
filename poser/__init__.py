"""dysfunctional programming in python"""
__version__ = "0.2.3"
__all__ = "λ", "Λ", "poser", "this", "star", "juxt"
import toolz

from . import util
from .poser import *


class juxt(toolz.functoolz.juxt):
    """An overloaded toolz juxtaposition that works with different objects and iterables."""

    _lambdaified = {}

    def __new__(self, funcs=None):
        if funcs is None:
            self = super().__new__(self)
            return self.__init__() or self
        if isinstance(funcs, str):
            funcs = Forward(funcs)
        if _sympy(funcs):
            import sympy

            if not funcs in self._lambdaified:
                self._lambdaified[funcs] = sympy.lambdify(
                    sorted(funcs.free_symbols, key=lambda x: x.name), funcs
                )
            return self._lambdaified[funcs]
        if callable(funcs) or not toolz.isiterable(funcs):
            return funcs
        self = super().__new__(self)
        return self.__init__(funcs) or self

    def __init__(self, object=None):
        self.funcs = object

    def __call__(self, *args, **kwargs):
        if isinstance(self.funcs, typing.Mapping):
            # Juxtapose a mapping object.
            object = type(self.funcs)()
            for key, value in self.funcs.items():
                if callable(key):
                    key = juxt(key)(*args, **kwargs)
                if callable(value):
                    value = juxt(value)(*args, **kwargs)
                object[key] = value
            return object
        if toolz.isiterable(self.funcs):
            # juxtapose an iterable type that returns the container type
            return type(self.funcs)(
                juxt(x)(*args, **kwargs)
                if (callable(x) or toolz.isiterable(x) or _sympy(x))
                else x
                for x in self.funcs
            )
        if callable(self.funcs):
            # call it ya can
            return self.funcs(*args, **kwargs)
        return self.funcs
