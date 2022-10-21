from dataclasses import dataclass
from typing import Any
from toolz import juxt, isiterable
from .forward import Forward

__all__ = ("juxt",)


@dataclass
class juxt:
    """An overloaded toolz juxtaposition that works with different objects and iterables."""

    funcs: Any = None

    def __new__(self, funcs=None):
        if funcs is None:
            self = super().__new__(self)
            self.__init__()
            return self
        if isinstance(funcs, str):
            funcs = Forward(funcs)

        if callable(funcs) or not isiterable(funcs):
            return funcs
        self = super().__new__(self)
        self.__init__(funcs)
        return self

    def __call__(self, *args, **kwargs):
        if isinstance(self.funcs, __import__("typing").Mapping):
            # Juxtapose a mapping object.
            object = type(self.funcs)()
            for key, value in self.funcs.items():
                if callable(key):
                    key = juxt(key)(*args, **kwargs)
                if callable(value):
                    value = juxt(value)(*args, **kwargs)
                object[key] = value
            return object
        if isiterable(self.funcs):
            # juxtapose an iterable type that returns the container type
            return type(self.funcs)(
                juxt(x)(*args, **kwargs) if (callable(x) or isiterable(x)) else x
                for x in self.funcs
            )
        if callable(self.funcs):
            # call it ya can
            return self.funcs(*args, **kwargs)
        return self.funcs
