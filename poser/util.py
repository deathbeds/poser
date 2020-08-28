import builtins
import functools
import importlib
import sys
import typing

import toolz

import operator


def attribute(property, *args, **kwargs):
    """Methodcaller, attrgetter logic."""
    def attribute(object):
        """Return an attrgetter or methodcaller."""
        object = getattr(object, property)
        return object(*args, **kwargs) if args or kwargs or callable(object) else object

    return attribute


def fold(callable, *args, **kwargs):
    @functools.wraps(callable)
    def call(*a, **k):
        return callable(*a, *args, **{**kwargs, **k})

    return call


def _sympy(object):
    """is the object a sympy expression?"""
    import sys

    if "sympy" not in sys.modules:
        return False
    import sympy

    if isinstance(object, sympy.Expr):
        return True
    return False


def map(callable, object, key=None):
    """A map function that works on mappings and sequences."""
    property = builtins.map
    if isinstance(object, typing.Mapping):
        if key is not None:
            object = getattr(toolz, f"key{property.__name__}")(key, object)
        return getattr(toolz, f"val{property.__name__}")(callable, object)
    return getattr(toolz, property.__name__)(callable, object)


def filter(callable, object, key=None):
    """A filter function that works on mappings and sequences."""
    property = builtins.filter
    if isinstance(object, typing.Mapping):
        if key is not None:
            object = getattr(toolz, f"key{property.__name__}")(key, object)
        return getattr(toolz, f"val{property.__name__}")(callable, object)
    return getattr(toolz, property.__name__)(callable, object)


def istype(object, cls: typing.Any) -> bool:
    """A convenience function for checking if an object is a type."""
    return isinstance(object, type) and issubclass(object, cls)


def normal_slice(slice: slice) -> bool:
    """is the object a conventional slice item with integer values."""
    return all(
        isinstance(x, (int, type(None)))
        for x in operator.attrgetter(*"start stop step".split())(slice)
    )


def _evaluate(object, property=None):
    """Take a dotted string and return the object it is referencing.

Used by the Forward types."""
    try:
        object = importlib.import_module(object)
        if property is None:
            return object
    except ModuleNotFoundError:
        module, _, next = object.rpartition(".")
        property = next if property is None else f"{next}.{property}"
    else:
        return operator.attrgetter(property)(object)
    return _evaluate(module, property)
