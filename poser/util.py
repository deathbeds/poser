import builtins
import functools
import importlib
import sys
import typing

import toolz

import operator


def setterattr(callable, name, object):
    setattr(object, name, callable(object))
    return object


def setteritem(f, name, object):
    if callable(name):
        name = name(object)
    object.__setitem__(name, f(object))
    return object


class Ø(BaseException):
    def __bool__(self):
        return False


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

    except ValueError:
        return Ø(F"can't load module")

    else:
        return operator.attrgetter(property)(object)

    return _evaluate(module, property)


def flatten(object):
    return toolz.concatv(*object)


def context(exit, enter, **kwargs):
    if not callable(enter):
        if isinstance(enter, str):
            try:
                enter = __import__("fsspec").open(enter, **kwargs)
            except:
                enter = open(enter, **kwargs)
    with enter as object:
        object = exit(object)
    return object


def dump(target, object, format=None, **options):
    try:
        __import__('anyconfig').dump(
            object, target, ac_parser=format, **options)
    except:
        __import__('json').dump(
            object, target, ac_parser=format, **options)
    return object


async def arunner(object):
    return await functools.partial()


async def _runner(*args, **kwargs):
    return await object(*args, **kwargs)


def join(left, right, _key, key_=None):
    return toolz.join(_key, left, right, key_ or _key, right)


def raises(object, exception, msg=None):
    msg = msg(object) if msg else object
    raise exception(msg)


def glob(object):
    import glob
    return glob.glob(object, recursive='**' in object)
