import builtins
from functools import wraps
import typing
import toolz


class Ø(BaseException):
    def __bool__(self):
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


def flip(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if args:
            a, *args = args
            if args:
                b, *args = args
                return f(b, a, *args, **kwargs)
            return f(a, **kwargs)
        return f()

    return wrap


def I(*tuple, **_):
    "A nothing special identity function, does pep8 peph8 me?"
    return tuple[0] if tuple else None


def istype(object, cls):
    """A convenience function for checking if an object is a type."""
    return isinstance(object, type) and issubclass(object, cls)


def is_normal_slice(slice):
    return all(
        isinstance(x, (int, type(None)))
        for x in operator.attrgetter(*"start stop step".split())(slice)
    )
