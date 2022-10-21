"""Forward references.

    >>> e = Forward('builtins.range')
    >>> e
    ForwardRef('builtins.range')
    >>> e._evaluate(), e
    (<class 'range'>, <class 'range'>)
    >>> λ['random.random']()
    0...
    >>> λ['random itertools'.split()]()
    [<module 'random'...>, <module 'itertools' (built-in)>]
    >>> λ['itertools.chain.__name__']()
    'chain'
"""

from importlib import import_module
from inspect import getdoc, signature
from operator import attrgetter
from typing import ForwardRef
from ast import parse


def forward_evaluate(object, property=None):
    """Take a dotted string and return the object it is referencing.

    Used by the Forward types."""
    try:
        object = import_module(object)
        if property is None:
            return object
    except ModuleNotFoundError:
        module, _, next = object.rpartition(".")
        property = next if property is None else f"{next}.{property}"
    else:
        return attrgetter(property)(object)
    return forward_evaluate(module, property)


class Forward(ForwardRef, _root=False):
    """A forward reference implementation that accesses object off of the sys.modules"""

    def __new__(cls, object=None, *args, **kwargs):
        if not isinstance(object, (str, type(None))):
            return object
        if isinstance(object, str):
            try:
                parse(object)
            except SyntaxError:
                return object  # if the forward reference isn't valid code...
        self = super().__new__(cls)
        if object is not None:
            self.__init__(object, *args, **kwargs)
        return self

    def __call__(self, *args, **kwargs):
        object = self._evaluate()
        return object(*args, **kwargs) if callable(object) else object

    def _evaluate(self, globalns=None, localns=None):
        self.__forward_value__ = forward_evaluate(self.__forward_arg__)
        self.__forward_evaluated__ = True
        return self.__forward_value__

    def __repr__(x):
        return repr(x._evaluate()) if x.__forward_evaluated__ else super().__repr__()

    @property
    def __signature__(x):
        return signature(x._evaluate())

    @property
    def __doc__(x):
        return getdoc(x._evaluate())

    def __str__(x):
        return x.__forward_arg__
