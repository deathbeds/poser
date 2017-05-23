# coding: utf-8

try:
    from .state import State
    from .callables import functor

except:
    from state import State
    from callables import functor

from copy import copy
from inspect import signature
from toolz.curried import compose, first, isiterable, partial
from six import iteritems, PY3


class Append(State):
    def __getitem__(self, object=slice(None)):
        return object != slice(None) and self.append(object) or self

    def __repr__(self):
        return repr(self.function)

    @property
    def append(self):
        return self.function.append


class Signature(object):
    @property
    def __signature__(self):
        try:
            return signature(first(self.function))
        except:
            return signature(self.__call__)


class Functions(Signature):
    __slots__ = ('function', )

    def __contains__(self, object):
        return any(object == function for function in self)

    def __delitem__(self, object):
        self.function = list(fn for fn in self if fn != object)
        return self

    def __setitem__(self, attr, object):
        self.function = list(object if fn == attr else fn for fn in self)
        return self

    def __iter__(self):
        for function in self.function or [functor()]:
            yield function

    def __reversed__(self):
        self.function = type(self.function)(reversed(self.function))
        return self


class Composition(Append, Functions):
    def __init__(self, function=list()):
        if not isiterable(function) or isinstance(function, (str, )):
            function = [function]
        super(Composition, self).__init__(copy(function))

    @staticmethod
    def _dispatch_(function):
        return isinstance(function, (dict, set, list, tuple)) and Juxtapose(
            function, type(function)) or functor(function)


class Juxtapose(Composition):
    __slots__ = ('function', 'type')

    def __init__(self, function, type_=None):
        type_ = type_ or type(function)
        if isinstance(function, dict):
            function = compose(tuple, iteritems)(function)
        super(Juxtapose, self).__init__(function, type_)

    def __call__(self, *args, **kwargs):
        return self.type(
            self._dispatch_(function)(*args, **kwargs) for function in self)


class Compose(Composition):
    def __call__(self, *args, **kwargs):
        for function in self:
            args, kwargs = (self._dispatch_(function)(*args, **kwargs), ), {}
        return first(args)


class Partial(Append, Functions):
    __slots__ = ('args', 'keywords', 'function')

    def __init__(self, *args, **kwargs):
        super(Partial, self).__init__(args, kwargs, Compose())


class Composer(Partial):
    def __getitem__(self, object=slice(None), *args, **kwargs):
        if isinstance(object, slice):
            object, self = self.function.function[object], copy(self)
            self.function = Compose(object)
            return self

        return super(Composer, self).__getitem__(
            (args or kwargs) and partial(object, *args, **kwargs) or object)

    @property
    def append(self):
        return self.function.function.append


class Calls(Composer):
    _decorate_ = staticmethod(functor)

    @property
    def __call__(self):
        return partial(self.function, *self.args, **self.keywords)


def doc(self):
    return getattr(first(self), '__doc__', '')


if PY3:
    for func in [Compose, Juxtapose, Calls]:
        setattr(func, '__doc__', property(doc))
