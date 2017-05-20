# coding: utf-8

# > `fidget` uses the python data model to compose higher-order functions.
# 
# ---

try:
    from .base import State
    from .callables import call, functor
except:
    from base import State
    from callables import call, functor

from copy import copy
from collections import OrderedDict
from toolz.curried import compose, first, isiterable, partial, map, merge
from six import iteritems, PY3


class Append(State):
    __slots__ = ('function', )

    def __init__(self, function=None, *args):
        if function is None:
            function = list()

        if not isiterable(function) or isinstance(function, (str, )):
            function = [function]

        super(Append, self).__init__(function, *args)

    def __getitem__(self, object=slice(None)):
        if isinstance(object, Append) and object._factory_:
            object = object()

        if object is call:
            return abs(self)

        if isinstance(object, call):
            return object(self)()

        return object != slice(None) and self.append(object) or self

    def __repr__(self):
        return repr(self.function)

    def append(self, object):
        self.function.append(object)


class Functions(Append):
    def __contains__(self, object):
        return any(object == function for function in self)

    def __delitem__(self, object):
        self.function = list(fn for fn in self if fn != object)
        return self

    def __setitem__(self, attr, object):
        self.function = list(object if fn == attr else fn for fn in self)
        return self

    def __iter__(self):
        for function in self.function:
            yield function

    def __reversed__(self):
        self.function = type(self.function)(reversed(self.function))
        return self


class Composite(Functions):
    def _dispatch_(self, function):
        return isinstance(function, (dict, set, list, tuple)) and Juxtapose(
            function, type(function)) or functor(function)


class Juxtapose(Composite):
    __slots__ = ('function', 'type')

    def __init__(self, function, type_=None):
        if isinstance(function, dict):
            type_ = type(function)
            function = compose(tuple, iteritems)(function)
        super(Juxtapose, self).__init__(function, type_)

    def __call__(self, *args, **kwargs):
        return self.type(
            call(*args)(self._dispatch_(function))(**kwargs)
            for function in self)


class Compose(Composite):
    def __call__(self, *args, **kwargs):
        for function in self:
            args, kwargs = (call(*args)(self._dispatch_(function))(
                **kwargs), ), {}
        return first(args)


class Partial(Functions):
    __slots__ = ('args', 'keywords', 'function')
    _decorate_, _composite_ = map(staticmethod, (functor, Compose))

    def __init__(self, *args, **kwargs):
        function = kwargs.pop('function', self._composite_())
        if not callable(function):
            function = self._composite_(function)
        super(Partial, self).__init__(args, kwargs, function)

    @property
    def __call__(self):
        return call(*self.args,
                    **self.keywords)(self._decorate_(self.function))

    def append(self, object):
        self.function.function.append(object)


class Composer(Partial):
    attributes = OrderedDict()

    @property
    def _factory_(self):
        return type(self).__name__.startswith('_') and type(
            self).__name__.endswith('_')

    def __getitem__(self, object=slice(None), *args, **kwargs):
        if self._factory_:
            self = self.function()

        if isinstance(object, slice):
            object, self = self.function.function[object], copy(self)
            self.function = self._composite_(object)
            return self

        return super(Composer, self).__getitem__(
            (args or kwargs) and call(*args, **kwargs)(object) or object)


class Attributes(object):
    namespaces = OrderedDict({'custom': {}})

    def __getattr__(self, attr):
        for ns in self.namespaces.values():
            if attr in ns:
                function = partial(self.__getitem__, ns[attr])
                return PY3 and setattr(function, '__doc__',
                                       getattr(ns[attr], '__doc__',
                                               "")) or function
        raise AttributeError("No attribute {}".format(attr))

    def __dir__(self):
        return list(super(Attributes, self).__dir__()) + list(
            merge(self.namespaces.values()).keys())
