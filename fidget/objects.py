# coding: utf-8

try:
    from .state import State
    from .callables import call, functor

except:
    from state import State
    from callables import call, functor

from copy import copy
from collections import OrderedDict
from functools import partial
from toolz.curried import compose, first, isiterable, merge
from six import iteritems, PY3


class Append(State):
    __slots__ = ('function', )
    namespaces = OrderedDict({'fidget': {}})

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

    def __getattr__(self, attr):
        for namespace in self.namespaces.values():
            if attr in namespace:
                callable = namespace[attr]
                doc = callable.__doc__
                if callable in type(self).__dict__.values():
                    callable = partial(callable, self)
                else:
                    callable = partial(self.__getitem__, callable)
                return PY3 and setattr(callable, '__doc__', doc) or callable
        raise AttributeError("No attribute {}".format(attr))

    def __dir__(self):
        return list(super(Append, self).__dir__()) + list(
            merge(self.namespaces.values()).keys())


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
    @staticmethod
    def _dispatch_(function):
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

    @property
    def __call__(self):
        return call(*self.args, **self.keywords)


class Composer(Partial):
    def __init__(self, *args, **kwargs):
        function = kwargs.pop('function', Compose())
        if not callable(function):
            function = Compose(function)
        super(Partial, self).__init__(args, kwargs, function)

    @property
    def _factory_(self):
        return type(self).__name__.startswith('_') and type(
            self).__name__.endswith('_')

    def __getitem__(self, object=slice(None), *args, **kwargs):
        if self._factory_:
            self = self.function()

        if isinstance(object, slice):
            object, self = self.function.function[object], copy(self)
            self.function = Compose(object)
            return self

        return super(Composer, self).__getitem__(
            (args or kwargs) and call(*args, **kwargs)(object) or object)

    def append(self, object):
        self.function.function.append(object)


class Calls(Composer):
    _decorate_ = staticmethod(functor)

    @property
    def __call__(self):
        return super(Calls, self).__call__(self._decorate_(self.function))


@property
def doc(self):
    string = ""
    for function in self:
        string += getattr(function, 'func', function).__doc__ or "" + "\n---\n"
    return string


for klass in (Calls, Partial, Compose, Juxtapose):
    klass.__doc__ = doc
