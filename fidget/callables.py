# coding: utf-8

try:
    from .base import State
except:
    from base import State
from six import PY3
from toolz import isiterable, partial


class functor(State):
    __slots__ = ('function', )

    def __call__(self, *args, **kwargs):
        return self.function(
            *args, **kwargs) if callable(self.function) else self.function


class flipped(functor):
    def __call__(self, *args, **kwargs):
        return super(flipped, self).__call__(*reversed(args), **kwargs)


class do(functor):
    def __call__(self, *args, **kwargs):
        super(do, self).__call__(*args, **kwargs)
        return args[0] if args else None


class call(State):
    __slots__ = ('args', 'kwargs')

    def __init__(self, *args, **kwargs):
        super(call, self).__init__(args, kwargs)

    def __call__(self, function=functor):
        return partial(functor(function), *self.args, **self.kwargs)


class starred(functor):
    def __call__(self, *args, **kwargs):
        args = args[0] if len(args) is 1 else (args, )
        if not isiterable(args):
            args = [(args, )]
        if isinstance(args, dict):
            args = kwargs.update(args) or tuple()
        return super(starred, self).__call__(*args, **kwargs)


class condition(functor):
    __slots__ = ('condition', 'function')


class ifthen(condition):
    def __call__(self, *args, **kwargs):
        return functor(self.condition)(*args, **kwargs) and super(
            ifthen, self).__call__(*args, **kwargs)


class ifnot(condition):
    def __call__(self, *args, **kwargs):
        return functor(self.condition)(*args, **kwargs) or super(
            ifnot, self).__call__(*args, **kwargs)


class step(condition):
    def __call__(self, *args, **kwargs):
        result = functor(self.condition)(*args, **kwargs)
        return result and super(step, self).__call__(result)


class excepts(functor):
    __slots__ = ('exceptions', 'function')

    def __call__(self, *args, **kwargs):
        try:
            return super(excepts, self).__call__(*args, **kwargs)
        except self.exceptions as e:
            return exception(e)


class exception(State):
    __slots__ = ('exception', )

    def __bool__(self):
        return not self.exception

    def __repr__(self):
        return repr(self.exception)


def doc(self):
    return getattr(self.function, '__doc__', '')


if PY3:
    for func in (functor, flipped, do, starred, ifthen, ifnot, excepts):
        setattr(func, '__doc__', property(doc))
