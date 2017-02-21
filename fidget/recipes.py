# coding: utf-8

from toolz.curried import (isiterable, take, drop, take_nth, compose, filter,
                           interpose, identity, flip, concat, map, pipe, first,
                           excepts)
from toolz.functoolz import juxt as tlz_juxt
from six import iteritems
from toolz.curried.operator import attrgetter
from toolz.functoolz import Compose


class functor(tlz_juxt):
    def __init__(self, value):
        self.funcs = value

    def __call__(self, *args, **kwargs):
        if callable(self.funcs):
            return self.funcs(*args, **kwargs)
        return self.funcs


def star_arguments(args, kwargs, function):
    return function(*args, **kwargs)


def item_to_args(obj):
    """heuristics for converting an object to args & kwargs."""
    args, kwargs = tuple(), dict()
    if bool(obj):
        if isinstance(obj, dict):
            kwargs = obj
        elif isiterable(obj):
            args = tuple(obj)
        else:
            args = (obj, )
    return args, kwargs


def compose_slice(slice):
    """compose functions that parity a slice.
    """
    sliced = []
    if slice.stop:
        sliced.append(take(slice.stop))
    if slice.start:
        sliced.append(drop(slice.start))
    if slice.step:
        sliced.append(take_nth(slice.step))
    return Compose(sliced or identity)


def raises(e):
    raise e


class juxt(tlz_juxt):
    def __init__(self, *funcs, **kwargs):
        self.excepts = kwargs.pop('excepts', None)
        if isinstance(first(funcs), dict) and len(funcs) == 1:
            funcs = [iteritems(funcs)]
        super(juxt, self).__init__(*funcs)

    def __call__(self, *args, **kwargs):
        for func in map(functor, self.funcs):
            yield excepts(
                Exception,
                func,
                handler=functor(self.excepts)
                if self.excepts is not None else raises)(*args, **kwargs)


def flip(func, *args):
    return func(*reversed(args))


# def docify(klass, *args):
#     return pipe(
#         [args, type(klass).__mro__],
#         concat, filter(flip(hasattr)('__doc__')),
#         filter(attrgetter('__doc__')),
#         map(attrgetter('__name__', '__doc__')),
#         map(interpose(' - ')),
#         map(' '.join),
#         '\n'.join,
#     )

# __*fin*__
