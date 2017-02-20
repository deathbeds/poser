# coding: utf-8

# %reload_ext autoreload
# %autoreload 2


try:
    from .model import (Callable, CallableFactory)
    from .recipes import functor, raises, juxt
    from .sequence import ListCallable
except:
    from model import (Callable, CallableFactory)
    from recipes import functor, raises, juxt
    from sequence import ListCallable

from collections import OrderedDict
from traitlets import Any, Set, Dict, validate

from six import iteritems
from toolz.curried import excepts, first, filter, compose, partial, identity


class ContainerCallable(Callable):
    funcs = Dict()
    excepts = Any(None)

    @validate('funcs')
    def _validate_value(self, change):
        funcs = change.pop('value', OrderedDict())
        if not isinstance(funcs, OrderedDict):
            return OrderedDict(funcs)
        return funcs

    @property
    def compose(self):
        return super(ContainerCallable, self).compose(
            juxt(
                map(partial(
                    juxt, excepts=self.excepts or raises),
                    iteritems(self.funcs))))

    def append(self, value):
        if not isinstance(value, dict):
            value = OrderedDict((value, ))
        for key, value in iteritems(value):
            self.funcs[key] = value
        self.funcs = self.funcs
        return self


class DictCallable(ContainerCallable):
    """Apply function composition to Dict objects.
    """

    @property
    def compose(self):
        return compose(OrderedDict, super(DictCallable, self).compose)


class ConditionCallable(ContainerCallable):
    """Apply function composition to Dict objects.
    """

    @property
    def compose(self):
        return compose(
            excepts(StopIteration,
                    compose(first, first,
                            filter(excepts(Exception, first, functor(False)))),
                    functor(None)), super(ConditionCallable, self).compose)


class SetCallable(ListCallable):
    """Apply function composition to Set objects.
    """
    funcs = Set(set())

    @property
    def compose(self):
        return compose(OrderedDict,
                       partial(zip, list(self.funcs)),
                       super(SetCallable, self).compose)

    def append(self, item):
        self.funcs.add(item)
        self.funcs = self.funcs
        return self


_s = _set_ = CallableFactory(funcs=SetCallable)
_d = _dict_ = CallableFactory(funcs=DictCallable)
_f = _conditional_ = CallableFactory(funcs=ConditionCallable)

# __*fin*__
