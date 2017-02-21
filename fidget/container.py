# coding: utf-8

try:
    from .model import (Callable, CallableFactory)
    from .recipes import functor, raises, juxt
except:
    from model import (Callable, CallableFactory)
    from recipes import functor, raises, juxt

from collections import OrderedDict
from traitlets import Any, Dict, validate

from six import iteritems
from toolz.curried import excepts, first, filter, compose, partial


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


_d = _dict_ = CallableFactory(funcs=DictCallable)
_f = _condition_ = CallableFactory(funcs=ConditionCallable)

# __*fin*__
