# coding: utf-8

try:
    from .model import Callable, CallableFactory
    from .recipes import juxt, compose
except:
    from model import Callable, CallableFactory
    from recipes import juxt, compose

from collections import OrderedDict
from traitlets import Any, List, Tuple, validate, Set, Callable as Callable_
from toolz.curried import concatv, identity, partial


class SequenceCallable(Callable):
    """Apply function composition to List objects.
    """
    funcs = Any(tuple())
    generator = Callable_(juxt)

    @property
    def coerce(self):
        return getattr(self.traits()['funcs'], 'klass', identity)

    @property
    def compose(self):
        return super(SequenceCallable,
                     self).compose(self.generator(self.funcs or [identity]))

    def append(self, item):
        self.set_trait('funcs', self.coerce(concatv(self.funcs, (item, ))))
        return self


class ListCallable(SequenceCallable):
    """Apply function composition to List objects.
    """
    funcs = List(list())

    @property
    def compose(self):
        composition = super(ListCallable, self).compose
        return compose(self.coerce, composition)


class TupleCallable(ListCallable):
    """Apply function composition to Tuple objects.
    """
    funcs = Tuple(tuple())


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
        self.set_trait('funcs', self.funcs)
        return self


_sequence_ = CallableFactory(funcs=SequenceCallable)
_l = _list_ = CallableFactory(funcs=ListCallable)
_t = _tuple_ = CallableFactory(funcs=TupleCallable)
_s = _set_ = CallableFactory(funcs=SetCallable)


_l[:][range][type].compose

# __*fin*__
