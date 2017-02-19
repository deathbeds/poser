# coding: utf-8

# %reload_ext autoreload
# %autoreload 2


try:
    from .model import Callables, CallableFactory
    from .recipes import juxt
except:
    from model import Callables, CallableFactory
    from recipes import juxt

from traitlets import List, Tuple, Callable, validate
from toolz.curried import compose, concatv


class SequenceCallable(Callables):
    """Apply function composition to List objects.
    """
    generator = Callable(juxt)

    @property
    def compose(self):
        return super(SequenceCallable,
                     self).compose(self.generator(self.funcs))


class ListCallable(SequenceCallable):
    """Apply function composition to List objects.
    """
    funcs = List(list())

    @property
    def coerce(self):
        return getattr(self.traits()['funcs'], 'klass', None)

    @property
    def compose(self):
        composition = super(ListCallable, self).compose
        return compose(self.coerce, composition)

    def append(self, item):
        self.funcs = self.coerce(concatv(self.funcs, (item, )))
        return self


class TupleCallable(ListCallable):
    """Apply function composition to Tuple objects.
    """
    funcs = Tuple(tuple())


_l = _list_ = CallableFactory(funcs=ListCallable)
_t = _tuple_ = CallableFactory(funcs=TupleCallable)

# __*fin*__
