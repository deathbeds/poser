# coding: utf-8

# %reload_ext autoreload
# %autoreload 2


try:
    from .model import Callable, CallableFactory
    from .recipes import juxt
except:
    from model import Callable, CallableFactory
    from recipes import juxt

from traitlets import Any, List, Tuple, validate
import traitlets
from toolz.curried import compose, concatv, identity


class SequenceCallable(Callable):
    """Apply function composition to List objects.
    """
    funcs = Any(tuple())
    generator = traitlets.Callable(juxt)

    @property
    def coerce(self):
        return getattr(self.traits()['funcs'], 'klass', identity)

    @property
    def compose(self):
        return super(SequenceCallable,
                     self).compose(self.generator(self.funcs))

    def append(self, item):
        self.funcs = self.coerce(concatv(self.funcs, (item, )))
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


_sequence_ = CallableFactory(funcs=SequenceCallable)
_l = _list_ = CallableFactory(funcs=ListCallable)
_t = _tuple_ = CallableFactory(funcs=TupleCallable)

# __*fin*__
