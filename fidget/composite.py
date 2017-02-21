# coding: utf-8

try:
    from .sequence import _list_, _tuple_, SequenceCallable, _sequence_, _set_
    from .container import _dict_, _condition_
    from .model import CallableFactory
    from .recipes import item_to_args, functor, compose_slice, juxt
except:
    from sequence import _list_, _tuple_, SequenceCallable, _sequence_, _set_
    from container import _dict_, _condition_
    from model import CallableFactory
    from recipes import item_to_args, functor, compose_slice, juxt

from toolz.functoolz import Compose
from toolz.curried import (compose, map, complement, reduce, groupby, do,
                           excepts, filter, flip, identity, get, first, second,
                           concatv, partial)
from collections import Iterable, OrderedDict
import traitlets
from inspect import isgenerator


def _check_types(types, value):
    return isinstance(value, types)


dispatcher = _condition_[OrderedDict([[
    partial(_check_types, (str, int, float)), get
], [partial(_check_types, set), _set_.__getitem__], [
    partial(_check_types, slice), compose_slice
], [partial(_check_types, list), _list_.__getitem__], [
    partial(_check_types, dict), _dict_.__getitem__
], [partial(_check_types, tuple), _tuple_.__getitem__
    ], [isgenerator, _sequence_.__getitem__], [identity, identity]])].compose


class CompositeAttributes:
    def pipe(self, value):
        return self[value]

    def do(self, value):
        return self.pipe(do(dispatcher(value)))

    def filter(self, value):
        return self.pipe(filter(dispatcher(value)))

    def filterfalse(self, value):
        return self.pipe(filter(complement(dispatcher(value))))

    def groupby(self, value):
        return self.pipe(groupby(dispatcher(value)))

    def identity(self, value):
        return self.pipe(value)()

    def map(self, value):
        return self.pipe(map(dispatcher(value)))

    def reduce(self, value):
        return self.pipe(reduce(dispatcher(value)))

    def excepts(self, value):
        return self.__class__(
            args=self.args, kwargs=self.kwargs)[excepts(
                Exception, self.compose, handler=functor(value))]


class CompositeSugarMixin(CompositeAttributes):
    def __mul__(self, value):
        return self.map(value)

    def __truediv__(self, value):
        return self.filter(value)

    def __floordiv__(self, value):
        return self.reduce(value)

    def __mod__(self, value):
        return self.filterfalse(value)

    def __matmul__(self, value):
        return self.groupby(value)

    def __lshift__(self, value):
        return self.do(value)

    def __or__(self, value):
        return self.excepts(value)


class Composite(CompositeSugarMixin, SequenceCallable):
    funcs = traitlets.List(list())
    generator = traitlets.Callable(
        compose(Compose, list, map(dispatcher), reversed))

    def append(self, item):
        self.set_trait('funcs', list(concatv(self.funcs, (item, ))))
        return self


class Stars(Composite):
    def __call__(self, item=[]):
        args, kwargs = item_to_args(item)
        return super(Stars, self).__call__(*args, **kwargs)


class FlipComposite(Composite):
    flip = True


_x = _comp_ = CallableFactory(funcs=Composite)
x_ = _pmoc_ = CallableFactory(funcs=FlipComposite)
stars = CallableFactory(funcs=Stars)

# __*fin*__
