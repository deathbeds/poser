# coding: utf-8

try:
    from .recipes import juxt
except:
    from recipes import juxt

from toolz.functoolz import Compose

from toolz.curried.operator import attrgetter
from toolz.curried import first, last, compose, concatv, merge


class Chain(object):
    def __init__(self, obj=compose(type, list, range), *args, **kwargs):
        self.funcs = [obj]
        self.args = args
        self.kwargs = kwargs

    def __getattr__(self, attr):
        if callable(last(self.funcs)) or len(self.funcs) == 1:
            self.funcs.append([])
        last(self.funcs).append(attrgetter(attr))
        return self

    def __call__(self, *args, **kwargs):
        last(self.funcs).append((args, kwargs))
        self.funcs.append([])
        return self

    def compose(self, obj=None):
        if obj is None:
            obj = first(self.funcs)
        for func in filter(bool, self.funcs[1:]):
            args, kwargs = last(func)
            compose(Compose, list, reversed)(func[:-1])(obj)(
                *concatv(self.args, args), **merge(self.kwargs, kwargs))
        return obj

    @property
    def _(self):
        return self.compose()


class ChainFactory(object):
    def __getattribute__(self, attr):
        return getattr(Chain(None), attr)

    def __call__(self, obj):
        return Chain(obj)


_c = _chain_ = ChainFactory()


# import pandas

# df  = pandas.util.testing.makeDataFrame()
# _c(df).sum().count()._

# __*fin*__
