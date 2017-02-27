# coding: utf-8

try:
    from .recipes import compose
except:
    from recipes import compose
from toolz.curried import do, partial, last, get, pipe, filter, concatv
from toolz.curried.operator import attrgetter, itemgetter


class ChainGenerator(object):
    """Generate This and Self compositions.
    """

    def __init__(self, func):
        self.func = func  # Union[This, Self]

    def __getattr__(self, attr):
        # (str) -> Union[This, Self]
        return getattr(self.func(), attr)

    def __getitem__(self, attr):
        # (str) -> Union[This, Self]
        return self.func()[attr]


def call(args, kwargs, obj):
    return obj(*args, **kwargs)


class This(object):
    def __init__(
            self,
            funcs=[], ):
        self.funcs = funcs.copy()
        self.funcs.append([])

    def __getattr__(self, attr):
        last(self.funcs).append(attrgetter(attr))
        return self

    def __getitem__(self, item):
        self.funcs[-1] = itemgetter(item)
        return self.__class__(self.funcs)

    def __call__(self, *args, **kwargs):
        if last(self.funcs) == []:
            return self.fn(args[0])
        self.compose_func(*args, **kwargs)
        if self.__class__ is Self:
            self.funcs[-1] = do(self.funcs[-1])
        return self.__class__(self.funcs)

    def compose_object(self):
        if isinstance(last(self.funcs), list):
            self.funcs[-1] = compose(*reversed(last(self.funcs)))

    def compose_func(self, *args, **kwargs):
        if isinstance(last(self.funcs), list):
            self.funcs[-1] = compose(*concatv(
                [partial(call, args, kwargs)],
                filter(bool, reversed(last(self.funcs)))))

    @property
    def fn(self):
        self.compose_object()
        return compose(*pipe(self.funcs, reversed, filter(bool)))


class Self(This):
    pass


this = ChainGenerator(This)
self = ChainGenerator(Self)


# import pandas as pd

# df = pd.util.testing.makeDataFrame()

# this.sum( ).index.fn( df )

# __*fin*__
