# coding: utf-8

try:
    from .recipes import item_to_args
except:
    from recipes import item_to_args
from collections import Sequence
from traitlets import HasTraits, Tuple, Dict
import traitlets
from toolz.curried import identity, merge, compose, partial, flip, isiterable
from inspect import isgenerator


class CallableSugar:
    def __pow__(self, value):
        args, kwargs = item_to_args(value)
        if args:
            self.set_trait('args', args)
        if kwargs:
            self.set_trait('kwargs', merge(self.kwargs, kwargs))
        return self

    def __rshift__(self, value):
        if value is compose:
            return self.compose
        if value is identity:
            return self()
        return self[value]


class Base(HasTraits):
    args = Tuple(tuple())
    kwargs = Dict(dict())
    flip = False

    @property
    def __getstate__(self):
        return self.compose.__getstate__

    @property
    def __setstate__(self):
        return self.compose.__setstate__

    def __repr__(self):
        if self.args or self.kwargs:
            return repr(self())
        return repr({
            'args': self.args,
            'kwargs': self.kwargs,
            'funcs': self.funcs
        })


class Callable(CallableSugar, Base):
    def compose(self, func):
        """Returns a pickleable callable of the current state.
        """
        if self.flip:
            func = flip(func)
        if self.args or self.kwargs:
            return partial(func, *self.args, **self.kwargs)
        return func

    @property
    def _(self):
        """Shorthand, literate syntax.
        """
        return self.compose

    def __call__(self, *args, **kwargs):
        """Call a higher order function
        """
        return self.compose(*args, **kwargs)

    def __getitem__(self, item=None):
        if item is None or item == slice(None):
            return self
        return self.append(item)


class CallableFactory(Callable):
    funcs = traitlets.Callable()

    def __call__(self, *args, **kwargs):
        return self.funcs(args=args, kwargs=kwargs)

    @property
    def coerce(self):
        return getattr(self.__class__.funcs, 'klass', identity)

    def __getitem__(self, item=slice(None)):
        if isinstance(self.funcs, type) and not issubclass(self.funcs,
                                                           HasTraits):
            return self.funcs(item)
        if isinstance(item, dict):
            return self.funcs(funcs=item)

        funcs = self.funcs()
        if item == slice(None):
            return funcs
        if isgenerator(item):
            item = self.coerce(item)
        if not isiterable(item):
            item = (item, )
        for i in item:
            funcs.append(i)
        return funcs

    def __rshift__(self, value):
        return self.funcs()[value]

    def __pow__(self, value):
        return self.funcs()**value

    def __and__(self, value):
        return self.funcs()[value]

# __*fin*__
