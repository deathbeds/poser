# coding: utf-8

try:
    from .recipes import item_to_args, flip, compose
except:
    from recipes import item_to_args, flip, compose

from collections import Sequence
from copy import copy
from inspect import isgenerator
from toolz.curried import identity, partial, isiterable, complement
from traitlets import HasTraits, Tuple, Dict, Callable as Callable_, Bool
from pickle import dumps


class CallableSugar:
    def __pow__(self, value):
        """1st in Operator Preference.  Updates arguments & keywords
        for iterators & lists, respectively.

        # set arguments
        _xx ** ('foo', 42)

        # set keywords
        _xx ** {'foo': 42, 'bar': [0, 10]}"""
        args, kwargs = item_to_args(value)
        return self.update(*args).update(**kwargs)

    def __rshift__(self, value):
        """Append an object to the composition.
        """
        return self[value]

    def __iter__(self):
        return iter(self.compose)


class Base(HasTraits):
    """Base Traitlets Class for `fidget` Composite Functions.
    """
    args = Tuple(tuple())
    kwargs = Dict(dict())
    _complement = Bool(False)
    flip = False

    def update(self, *args, **kwargs):
        """Update the compositions args & kwargs.
        """
        if args:
            self.set_trait('args', args)
        if kwargs:
            self.set_trait('kwargs', kwargs)
        return self

    def __copy__(self, *args, **kwargs):
        return self.__class__(
            funcs=list(self.funcs),
            args=list(self.args),
            kwargs=dict(self.kwargs))

    def copy(self, *args, **kwargs):
        return copy(self).update(*args).update(**kwargs)

    def __repr__(self):
        if self.args or self.kwargs:
            return repr(self())
        return repr({
            'args': self.args,
            'kwargs': self.kwargs,
            'funcs': self.funcs
        })

    @property
    def __getstate__(self):
        return self.compose.__getstate__

    @property
    def __setstate__(self):
        return self.compose.__setstate__

    def __reversed__(self):
        funcs = self.funcs
        if not isinstance(self.funcs, Sequence):
            funcs = tuple(funcs)
        self.funcs = self.coerce(reversed(funcs))
        return self

    @property
    def _pickable(self):
        try:
            composition = dumps(self.compose)
            del composition
            return True
        except:
            return False

    def __enter__(self):
        return self.copy()

    def __exit__(self, type, value, traceback):
        return


class Callable(CallableSugar, Base):
    def compose(self, func):
        """Composition the functions in funcs and apply partial arguments
        and keywords.
        """
        if self.flip:
            func = partial(flip, func)

        if self._complement:
            func = complement(func)

        if self.args or self.kwargs:
            return partial(func, *self.args, **self.kwargs)
        return func

    @property
    def _(self):
        """Shorthand for _xx.compose.
        """
        return self.compose

    def __call__(self, *args, **kwargs):
        """Call the composition, update the the keyword arguments
        and apply the arguments.
        """
        return self.compose(*args, **kwargs)

    def __getitem__(self, item=None):
        """Append a new object to the current instance.
        """
        if item is compose:
            return self.compose
        if item is identity:
            return self()
        if item is copy:
            return copy(self)

        if item is None or item == slice(None):
            return self
        return self.append(item)


class CallableFactory(Callable):
    funcs = Callable_()

    def __call__(self, *args, **kwargs):
        return self.funcs(args=args, kwargs=kwargs)

    @property
    def coerce(self):
        """returns a callable method to enforce the `funcs` type.
        """
        return getattr(self.__class__.funcs, 'klass', identity)

    def __getitem__(self, item=slice(None)):
        """Logical to append new objects to the composition.
        """
        funcs = self.funcs()

        if item == slice(None):
            return funcs

        if isinstance(item, dict):
            return funcs.append(item)

        if isgenerator(item):
            item = self.coerce(item)

        if not isiterable(item):
            item = (item, )

        for i in item:
            funcs.append(i)

        return funcs

    def __rshift__(self, value):
        return self.funcs().pipe(value)

    def __lshift__(self, value):
        return self.funcs().do(value)

    def __pow__(self, value):
        return self.funcs().__pow__(value)

    def __copy__(self, *args, **kwargs):
        return self.funcs(args=list(self.args), kwargs=dict(self.kwargs))

# __*fin*__
