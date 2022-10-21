from abc import ABCMeta
import dataclasses
import functools
import itertools
import sys
import typing

import toolz
from typing import Any, ChainMap
from toolz.functoolz import compose
from toolz.curried import *
from dataclasses import dataclass, field
from .util import fold, map, filter, istype, normal_slice, _evaluate, attribute
from .functools import map, filter, I, Ø, flip, is_normal_slice, istype
from .juxtaposition import juxt

__all__ = "λ", "poser"


@dataclass(eq=True)
class interface:
    funcs: tuple[Any] = field(default_factory=tuple)
    arguments: tuple[Any] = field(default_factory=tuple)
    kwargs: dict[str, Any] = field(default_factory=dict)
    exceptions: tuple[BaseException] = field(default_factory=tuple)

    @classmethod
    def as_dataclass(cls, **kwargs):
        self = interface.__new__(cls)
        interface.__init__(self, **kwargs)
        return self


class base:
    @classmethod
    def append_function(cls, self, func=None, *arguments, **kwargs):
        """the primary logic used to add a function to the composition"""
        if func is not None:
            func = juxt(func)

        if isinstance(self, type):
            # if the func isn't instantiated, instantiate it.
            self = self()

        if func is None:
            # None is a valid answer
            return self
        if isinstance(func, slice):
            # slices are weird
            if is_normal_slice(func):
                # normal slices operate on a lists of functions.
                # maybe this raises an error.
                return type(self)(funcs=list(self)[func])
            else:
                # slices of callable are crazy!
                # They provide access to the common filter, map, pipe pattern.
                # slice(filter, map, pipe)
                callable(func.start) and self.filter(func.start)
                callable(func.stop) and self.map(func.stop)
                callable(func.step) and self.partial(func.step)
                return self

        # ball the func into a partial when arguments or kwargs are provided
        if arguments or kwargs:
            func = toolz.partial(func, *arguments, **kwargs)

        self.funcs += (func,)
        return self


class methods(base):
    def args(self, *arguments, **kwargs):
        if self.funcs:
            *self.funcs, method = self.funcs
            self.append_function(self, method, *arguments, **kwargs)
            return self
        self.arguments, self.kwargs = arguments, kwargs
        return self

    def complement(self, object=None):
        if object is None:
            return self.as_dataclass(funcs=(toolz.complement(self),))
        return self.append_function(self, toolz.complement(juxt(object)))

    def condition(self, object):
        if isinstance(object, (tuple, type)):
            return self.isinstance(object)
        return ifthen(self.append_function(self, object))

    def do(self, callable):
        return self.append_function(self, toolz.curried.do(juxt(callable)))

    def duplicate(self):
        return self.as_dataclass(**dataclasses.asdict(self))

    dup = duplicate

    def excepts(self, *exceptions):
        return type(self).as_dataclass(funcs=(self,), exceptions=exceptions)

    def extend(self, *funcs):
        for f in funcs:
            self.append_function(self, f)
        return self

    def filter(self, object, key=None):
        if object is not None:
            object = juxt(object)
        if key is not None:
            key = juxt(key)
        return self.append_function(self, filter, object, key=key)

    def flip(self, object=None):
        if object is None:
            return self.as_dataclass(funcs=(flip(self),))
        return self.append_function(self, toolz.flip(object))

    @functools.wraps(toolz.groupby)
    def groupby(self, callable, *arguments, **kwargs):
        return self.append_function(self, toolz.curried.groupby, juxt(callable))

    I = staticmethod(I)

    def ifnot(self, callable):
        return ifnot(self)[callable]

    def ifthen(self, callable):
        return ifthen(self).partial(callable)

    def map(self, callable, key=None):
        """Append an overloaded map that works with iterables and mappings."""
        if callable is not None:
            callable = juxt(callable)
        if key is not None:
            key = juxt(key)

        return self.append_function(self, map, callable, key=key)

    def partial(self, object, *arguments, **kwargs):
        return self.append_function(self, object, *arguments, **kwargs)

    def pipe(self, *arguments, **kwargs):
        return self(*arguments, **kwargs)

    def reduce(self, callable):
        return self.append_function(self, toolz.curried.reduce, juxt(callable))

    def __abs__(self):
        # might want to invoke excepts
        if self.funcs:
            if len(self.funcs):
                return self.funcs[0]
            return toolz.functoolz.compose_left(*self.funcs)
        return self.I

    def __reversed__(self):
        data = dataclasses.asdict(self)
        data["funcs"] = tuple(reversed(data["funcs"]))
        return type(self)(**data)

    def __iter__(self):
        yield from self.funcs

    def __bool__(self):
        return not isinstance(self, type)

    def __len__(self):
        if isinstance(self, type):
            return 0
        return len(self.funcs)

    append = __pos__ = __rshift__ = __sub__ = __rsub__ = partial
    __isub__ = __add__ = __radd__ = __iadd__ = __getitem__ = partial
    __mul__ = __rmul__ = __imul__ = map
    __truediv__ = __rtruediv__ = __itruediv__ = filter
    __matmul__ = __rmatmul__ = __imatmul__ = groupby
    __mod__ = __rmod__ = __imod__ = reduce
    __xor__ = excepts
    __and__ = ifthen
    __or__ = ifnot
    __pow__ = __ipow__ = condition
    __lshift__ = do
    __neg__ = complement
    __invert__ = flip


class attributes:
    _flipped_methods = {}
    _partial_methods = {}
    _instance_methods = dict()
    _methods = ChainMap(_partial_methods, _flipped_methods, _instance_methods)

    def __getattr__(self, name):
        method = self._partial_methods.get(name)
        if method is None:
            method = self._flipped_methods.get(name)
            if method is None:
                method = self._instance_methods.get(name)
                if method is None:
                    return object.__getattribute__(self, name)

                @functools.wraps(method)
                def wrap_instance(*arguments, **kwargs):
                    return self.append_function(self, method(*arguments, **kwargs))

                return wrap_instance
            else:
                method = flip(method)

        @functools.wraps(method)
        def wrap_method(*arguments, **kwargs):
            return self.append_function(self, method, *arguments, **kwargs)

        return wrap_method

    @classmethod
    def add_method(cls, method, name=None, *, kind="partial"):
        if name is None:
            name = method.__name__

        data = object.__getattribute__(cls, f"_{kind}_methods")
        data.setdefault(name, method)

    def __dir__(self):
        return list(self._methods) + object.__dir__(self)


class composition(methods, attributes, ABCMeta):
    def _ipython_key_completions_(*self):
        object = []
        try:
            object = __import__("IPython").core.completerlib.module_completion("import")
        except:
            return []  # we wont need this if ipython ain't around
        return object + list(
            itertools.chain(
                *[
                    [f"{k}.{v}" for v in dir(v) if not v.startswith("_")]
                    for k, v in sys.modules.items()
                ]
            )
        )


class λ(interface, attributes, metaclass=composition):
    """Extensible function partial composition that excepts Exceptions.

    All of position is overlay on the toolz library that supports conventions for functional programming in python.
    The toolz documentation are some of the best in python.

    >>> composition([list, range], 5)(10)
    [5, 6, 7, 8, 9]

    λ or poser is a convenience type for building compositions.

    >>> composition([range], 1)
    composition(<class 'range'>,)
    >>> λ(5).partial(range).list()(10)
    [5, 6, 7, 8, 9]

    >>> assert λ() ^ λ
    >>> assert bool(composition) and bool(composition())

    >>> composition([range, enumerate, dict], 5)
    composition(<class 'range'>, <class 'enumerate'>, <class 'dict'>)

    >>> assert len(λ()) ^ len(λ)

    >>> assert composition().partial(range).partial().partial(type)(1) is range
    """

    def __new__(cls, *arguments, **kwargs):
        if cls is λ:
            cls = poser
        self = super().__new__(cls)
        self.__init__(arguments=arguments, kwargs=kwargs)
        return self

    def __init__(self, *arguments, **kwargs):
        super().__init__(arguments=arguments, kwargs=kwargs)

    def _prepare_args(self, *arguments, **kwargs):
        if arguments:
            object, *_ = arguments
        else:
            object = None
        return self.arguments + arguments, {**self.kwargs, **kwargs}

    def __call__(self, *arguments, **kwargs):
        """Call a partial composition with arguments and kwargs."""
        # A starred method has already consumed the arguments and kwargs.

        # Iterate over the callables in the composition piping
        # the output across each function
        arguments, kwargs = self._prepare_args(*arguments, **kwargs)
        if self.exceptions:
            for callable in self:
                try:
                    arguments, kwargs = (callable(*arguments, **kwargs),), {}
                    object = arguments[0]
                except self.exceptions as Exception:
                    return Ø(Exception)
        else:
            for callable in self:
                arguments, kwargs = (callable(*arguments, **kwargs),), {}
                object = arguments[0]

        return object


class poser(methods, λ):
    _ipython_key_completions_ = composition._ipython_key_completions_


@dataclass
class conditional(poser):
    predicate: typing.Callable = None

    def __new__(cls, predicate, *arguments, **kwargs):
        self = super().__new__(cls)
        self.__init__(predicate, *arguments, **kwargs)
        return self

    def __init__(self, *arguments, **kwargs):
        if arguments:
            predicate, *arguments = arguments
            kwargs.setdefault("predicate", predicate)

        super().__init__()
        self.__dict__.update(kwargs)


class ifthen(conditional):
    def __call__(self, *arguments, **kwargs):
        object = self.predicate(*arguments, **kwargs)
        return super().__call__(*arguments, **kwargs) if object else object


class ifnot(conditional):
    def __call__(self, *arguments, **kwargs):
        object = self.predicate(*arguments, **kwargs)
        return object if object else super().__call__(*arguments, **kwargs)


__test__ = globals().get("__test__", {})
__test__[
    __name__
] = """
#### Tests

Initializing a composition.

    >>> assert λ() == λ+... == +λ
    >>> +λ
    λ(<function I at ...>,)

Composing compositions.

    >>> λ[callable]
    λ(<built-in function callable>,)
    >>> assert λ[callable] == λ+callable == callable+λ == λ.partial(callable) == λ-callable == callable-λ
    >>> assert λ[callable] != λ[callable][range]

Juxtapositions.

    >>> λ[type, str] #doctest: +ELLIPSIS
    λ(<...juxt object at ...>,)
    >>> λ[type, str](10)
    (<class 'int'>, '10')
    >>> λ[{type, str}][type, len](10)
    (<class 'set'>, 2)
    >>> λ[{'a': type, type: str}](10)
    {'a': <class 'int'>, <class 'int'>: '10'}
    >>> λ[[[[[{'a': type, type: str}]]]],]
    λ(<...juxt object at ...>,)
    >>> λ[[[[[{'a': type, type: str}]]]],](10)
    ([[[[{'a': <class 'int'>, <class 'int'>: '10'}]]]],)

    
Mapping.

    >>> (λ[range] * type + list)(3)
    [<class 'int'>, <class 'int'>, <class 'int'>]
    >>> λ[range].map((type, str))[list](3)
    [(<class 'int'>, '0'), (<class 'int'>, '1'), (<class 'int'>, '2')]
    
Filtering

    >>> (λ[range] / λ[(3).__lt__, (2).__rfloordiv__][all] + list)(10)
    [4, 5, 6, 7, 8, 9]
    >>> (λ[range] / (λ[(3).__lt__, (2).__rmod__][all]) + list)(10)
    [5, 7, 9]
    
Filtering Mappings

    >>> λ('abc').enumerate().dict().filter('ab'.__contains__)()
    {0: 'a', 1: 'b'}
    >>> λ('abc').enumerate().dict().filter(λ().partial(operator.__contains__, 'bc') , (1).__lt__)()
    {2: 'c'}
    >>> λ('abc').enumerate().dict().filter(λ(),(1).__lt__)()
    {2: 'c'}
    
Groupby
    
    >>> assert λ[range] @ (2).__rmod__ == λ[range].groupby((2).__rmod__)
    >>> (λ[range] @ (2).__rmod__)(10)
    {0: [0, 2, 4, 6, 8], 1: [1, 3, 5, 7, 9]}
    
Reduce
    
    >>> assert λ[range]%int.__add__ == λ[range].reduce(int.__add__)
    >>> (λ[range] % int.__add__)(10)
    45
    
conditionals

    >>> λ[λ**int+bool, λ**str](10)
    (True, False)


Normal slicing when all the slices are integers or None

    >>> len(λ.range().skip()), len(λ.range().skip())
    (1, 1)

Callable slicing
    
    >>> slice(filter, pipe, map)
    slice(<...filter...pipe...map...>)
    >>> λ.range(6)[(Λ-1)%2:λ[str, type]:dict]+...
    {'0': <class 'int'>, '2': <class 'int'>, '4': <class 'int'>}
    
    >>> λ[bool:type]
    λ(...map...filter...)


Length and Logic

    >>> len(λ), bool(λ)
    (0, False)
    >>> len(λ()), bool(λ())
    (1, True)
    
Syntactic sugar causes cancer of the semicolon.  

    
Starred functions allows arguments and dictionaries to be defined in iterables.

    >>> star.range()([0,10])
    range(0, 10)
    >>> star[dict](λ[range][reversed][enumerate][[list]](3))
    {0: 2, 1: 1, 2: 0}
    
    >>> star((0,)).range()((1, 2))
    range(0, 1, 2)

   
   
Unary functions:

    >>> (~λ[range])(10, 2)
    range(2, 10)
    >>> assert not (-λ[bool])('abc')
    >>> assert (-λ[bool])('')
    >>> assert +λ == λ() == λ+...
    
    >>> reversed(λ[list][range][int])('5')
    [0, 1, 2, 3, 4]
    
    
Math:

    >>> λ.range(2, 10) * λ(1).range().mean() + list + ...
    [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5]
    
Exceptions:
    
    >>> λ[str.split].map(λ[int] ^ BaseException | str).list()("10 aaa")
    [10, 'aaa']

Extra:
`poser` has an enormous api for gluing functions together across python.
    
    >>> assert λ(λ) + dir + len + (Λ>650) + ...
    >>> (Λ[1][1]['a'].__add__(22).__truediv__(7))\\
    ...     ((None, [None, {'a': 20}]))
    6.0

    >>> assert λ.sub(10).add(3).truediv(2)(20) == 6.5
    >>> assert λ.fnmatch('abc*')('abcde')
    
    
"""
