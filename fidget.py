# coding: utf-8

from copy import copy
from functools import wraps, total_ordering, partial
from importlib import import_module
from decorator import decorate
from six import iteritems, PY34
from toolz.curried import (isiterable, first, excepts, flip, last, complement,
                           map, identity, concatv, valfilter, merge, curry,
                           groupby, concat, get, keyfilter, compose, reduce)
from operator import (methodcaller, itemgetter, attrgetter, not_, truth, abs,
                      invert, neg, pos, index, eq)

__all__ = [
    '_x', '_xx', 'x_', '_y', 'call', 'defaults', 'ifthen', 'copy', 'macro'
]


class State(object):
    """`State` defines the data model attributes that copy and pickle objects.
    
    Here is the example
    """

    def __getstate__(self):
        return tuple(map(partial(getattr, self), self.__slots__))

    def __setstate__(self, state):
        for key, value in zip(self.__slots__, state):
            setattr(self, key, value)

    def __copy__(self, *args):
        new = self.__class__()
        return new.__setstate__(tuple(map(copy, self.__getstate__()))) or new


State.__deepcopy__ = State.__copy__


def _wrapper(function, caller, *other_decorators):
    """`_wrapper` is used to decorate objects with `itertools.wraps`,
    `identity`, or sometimes `decorate`.
    """

    for wrap in concatv(other_decorators, (wraps, )):
        try:
            return wrap(function)(caller)
        except:
            pass
    return caller


def functor(function):
    """`functor` will return the called `function` if it is `callable` or
    `function`.
    
            functor(range)(10)
            functor(3)(10)
    """

    def caller(*args, **kwargs):
        if callable(function):
            return function(*args, **kwargs)
        return function

    return callable(function) and _wrapper(function, caller) or caller


class call(State):
    """`call` creates a `callable` with `partial` arguments.  Arguments are
    immutable, keywords are mutable.
    
            f = call(10, 20)(range)
            f(3)
            f(2)
    
    `call` applies `functor` by default
    
            call(10, 20)(3)()
    
    polymorphisms
    
            f(*args, **kwargs)(func)()
            f(*args)(func)(**kwargs)
            f()(func)(*args, **kwargs)
    
    `fidget` relies heavily on call through the composition process.
    """

    __slots__ = ('args', 'kwargs')

    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs

    def __call__(self, function=identity):
        def caller(*args, **kwargs):
            return functor(function)(*concatv(self.args, args), **merge(
                self.kwargs, kwargs))

        return callable(function) and _wrapper(function, caller) or caller


def do(function):
    """`do` calls a function & returns the `first` argument
    """

    def caller(*args, **kwargs):
        function(*args, **kwargs)
        return first(args) if len(args) else tuple()

    return _wrapper(function, caller, curry(decorate))


def flipped(function):
    """call a `function` with the argument order `flipped`.
    
    > `flip` only works with binary operations.
    """

    def caller(*args, **kwargs):
        return call(*reversed(args), **kwargs)(function)()

    return _wrapper(function, caller, curry(decorate))


def stars(function):
    """`stars` converts iterables to starred arguments, and vice versa.
    """

    def caller(*args, **kwargs):
        if all(map(isiterable, args)):
            combined = groupby(flip(isinstance)(dict), args)
            args = concat(get(False, combined, tuple()))
            kwargs = merge(kwargs, *get(True, combined, [{}]))
            return call(*args)(function)(**kwargs)
        return call(args)(function)(**kwargs)

    return _wrapper(function, caller, curry(decorate))


def defaults(default):
    """`defaults` to another operation if `bool(function) is False`
    """

    def caller(function):
        def defaults(*args, **kwargs):
            return call(*args)(function)(**kwargs) or call(*args)(default)(
                **kwargs)

        return _wrapper(function, defaults, curry(decorate))

    return caller


def ifthen(condition):
    """`ifthen` requires a `condition` to be true before executing `function`.
    """

    def caller(function):
        def ifthen(*args, **kwargs):
            return call(*args)(condition)(**kwargs) and call(*args)(function)(
                **kwargs)

        return _wrapper(function, ifthen, curry(decorate))

    return caller


@total_ordering
class Functions(State):
    """`Functions` is the base class for `map` and `flatmap` function
    compositions.
    New functions are added to the compositions using the `__getitem__`
    attribute.
    """

    __slots__ = ('_functions', )
    _log = None

    def __init__(self, functions=tuple()):
        if not isiterable(functions) or isinstance(functions, (str, )):
            functions = (functions, )

        self._functions = (isinstance(functions, dict) and iteritems or
                           identity)(functions)

    def __getitem__(self, item):
        if isinstance(item, slice):
            if item == slice(None):
                pass
            else:
                self._functions = self._functions[item]
        else:
            self._functions = tuple(concatv(self._functions, (item, )))
        return self

    def __enter__(self):
        return copy(self[:])

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def __iter__(self):
        for function in self._functions:
            yield (isinstance(function, (dict, set, list, tuple)) and
                   call(codomain=type(function))(Juxtapose) or
                   identity)(function)

    def __len__(self):
        return len(self._functions)

    def __hash__(self):
        return hash(self._functions)

    def __eq__(self, other):
        if isinstance(other, Functions):
            return hash(self) == hash(other)
        return False

    def __lt__(self, other):
        if isinstance(other, Functions):
            return (len(self) < len(other)) and all(
                eq(*i) for i in zip(self, other))
        return False

    def __reversed__(self):
        self._functions = tuple(reversed(self._functions))
        return self

    def __repr__(self):
        return str(self._functions)

    def __contains__(self, item):
        return any(item == function for function in self)

    def __delitem__(self, item):
        self._functions = tuple(fn for fn in self if fn != item)
        return self

    def __setitem__(self, key, value):
        self._functions = tuple(value if fn == key else fn for fn in self)
        return self

    def __abs__(self):
        return self.__call__


class Juxtapose(Functions):
    """`Juxtapose` applies the same arguments and keywords to many functions.
    """

    __slots__ = ('_functions', '_codomain')

    def __init__(self, functions=tuple(), codomain=identity):
        self._codomain = codomain
        super(Juxtapose, self).__init__(functions)

    def __call__(self, *args, **kwargs):
        return self._codomain(
            call(*args)(function)(**kwargs) for function in self)


class Compose(Functions):
    """`Compose` chains functions together.
    """

    def __call__(self, *args, **kwargs):
        for function in self:
            args, kwargs = (call(*args)(function)(**kwargs), ), {}
        return first(args)


class Callables(Functions):
    """`Callables` are `Functions` that store partial `argument` & `keyword`
    states.
    """

    _functions_default_ = Compose
    _factory_, _do, _func_ = None, False, staticmethod(identity)
    __slots__ = ('_functions', '_args', '_keywords')

    def __getitem__(self, item=None):
        if self._factory_:
            self = self()
        if item is call:
            return abs(self)

        if isinstance(item, call):
            return item(self)()
        self._functions[item]
        return self

    def __init__(self, *args, **kwargs):
        self._functions = kwargs.pop('functions', self._functions_default_())
        self._args, self._keywords = args, kwargs

    def __func__(self):
        if self._do:
            return do(self._functions)
        return (self._factory_ and identity or self._func_)(self._functions)

    def __hash__(self):
        return hash((self._functions, self._args, self._do, self.__class__))

    @property
    def __call__(self):
        return call(*self._args, **self._keywords)(self.__func__())

    def __lshift__(self, item):
        if self._factory_:
            return type('_Do_', (Callables, ), {'_do': True})()[item]
        return self[do(item)]

    def __invert__(self):
        self = self[:]
        self._functions = reversed(self._functions)
        return self

    __and__ = __add__ = __rshift__ = __sub__ = __getitem__


class Juxtaposition(Callables):
    """`Juxtaposition` is `Juxtapose` with arguments.
    """

    _functions_default_ = Juxtapose


class Composition(Callables):
    """`Composition` is `Compose` with arguments.
    """

    def __getitem__(self, item=None, *args, **kwargs):
        return super(Composition, self).__getitem__(
            (args or kwargs) and call(*args, **kwargs)(item) or item)

    def __xor__(self, item, method=ifthen):
        """** operator requires an argument to be true because executing.
        """
        self = self[:]
        if isinstance(item, type):
            if issubclass(item, Exception) or isiterable(item) and all(
                    map(flip(isinstance)(Exception), item)):
                method = excepts
        elif isiterable(item) and all(map(flip(isinstance)(type), item)):
            item = flip(isinstance)(item)
        self._functions = Compose([method(item)(self._functions)])
        return self

    def __or__(self, item):
        """| returns a default value if composition evaluates true.
        """
        self = self[:]
        self._functions = Compose([defaults(item)(self._functions)])
        return self

    def __pos__(self):
        return self[bool]

    def __neg__(self):
        return self[complement(bool)]

    __pow__ = __xor__
    __mul__ = __getitem__


def macro(attr, method, cls=Composition):
    """Adds attributes to `Compositon` `cls` to extend an api to contain named
    actions.
    """

    _impartial = not isinstance(method, partial)
    method = not _impartial and method.func or method

    def caller(self, *args, **kwargs):
        return (
            args or
            kwargs) and self[_impartial and partial(method, *args, **kwargs) or
                             method(*args, **kwargs)] or self[method]

    setattr(cls, attr, getattr(cls, attr, wraps(method)(caller)))


for attr, method in [('__matmul__', groupby), ('__div__', map), (
        '__truediv__', map), ('__floordiv__', filter), ('__mod__', reduce)]:
    macro(attr, method)


# Attributes and symbols to compose functions.
# 
# Create attributes that apply function programming methods from `toolz` and `operator`.
# 
#         _x.map(_x.add(10)).filter(_x.gt(4)).pipe(list)(range(20))
def _rattribute_(attr):
    """Add right operators from the python data model.
    """

    attr = """__{}__""".format(attr)

    def caller(self, other):
        self = self[:]
        if isinstance(other, call):
            other = self.__class__(*other.args, **other.kwargs)
        else:
            other = self.__class__()[other]
        return methodcaller(attr, copy(self))(other) if self else other

    return wraps(getattr(Composition, attr))(caller)


s = "__{}{}__".format
for attr in [
        'add', 'sub', 'mul', 'matmul', 'div', 'truediv', 'floordiv', 'mod',
        'lshift', 'rshift', 'and', 'xor', 'or'
]:
    setattr(Composition, s('i', attr), getattr(Composition, s('', attr)))
    setattr(Composition, s('r', attr), _rattribute_(attr))
del _rattribute_

for attr, method in [['call'] * 2, ['do', 'lshift'], ['pipe', 'getitem']]:
    setattr(Composition, attr, getattr(Composition, s('', method)))

for imports in ('toolz', 'operator'):
    """introduce fucntional programming namespaces from `toolz` and `operator`.
    """

    for attr, method in compose(iteritems,
                                valfilter(callable),
                                keyfilter(compose(str.lower, first)), vars,
                                import_module)(imports):
        macro(attr, (identity
                     if method in (flip, ) or imports == 'toolz' else partial
                     if method in (methodcaller, itemgetter, attrgetter, not_,
                                   truth, abs, invert, neg, pos, index) else
                     flipped)(method))


class Flipped(Composition):
    """`Flipped` is a `Composition` with the arguments reversed; keywords have
    no ordering.
    """

    _func_ = staticmethod(flipped)


class Starred(Composition):
    """`Starred` is a `Composition` that applies an iterable as starred
    arguments.
    
            _xx([10, 20, 3]) == _x(10, 20, 3)
    """

    _func_ = staticmethod(stars)


if PY34:

    class This(Callables):
        """`This` composes functions for an object applying 
        attribute getter, item getter, and method caller.
        """

        def __getattr__(self, attr):
            if any(
                    attr.startswith(key)
                    for key in ('__', '_repr_', '_ipython_')):
                return self
            return super(This, self).__getitem__(
                callable(attr) and attr or attrgetter(attr))

        def __getitem__(self, item):
            return super(This, self).__getitem__(
                callable(item) and item or itemgetter(item))

        def __call__(self, *args, **kwargs):
            previous = last(self._functions._functions)
            if type(previous) == attrgetter:
                attrs = previous.__reduce__()[-1]
                if len(attrs) == 1:
                    self._functions = self._functions[:-1]
                return self[methodcaller(first(attrs), *args, **kwargs)]
            return super(This, self).__call__(*args, **kwargs)

    this = type('_This_', (This, ), {'_factory_':
                                     True})(functions=Compose([This]))

    __all__ += ['this']

del imports, attr, method, PY34

_y, _x, x_, _xx = tuple(
    type('_{}_'.format(f.__name__), (f, ), {
        '_factory_': True,
    })(functions=Compose([f]))
    for f in (Juxtaposition, Composition, Flipped, Starred))
