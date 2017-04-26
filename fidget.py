# coding: utf-8

from copy import copy
from functools import wraps, total_ordering, partial
from importlib import import_module
from six import iteritems
from toolz.curried import (isiterable, first, excepts, flip, complement, map,
                           identity, concatv, valfilter, merge, groupby,
                           concat, get, keyfilter, compose, reduce)
from operator import (methodcaller, itemgetter, attrgetter, not_, truth, abs,
                      invert, neg, pos, index, eq)

__all__ = ['_x', '_xx', 'x_', '_y', 'call', 'defaults', 'ifthen', 'copy']


class State(object):
    def __init__(self, *args, **kwargs):
        for i, slot in enumerate(self.__slots__):
            setattr(self, slot, kwargs.pop(slot, args[i]))

    def __getstate__(self):
        return tuple(map(partial(getattr, self), self.__slots__))

    def __setstate__(self, state):
        for key, value in zip(self.__slots__, state):
            setattr(self, key, value)

    def __copy__(self, *args):
        new = self.__class__()
        return new.__setstate__(tuple(map(copy, self.__getstate__()))) or new

    __deepcopy__ = __copy__


class function(State):
    __slots__ = ('function', )

    def __repr__(self):
        return repr(self.function)


class functor(function):
    def __call__(self, *args, **kwargs):
        return self.function(
            *args, **kwargs) if callable(self.function) else self.function


class flipped(functor):
    def __call__(self, *args, **kwargs):
        return super(flipped, self).__call__(*reversed(args), **kwargs)


class do(functor):
    def __call__(self, *args, **kwargs):
        super(do, self).__call__(*args, **kwargs)
        return args[0] if args else None


class stars(functor):
    def __call__(self, *args, **kwargs):
        arguments = groupby(flip(isinstance)(dict),
                            args) if all(map(isiterable, args)) else {
                                False: [[args]]
                            }
        return super(stars, self).__call__(
            *concat(get(False, arguments, tuple())), **merge(
                kwargs, *get(True, arguments, [{}])))


class condition(functor):
    __slots__ = ('condition', 'function')


class ifthen(condition):
    def __call__(self, *args, **kwargs):
        return functor(self.condition)(*args, **kwargs) and super(
            condition, self).__call__(*args, **kwargs)


class defaults(condition):
    def __call__(self, *args, **kwargs):
        return super(condition, self).__call__(
            *args, **kwargs) or functor(self.condition)(*args, **kwargs)


def doc(self):
    return getattr(self.function, '__doc__', '')


for func in (functor, flipped, do, stars, ifthen, defaults):
    setattr(func, '__doc__', property(doc))


class call(State):
    __slots__ = ('args', 'kwargs')

    def __init__(self, *args, **kwargs):
        super(call, self).__init__(args, kwargs)

    def __call__(self, function=identity):
        return partial(functor(function), *self.args, **self.kwargs)


class Function(State):
    """`Functions` is the base class for `map` and `flatmap` function
    compositions.
    New functions are added to the compositions using the `__getitem__`
    attribute.
    """
    __slots__ = ('_functions', )

    def __init__(self, functions=tuple(), *args):
        if not isiterable(functions) or isinstance(functions, (str, )):
            functions = (functions, )

        super(Function, self).__init__(
            (isinstance(functions, dict) and iteritems or
             identity)(functions), *args)

    def __getitem__(self, item):
        if isinstance(item, slice):
            if item != slice(None):
                self._functions = self._functions[item]
        else:
            self._functions = tuple(concatv(self._functions, (item, )))
        return self

    def __iter__(self):
        for function in self._functions:
            yield (isinstance(function, (dict, set, list, tuple)) and
                   call(codomain=type(function))(Juxtapose) or
                   identity)(function)

    def __reversed__(self):
        self._functions = tuple(reversed(self._functions))
        return self

    def __repr__(self):
        return str(self._functions)

    def __delitem__(self, item):
        self._functions = tuple(fn for fn in self if fn != item)
        return self

    def __setitem__(self, key, value):
        self._functions = tuple(value if fn == key else fn for fn in self)
        return self

    def __abs__(self):
        return self.__call__


@total_ordering
class Composite(Function):
    def __enter__(self):
        return copy(self[:])

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def __len__(self):
        return len(self._functions)

    def __hash__(self):
        return hash(self._functions)

    def __eq__(self, other):
        if isinstance(other, Composite):
            return hash(self) == hash(other)
        return False

    def __lt__(self, other):
        if isinstance(other, Composite):
            return (len(self) < len(other)) and all(
                eq(*i) for i in zip(self, other))
        return False

    def __contains__(self, item):
        return any(item == function for function in self)


class Juxtapose(Composite):
    """`Juxtapose` applies the same arguments and keywords to many functions.
    """
    __slots__ = ('_functions', '_codomain')

    def __init__(self, functions=tuple(), codomain=identity):
        super(Juxtapose, self).__init__(functions, codomain)

    def __call__(self, *args, **kwargs):
        return self._codomain(
            call(*args)(function)(**kwargs) for function in self)


class Compose(Composite):
    """`Compose` chains functions together.
    """

    def __call__(self, *args, **kwargs):
        for function in self:
            args, kwargs = (call(*args)(function)(**kwargs), ), {}
        return first(args)


class Partial(Composite):
    """`Callables` are `Functions` that store partial `argument` & `keyword`
    states.
    """
    __slots__ = ('_args', '_keywords', '_functions')
    _decorate_, _composite_ = staticmethod(identity), staticmethod(Compose)

    def __init__(self, *args, **kwargs):
        super(Partial, self).__init__(args, kwargs,
                                      kwargs.pop('functions',
                                                 self._composite_()))

    def __getitem__(self, item=None):
        if self._factory_:
            self = type(self).__mro__[1]()

        if item is call:
            return abs(self)

        if isinstance(item, call):
            return item(self)()

        self._functions[item]
        return self

    @property
    def _function(self):
        return self._decorate_(self._functions)

    def __hash__(self):
        return hash((self._functions, self._args, type(self)))

    @property
    def __call__(self):
        return call(*self._args, **self._keywords)(self._function)

    def __invert__(self):
        self = self[:]
        return setattr(self, '_functions', reversed(self._functions)) or self

    @property
    def _factory_(self):
        return type(self).__name__.startswith('_') and type(
            self).__name__.endswith('_')

    __and__ = __add__ = __rshift__ = __sub__ = __getitem__


class Juxtaposition(Partial):
    _composite_ = staticmethod(Juxtapose)


class Composition(Partial):
    def __getitem__(self, item=None, *args, **kwargs):
        return super(Composition, self).__getitem__(
            (args or kwargs) and call(*args, **kwargs)(item) or item)

    def __xor__(self, item):
        """** operator requires an argument to be true because executing.
        """
        self, method = self[:], ifthen
        if isinstance(item, type):
            if issubclass(item, Exception) or isiterable(item) and all(
                    map(flip(isinstance)(Exception), item)):
                method = excepts
        elif isiterable(item) and all(map(flip(isinstance)(type), item)):
            item = flip(isinstance)(item)
        self._functions = Compose([method(item, self._functions)])
        return self

    def __or__(self, item):
        """| returns a default value if composition evaluates true.
        """
        self = self[:]
        self._functions = Compose([defaults(item, self._functions)])
        return self

    def __pos__(self):
        return self[bool]

    def __neg__(self):
        return self[complement(bool)]

    def __lshift__(self, item):
        return Do()[item] if self._factory_ else self[do(item)]

    __pow__, __mul__ = __xor__, __getitem__


class Flipped(Composition):
    _decorate_ = staticmethod(flipped)


class Starred(Composition):
    _decorate_ = staticmethod(stars)


class Do(Composition):
    _decorate_ = staticmethod(do)


def macro(attr, method, cls=Composition):
    """Adds attributes to `Compositon` `cls` to extend an api to contain named
    actions.
    """
    _impartial = not isinstance(method, partial)
    method = not _impartial and method.func or method

    def _macro(self, *args, **kwargs):
        return (
            args or
            kwargs) and self[_impartial and partial(method, *args, **kwargs) or
                             method(*args, **kwargs)] or self[method]

    setattr(cls, attr, getattr(cls, attr, wraps(method)(_macro)))


for attr, method in [('__matmul__', groupby), ('__div__', map), (
        '__truediv__', map), ('__floordiv__', filter), ('__mod__', reduce)]:
    macro(attr, method)


def _right_(attr):
    """Add right operators from the python data model.
    """

    def caller(self, other):
        self = self[:]
        if isinstance(other, call):
            other = self.__class__(*other.args, **other.kwargs)
        else:
            other = self.__class__()[other]
        return methodcaller(attr, copy(self))(other) if self else other

    return wraps(getattr(Composition, """__{}__""".format(attr)))(caller)


s = "__{}{}__".format
for attr in [
        'add', 'sub', 'mul', 'matmul', 'div', 'truediv', 'floordiv', 'mod',
        'lshift', 'rshift', 'and', 'xor', 'or'
]:
    setattr(Composition, s('i', attr), getattr(Composition, s('', attr)))
    setattr(Composition, s('r', attr), _right_(attr))

for attr, method in [['call'] * 2, ['do', 'lshift'], ['pipe', 'getitem']]:
    setattr(Composition, attr, getattr(Composition, s('', method)))

# introduce functional programming namespaces from `toolz` and `operator`.
for imports in ('toolz', 'operator'):
    attrs = compose(iteritems,
                    valfilter(callable),
                    keyfilter(compose(str.lower, first)), vars,
                    import_module)(imports)
    for attr, method in attrs:
        method = (identity
                  if method in (flip, ) or imports == 'toolz' else partial
                  if method in (methodcaller, itemgetter, attrgetter, not_,
                                truth, abs, invert, neg, pos, index) else
                  flipped)(method)
        macro(attr, method)

_y, _x, x_, _xx = tuple(
    type('_{}_'.format(function.__name__), (function, ),
         {})(functions=Compose([function]))
    for function in (Juxtaposition, Composition, Flipped, Starred))

del attr, attrs, doc, func, imports, method, s
