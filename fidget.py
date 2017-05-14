# coding: utf-8

# > `fidget` uses the python data model to compose higher-order functions.
# 
# ---

from copy import copy
from functools import wraps, total_ordering, partial
from importlib import import_module
from six import iteritems, PY2
from toolz.curried import (isiterable, filter, first, excepts, flip,
                           complement, map, identity, concatv, valfilter,
                           merge, groupby, concat, get, keyfilter, compose,
                           reduce)
from six.moves.builtins import hasattr, getattr, isinstance, issubclass, setattr
from operator import (methodcaller, itemgetter, attrgetter, not_, truth, abs,
                      invert, neg, pos, index, eq)

__all__ = [
    '_x', '_xx', '_f', 'x_', '_y', '_h', 'call', 'default', 'ifthen', 'copy'
]


@total_ordering
class State(object):
    __slots__ = tuple()

    def __init__(self, *args, **kwargs):
        for i, slot in enumerate(self.__slots__):
            setattr(self, slot, kwargs.pop(slot, args[i]))

    def __getstate__(self):
        return tuple(map(partial(getattr, self), self.__slots__))

    def __setstate__(self, state):
        for key, value in zip(self.__slots__, state):
            setattr(self, key, value)

    def __copy__(self, *args):
        new = type(self)()
        return new.__setstate__(tuple(map(copy, self.__getstate__()))) or new

    def __hash__(self):
        return hash(tuple(getattr(self, attr) for attr in self.__slots__))

    def __eq__(self, other):
        return isinstance(other, State) and hash(self) == hash(other)

    def __enter__(self):
        return copy(self[:])

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def __abs__(self):
        return self.__call__

    def __lt__(self, other):
        if isinstance(other, State):
            return (len(self) < len(other)) and all(
                eq(*i) for i in zip(self, other))
        return False

    def __len__(self):
        return sum(1 for f in self)

    __deepcopy__ = __copy__


class functor(State):
    """Evaluate a function if it is `callable`, otherwise return value."""
    __slots__ = ('function', )

    def __call__(self, *args, **kwargs):
        return self.function(
            *args, **kwargs) if callable(self.function) else self.function


class flipped(functor):
    """Evaluate a function with the arguments reversed."""

    def __call__(self, *args, **kwargs):
        return super(flipped, self).__call__(*reversed(args), **kwargs)


class do(functor):
    """Evaluate a function and return the arguments"""

    def __call__(self, *args, **kwargs):
        super(do, self).__call__(*args, **kwargs)
        return args[0] if args else None


class stars(functor):
    """Evaluate a function that expands sequences and collections to star arguments and keywords.
    """

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
    """Evaluate the condition, and if it is True evaluate function.
    """

    def __call__(self, *args, **kwargs):
        return functor(self.condition)(*args, **kwargs) and super(
            ifthen, self).__call__(*args, **kwargs)


class default(condition):
    """Evaluate the function, and if it is False evaluate condition.
    """

    def __call__(self, *args, **kwargs):
        return super(default, self).__call__(
            *args, **kwargs) or functor(self.condition)(*args, **kwargs)


class step(condition):
    def __call__(self, *args, **kwargs):
        result = functor(self.condition)(*args, **kwargs)
        return result and super(step, self).__call__(result)


class call(State):
    __slots__ = ('args', 'kwargs')

    def __init__(self, *args, **kwargs):
        super(call, self).__init__(args, kwargs)

    def __call__(self, function=functor):
        return partial(functor(function), *self.args, **self.kwargs)


def doc(self):
    return getattr(self.function, '__doc__', '')


for func in (functor, flipped, do, stars, ifthen, default, call):
    not PY2 and setattr(func, '__doc__', property(doc))


class Function(State):
    __slots__ = ('function', )

    def __getitem__(self, object=slice(None)):
        if object is call:
            return abs(self)

        if isinstance(object, call):
            return object(self)()

        return object != slice(None) and self.function.append(object) or self

    def __repr__(self):
        return str(self.function)


class Functions(State):
    __slots__ = ('functions', )

    def __repr__(self):
        return str(self.functions)

    def __init__(self, functions=tuple(), *args):
        if not isiterable(functions) or isinstance(functions, (str, )):
            functions = (functions, )

        super(Functions, self).__init__(
            (isinstance(functions, dict) and compose(tuple, iteritems) or
             identity)(functions), *args)

    def __delitem__(self, object):
        self.functions = tuple(fn for fn in self if fn != object)
        return self

    def __setitem__(self, key, value):
        self.functions = tuple(value if fn == key else fn for fn in self)
        return self

    def __iter__(self):
        for function in self.functions:
            yield function

    def __reversed__(self):
        self.functions = type(self.functions)(reversed(self.functions))
        return self

    def insert(self, index, object):
        self.functions = tuple(
            concat((self.functions[:index], (object, ), self.functions[index:]
                    )))

    def append(self, object):
        self.extend((object, ))

    def extend(self, iterable):
        self.functions = tuple(concatv(self.functions, iterable))


class Composite(Functions):
    def __getitem__(self, object=slice(None)):
        if isinstance(object, slice):
            if object != slice(None):
                with self as self:
                    self.append(object)
        else:
            self = super(Composite, self).__getitem__(object)
        return self

    def __contains__(self, object):
        return any(object == function for function in self)

    def _dispatch_(self, function):
        return (isinstance(function, (dict, set, list, tuple)) and Juxtapose or
                functor)(function)


class Juxtapose(Composite):
    def __init__(self, functions=tuple()):
        super(Juxtapose, self).__init__(functions)

    def __call__(self, *args, **kwargs):
        return type(self.functions)(call(*args)(self._dispatch_(function))(
            **kwargs) for function in self)


class Compose(Composite):
    def __call__(self, *args, **kwargs):
        for function in self:
            args, kwargs = (call(*args)(self._dispatch_(function))(
                **kwargs), ), {}
        return first(args)


class ComposeLeft(Compose):
    def __iter__(self):
        return reversed(tuple(super(ComposeLeft, self).__iter__()))

    def __hash__(self):
        return hash(tuple(reversed(self.functions)))


class Composable(Compose):
    def __init__(self, function):
        super(Composable, self).__init__([function])


class Partial(Function):
    __slots__ = ('args', 'keywords', 'function')
    _decorate_ = staticmethod(functor)
    _composite_ = staticmethod(Compose)

    def __init__(self, *args, **kwargs):
        super(Partial, self).__init__(args, kwargs,
                                      kwargs.pop('function',
                                                 self._composite_()))

    @property
    def __call__(self):
        return call(*self.args,
                    **self.keywords)(self._decorate_(self.function))


class Composition(Partial):
    def __getitem__(self, object=slice(None), *args, **kwargs):
        if self._factory_:
            self = type(self).__mro__[1]()

        if isinstance(object, (int, slice)):
            self = copy(self)
            return setattr(
                self, 'function',
                self._composite_(self.function.functions[object])) or self

        return super(Composition, self).__getitem__(
            (args or kwargs) and call(*args, **kwargs)(object) or object)

    @property
    def _factory_(self):
        return type(self).__name__.startswith('_') and type(
            self).__name__.endswith('_')


class Juxtaposition(Composition):
    _composite_ = staticmethod(Juxtapose)


class Composer(Composition):
    def __xor__(self, object):
        """** operator requires an argument to be true because executing.
        """
        self = self[:]
        if isiterable(object):
            if all(map(flip(isinstance)(Exception), object)):
                method = excepts
            elif all(map(flip(isinstance)(type), object)):
                object = flip(isinstance)(object)
        self.function = Compose([ifthen(object, self.function)])
        return self

    def __or__(self, object):
        self = self[:]
        self.function = Compose([default(object, self.function)])
        return self

    def __and__(self, object):
        self = self[:]
        self.function = Compose([step(self.function, object)])
        return self

    def __pos__(self):
        return self[bool]

    def __neg__(self):
        return self[complement(bool)]

    def __lshift__(self, object):
        return Do()[object] if self._factory_ else self[do(object)]

    __pow__ = __xor__
    __invert__ = Functions.__reversed__
    __mul__ = __add__ = __rshift__ = __sub__ = Composition.__getitem__


class Flipped(Composer):
    _decorate_ = staticmethod(flipped)


class Starred(Composer):
    _decorate_ = staticmethod(stars)


class Do(Composer):
    _decorate_ = staticmethod(do)


class Reversed(Composer):
    _composite_ = staticmethod(ComposeLeft)


def macro(attr, method, cls=Composer, composable=False, force=False):
    """Extent the Composer api.
    """
    if not hasattr(cls, attr) or force:
        _partial = isinstance(method, partial)
        method = _partial and method.func or method

        def _macro(self, *args, **kwargs):
            if len(args) is 1 and composable and not _partial:
                args = (Composable(args[0]), )

            return self[method(*args, **kwargs) if _partial else partial(
                method, *args, **kwargs) if args or kwargs else method]

        setattr(cls, attr, getattr(cls, attr, wraps(method)(_macro)))


composables = []
for attr, method in [('__matmul__', groupby), ('__div__', map), (
        '__truediv__', map), ('__floordiv__', filter), ('__mod__', reduce)]:
    composables += macro(attr, method, Composer, True) or [method.func]


def _right_(attr):
    _attr_ = """__{}__""".format(attr)

    def caller(self, other):
        if self:
            return getattr(type(self)()[other], _attr_)(self)
        return self[other]

    return wraps(getattr(Composer, _attr_))(caller)


s = "__{}{}__".format
for attr in [
        'add', 'sub', 'mul', 'matmul', 'div', 'truediv', 'floordiv', 'mod',
        'lshift', 'rshift', 'and', 'xor', 'or', 'pow'
]:
    setattr(Composer, s('i', attr), getattr(Composer, s('', attr)))
    setattr(Composer, s('r', attr), _right_(attr))

for attr, method in [['call'] * 2, ['do', 'lshift'], ['pipe', 'getitem']]:
    setattr(Composer, attr, getattr(Composer, s('', method)))

composables += attrgetter('keyfilter', 'keymap', 'valfilter', 'valmap',
                          'itemfilter', 'itemmap')(import_module('toolz'))

for imports in ('toolz', 'operator', 'six.moves.builtins', 'itertools'):
    for attr, method in compose(iteritems,
                                valfilter(callable),
                                keyfilter(compose(str.lower, first)), vars,
                                import_module)(imports):

        if attr[0].islower():
            if method in (methodcaller, itemgetter, attrgetter, not_, truth,
                          abs, invert, neg, pos, index):
                method = partial(method)
            elif method in (flip, ) or imports == 'toolz':
                method = functor(method)
            elif imports.endswith('builtins') and method not in (
                    hasattr, getattr, isinstance, issubclass, setattr):
                pass
            else:
                method = flipped(method)

            macro(attr, method, Composer, method in composables)


class Lambda(Composer):
    def __getitem__(self, objects):
        self = super(Lambda, self).__getitem__()
        return any(
            self.function.append(object)
            for object in isiterable(objects) and objects or
            (objects, )) or self


_y, _x, _f, x_, _xx, _h = tuple(
    type('_{}_'.format(function.__name__), (function, ),
         {})(function=Compose([function]))
    for function in (Juxtaposition, Composer, Reversed, Flipped, Starred,
                     Lambda))

del attr, doc, func, imports, method, s


def load_ipython_extension(ip):
    """%%fidget magic that displays code cells as markdown, then runs the cell.
    """
    from IPython import display
    ip.register_magic_function(
        _x[lambda l, cell: cell][_x << display.Markdown >>
                                 display.display][ip.run_cell][None], 'cell',
        'fidget')
