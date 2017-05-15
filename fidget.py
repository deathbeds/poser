# coding: utf-8

# > `fidget` uses the python data model to compose higher-order functions.
# 
# ---

from copy import copy
from functools import wraps, total_ordering, partial
from importlib import import_module
from six import iteritems, PY2
from toolz.curried import (isiterable, filter, first, flip, complement, map,
                           identity, valfilter, merge, groupby, concat, get,
                           keyfilter, compose, reduce)
from six.moves.builtins import hasattr, getattr, isinstance, issubclass, setattr
from operator import (methodcaller, itemgetter, attrgetter, not_, truth, abs,
                      invert, neg, pos, index, eq)

__all__ = ('function', 'flips', 'stars', 'does', 'each', 'when',
           'call')  # noqa: F822


@total_ordering
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
    __slots__ = ('function', )

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


class starred(functor):
    def __call__(self, *args, **kwargs):
        arguments = groupby(flip(isinstance)(dict),
                            args) if all(map(isiterable, args)) else {
                                False: [[args]]
                            }
        return super(starred, self).__call__(
            *concat(get(False, arguments, tuple())), **merge(
                kwargs, *get(True, arguments, [{}])))


class condition(functor):
    __slots__ = ('condition', 'function')


class ifthen(condition):
    def __call__(self, *args, **kwargs):
        return functor(self.condition)(*args, **kwargs) and super(
            ifthen, self).__call__(*args, **kwargs)


class default(condition):
    def __call__(self, *args, **kwargs):
        return super(default, self).__call__(
            *args, **kwargs) or functor(self.condition)(*args, **kwargs)


class step(condition):
    def __call__(self, *args, **kwargs):
        result = functor(self.condition)(*args, **kwargs)
        return result and super(step, self).__call__(result)


class excepts(functor):
    __slots__ = ('exceptions', 'function')

    def __call__(self, *args, **kwargs):
        try:
            return super(excepts, self).__call__(*args, **kwargs)
        except self.exceptions as e:
            return exception(e)


class exception(State):
    __slots__ = ('exception', )

    def __bool__(self):
        return not self.exception

    def __repr__(self):
        return repr(self.exception)


if not PY2:

    def doc(self):
        return getattr(self.function, '__doc__', '')

    for func in (functor, flipped, do, starred, ifthen, default, excepts):
        setattr(func, '__doc__', property(doc))


class call(State):
    __slots__ = ('args', 'kwargs')

    def __init__(self, *args, **kwargs):
        super(call, self).__init__(args, kwargs)

    def __call__(self, function=functor):
        return partial(functor(function), *self.args, **self.kwargs)


class Append(State):
    __slots__ = ('function', )

    def __getitem__(self, object=slice(None)):
        if object is call:
            return abs(self)

        if isinstance(object, call):
            return object(self)()

        return object != slice(None) and self.append(object) or self

    def __repr__(self):
        return repr(self.function)

    def append(self, object):
        self.function += [object]


class Functions(Append):
    def __init__(self, function=None, *args):
        if function is None:
            function = list()

        if not isiterable(function) or isinstance(function, (str, )):
            function = [function]

        super(Functions, self).__init__(
            (isinstance(function, dict) and compose(list, iteritems) or
             identity)(function), *args)

    def __delitem__(self, object):
        self.function = list(fn for fn in self if fn != object)
        return self

    def __setitem__(self, attr, object):
        self.function = list(object if fn == attr else fn for fn in self)
        return self

    def __iter__(self):
        for function in self.function:
            yield function

    def __reversed__(self):
        self.function = type(self.function)(reversed(self.function))
        return self


class Composite(Functions):
    def __contains__(self, object):
        return any(object == function for function in self)

    def _dispatch_(self, function):
        return (isinstance(function, (dict, set, list, tuple)) and Juxtapose or
                functor)(function)


class Juxtapose(Composite):
    def __call__(self, *args, **kwargs):
        return type(self.function)(call(*args)(self._dispatch_(function))(
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
        return hash(tuple(reversed(self.function)))


class Composable(Compose):
    def __init__(self, function):
        super(Composable, self).__init__([function])


class Partial(Functions):
    __slots__ = ('args', 'keywords', 'function')
    _decorate_, _composite_ = map(staticmethod, (functor, Compose))

    def __init__(self, *args, **kwargs):
        function = kwargs.pop('function', self._composite_())
        if not callable(function):
            function = self._composite_(function)
        super(Partial, self).__init__(args, kwargs, function)

    @property
    def __call__(self):
        return call(*self.args,
                    **self.keywords)(self._decorate_(self.function))

    def append(self, object):
        self.function.function += [object]


class Composer(Partial):
    @property
    def _factory_(self):
        return type(self).__name__.startswith('_') and type(
            self).__name__.endswith('_')

    def __getitem__(self, object=slice(None), *args, **kwargs):
        if self._factory_:
            self = self.function()

        if isinstance(object, (int, slice)):
            object, self = self.function.function[object], copy(self)
            self.function = self._composite_(object)
            return self

        return super(Composer, self).__getitem__(
            (args or kwargs) and call(*args, **kwargs)(object) or object)


class Function(Composer):
    def __xor__(self, object):
        self, _isinstance = self[:], flip(isinstance)  # noqa: F823
        if isiterable(object):
            if all(map(_isinstance(type), object)) and all(
                    map(flip(issubclass)(BaseException), object)):
                self.function = Compose(excepts(object, self.function))
                return self

            if all(map(_isinstance(BaseException), object)):
                object = tuple(map(type, object))

            if all(map(_isinstance(type), object)):
                object = _isinstance(object)

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
        return Does()[object] if self._factory_ else self[do(object)]

    __pow__ = __xor__
    __invert__ = Functions.__reversed__
    __mul__ = __add__ = __rshift__ = __sub__ = Composer.__getitem__


for name, func in (('Flips', flipped), ('Stars', starred), ('Does', do),
                   ('Each', map), ('When', filter)):
    locals()[name] = type(name, (Function, ),
                          {'_decorate_': staticmethod(func)})


class Juxt(Composer):
    _composite_ = staticmethod(Juxtapose)


class Composition(Function):
    _composite_ = staticmethod(ComposeLeft)


def macro(attr, method, composable=False, cls=Function, force=False):
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
    composables += macro(attr, method, True) or [method.func]


def _right_(attr):
    _attr_ = """__{}__""".format(attr)

    def caller(self, object):
        object = type(self)()[object]
        return getattr(object, _attr_)(self[:]) or object

    return wraps(getattr(Function, _attr_))(caller)


s = "__{}{}__".format
for attr in [
        'add', 'sub', 'mul', 'matmul', 'div', 'truediv', 'floordiv', 'mod',
        'lshift', 'rshift', 'and', 'xor', 'or', 'pow'
]:
    setattr(Function, s('i', attr), getattr(Function, s('', attr)))
    setattr(Function, s('r', attr), _right_(attr))

for attr, method in [['call'] * 2, ['do', 'lshift'], ['pipe', 'getitem']]:
    setattr(Function, attr, getattr(Function, s('', method)))

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

            macro(attr, method, method in composables)

for func in (Function, Juxt, Flips, Stars, Does, Each, When, Composition):
    locals()[func.__name__.lower()] = type('_{}_'.format(func.__name__), (
        func, ), {})(function=Compose([func]))

del attr, doc, func, imports, method, s, name


def load_ipython_extension(ip):
    """%%fidget magic that displays code cells as markdown, then runs the cell.
    """
    from IPython import display
    ip.register_magic_function(function[lambda l, cell: cell][does[
        display.Markdown][display.display]][ip.run_cell][None], 'cell',
                               'fidget')
