
# coding: utf-8

__test__ = dict(
    gold="""
    >>> g = a**str*[[str.upper]]|a**int*range|a**float*type
    >>> f = a/g*list
    >>> assert f(['keep it 💯', 10, 10.]) == [[["KEEP IT 💯"]], range(10), float]
    >>> assert copy(g).type()({}) is Ø""",
    simple=""">>> assert simple[range](10) == range(10)
    >>> assert simple[range][type](10) is range
    >>> assert simple[range][type][type](10) is type

    Begin a composition with a decorator.
    >>> @simple.append
    ... def thang(x):
    ...     return range(x)
    >>> assert thang[len](10) is 10""",
    juxt="""
    >>> assert juxt[range](10) == [range(10)]
    >>> assert juxt[range][type](10) == [range(10), int]
    >>> assert juxt[range][type][str](10) == [range(10), int, '10']

    Create a juxtaposable object `model`.
    >>> @juxt.append
    ... def model(x): return range(x)

    Append more functions to the `model`
    >>> @model.append
    ... def _(x): return type(x)

    >>> assert isinstance(Juxtaposition({})(), dict) and isinstance(Juxtaposition([])(), list) and isinstance(Juxtaposition(tuple())(), tuple)""")


# # Complex Composite Functions
#
# Complex composite functions have real and imaginary parts that may except specific errors.
#
# ## Composition
#
# Composite functions use Python syntax to append callable objects to compositions and juxtapositions.
#
# ### Operator

from functools import partialmethod, wraps, partial

import operator
from collections import Sized, Mapping
from toolz import isiterable, excepts, identity, complement, concat, reduce, groupby
import sys
import inspect

from copy import copy

dunder = '__{}__'.format

__all__ = 'a', 'an', 'the', 'flip', 'parallel', 'star', 'do', 'preview', 'x', 'op', 'juxt', 'cache', 'store', 'Composition',  # 'Juxtaposition'


# Composing function strictly through the Python datamodel.

def call(object, *tuple, exceptions=None, **dict):
    """(a**callable)|identity(a**callable)|identity"""
    if callable(object):
        if Exception:
            object = excepts(Exception, object, identity)
        return object(*tuple, **dict)
    return object


def null(*args, **kwargs): return args[0] if args else None


class partial_object(partial):
    def __call__(
        self,
        object): return self.func(
        object,
        *self.args,
        **self.keywords)


class State:
    def __hash__(self): return hash(map(hash, self))

    def __getstate__(self):
        return tuple(getattr(self, slot, None) for slot in self.__slots__)

    def __setstate__(self, tuple):
        return list(setattr(self, str, object)
                    for str, object in zip(self.__slots__, tuple)) and self

    @property
    def __name__(self): return type(self).__name__
    __signature__ = inspect.signature(null)


class flip(State):
    __slots__ = 'callable',

    def __init__(self, callable): self.callable = callable

    def __call__(self, *
                 tuple, **dict): return call(self.callable, *
                                             reversed(tuple), **dict)


class Ø(BaseException):
    def __bool__(self): return False


Null = Ø


class Outer(State):
    __slots__ = 'callable',

    def __init__(self, object=None):
        self.callable = list() if object is None else object

    def __iter__(self):
        callable = self.callable
        # Juxt is defined way later on unfortunately, but this is the most
        # efficient place.
        yield from map(
            Juxt, dict.items(callable) if isinstance(callable, dict) else callable)

    def __call__(self, *tuple, **dict):
        for callable in self:
            tuple, dict = (call(callable, *tuple, **dict),), {}
        return null(*tuple, **dict)

    def __len__(self): return len(
        self.callable) if isinstance(
        self.callable,
        Sized) else 0

    def __repr__(self): return repr(self.callable)

    def __bool__(self): return bool(len(self))


class Inner(Outer):
    def __iter__(self):
        yield from self.callable and super().__iter__() or (True,)

    def __call__(self, *tuple, **dict):
        object = super().__call__(*tuple, **dict)
        return object if object else Null(self.callable)


def _is_isinstance(object):
    """Prepare types and types as isinstance functions"""
    if isinstance(object, type) and object is not bool:
        object = object,
    if isinstance(object, tuple):
        object = partial_object(isinstance, object)
    return object


class Pose(State):
    __slots__ = 'inner', 'outer', 'exceptions'
    _repr_token_ = '^'

    def __init__(self, inner=None, outer=None, *, exceptions=None):
        inner = _is_isinstance(inner)
        if inner is not None and not isiterable(
                inner) or isinstance(inner, Pose):
            inner = [inner]

        for name in ('inner', 'outer'):
            object = locals().get(name)
            if object is not None and not isiterable(
                    object) or isinstance(object, Pose):
                locals()[name] = [object]

        self.inner, self.outer = Inner(inner), Outer(outer)
        self.exceptions = exceptions

    def __bool__(self): return bool(self.outer)

    def __repr__(self): return self._repr_token_.join(
        map(repr, (self.inner, self.outer)))

    def __call__(self, *tuple, **dict):
        return call(
            self.outer,
            *tuple,
            **dict,
            exceptions=dict.pop(
                'exceptions',
                self.exceptions))


class Juxt(Pose):
    """Juxtapose arguments cross callables."""
    def __new__(cls, object=None, **kwargs):
        if object is None:
            object = list()
        if callable(object):
            return object
        if isiterable(object) and not isinstance(object, str):
            self = super().__new__(cls)
            return self.__init__(**kwargs) or self
        return object

    def __init__(self,
                 outer=None,
                 **kwargs): super().__init__(outer=outer,
                                             **kwargs)

    def __call__(self, *args, **kwargs):
        iter = (call(callable, *args, **kwargs) for callable in self.outer)
        return type(
            self.outer.callable)(iter) if isinstance(
            self.outer.callable,
            Sized) else iter


# # Composites

class Pro(Pose):
    """Propose a non-null inner condition then evaluate the outer function."""

    def __call__(self, *tuple, **dict):
        object = self.inner(*tuple, **dict)
        if self.outer:
            return object if isinstance(
                object, Null) else super().__call__(
                *tuple, **dict)

        # If there is not outer function return a boolean.
        return not isinstance(object, Null)


class Ex(Pose):
    """Pipe non-null inner return values to the outer callable."""
    _repr_token_ = '&'

    def __call__(self, *tuple, **dict):
        object = self.inner(*tuple, **dict)
        if object is True:
            object = null(*tuple, **dict)
        return object if isinstance(object, Null) else super().__call__(object)


class Im(Pose):
    """If the inner function is Null evaluate the outer function."""
    _repr_token_ = '|'

    def __call__(self, *tuple, **dict):
        object = self.inner(*tuple, **dict)
        if object is True:
            object = null(*tuple, **dict)
        return super().__call__(*tuple, **dict) if isinstance(object, Null) else object


def _partial_(self, object):
    tuple, dict = getattr(self, 'args', []), getattr(self, 'kwargs', {})
    return partial(object, *tuple, **dict) if tuple or dict else object


class Conditions:
    # The [*]positions are defined later.
    def __pow__(self, object):
        # Sugar for isinstance implementations.
        return Proposition(inner=([] if isinstance(self, Lambda) else [
                           self]) + [_is_isinstance(object)])

    def __and__(self, object):
        """Evaluate object if the current proposition is True.

        >>> f = a[bool] & range
        >>> assert not f(0) and f(10) == range(10)
        """
        return Exposition(
            inner=[] if isinstance(
                self,
                Lambda) else [self],
            outer=[object])

    def __or__(self, object):
        """Evaluate object if the current proposition is False.

        >>> f = a[bool] | range
        >>> assert f(10) is 10 and f(0) == range(0)
        """
        return Imposition(
            inner=[] if isinstance(
                self,
                Lambda) else [self],
            outer=[object])

    def __xor__(self, object):
        """Evaluate a proposition returning exceptions in object.

        >>> f = a * range ^ TypeError
        >>> assert f(10) == range(10) and isinstance(f('string'), TypeError)
        """
        return setattr(self, 'exceptions', object) or self

    then = __and__
    ifnot = __or__
    instance = ifthen = __pow__
    excepts = __xor__


class __getattr__(object):
    def __init__(self, object, callable=None, parent=None):
        self.object = object
        self.callable = callable
        self.parent = parent

    def __getattr__(self, object):
        parent = self.callable
        # Convert the attribute to a callable.
        if self.callable:
            object = getattr(self.callable, object)

        if object in sys.modules:
            object = sys.modules.get(object)

        elif isinstance(object, str):
            for module in map(__import__, Attributes.shortcuts):
                if hasattr(module, object):
                    object = getattr(module, object)
                    break
            else:
                raise AttributeError(object)

        # Decorate the discovered attribute with the correct partials or call.
        wrapper = False

        for decorator, set in Attributes.decorators.items():
            if object in set:
                object = partial(decorator, object)
                break
        else:
            if callable(object) and not isinstance(object, type):
                wrapper = wraps(object)
                object = partial(
                    isinstance(
                        parent,
                        type) and partial_object or partial,
                    object)

        # Wrap the new object for interaction
        object = __getattr__(self.object, object, parent)
        return wrapper(object) if wrapper else object

    def __call__(self, *tuple, **dict):
        object = self.callable
        return self.object.append(object(*tuple,
                                         **dict) if isinstance(object,
                                                               partial) else partial(object,
                                                                                     *tuple,
                                                                                     **dict))

    def __repr__(self):
        return repr(
            isinstance(
                self.callable,
                partial) and self.callable.args and self.callable.args[0] or self.callable)

    def __dir__(self):
        if not self.callable or isinstance(self, Attributes):
            base = (
                list(filter(partial_object(complement(str.__contains__), '.'), sys.modules.keys()))
                + list(concat(dir(__import__(module)) for module in Attributes.shortcuts)))
        else:
            base = dir(self.callable)
        return super().__dir__() + base


class Attributes:
    """
    >>> assert not any(x in dir(x) for x in sys.modules if not '.' in x)
    >>> assert all(x in dir(a) for x in sys.modules if not '.' in x)
    """
    shortcuts = 'statistics', 'toolz', 'requests', 'builtins', 'json', 'pickle', 'io', 'collections', 'itertools', 'functools', 'pathlib', 'importlib', 'inspect', 'operator'
    decorators = dict()

    def __getattr__(self, attr):
        """Access attributes from sys.modules or self.shortcuts"""
        return __getattr__(self).__getattr__(attr)

    def __dir__(self): return dir(__getattr__(self))


Attributes.decorators[partial_object] = [__import__('fnmatch').fnmatch]
Attributes.decorators[call] = operator.attrgetter(
    'attrgetter', 'itemgetter', 'methodcaller')(operator)
Attributes.decorators[partial_object] += [item for item in vars(
    operator).values() if item not in Attributes.decorators[call]]


def _append_(self, object): return self.append(object)


class Append:
    def append(self, object): return self.outer.callable.append(object) or self
    __getitem__ = _append_


class Symbols:
    """Operations that operator on containers.

    >>> assert a@range == a.groupby(range)
    >>> assert a/range == a.map(range)
    >>> assert a//range == a.filter(range)
    >>> assert a%range == a.reduce(range)
    >>> assert copy(a%range) == a.reduce(range)
    """

    def _left(self, callable, object=None, partial=partial_object):
        return self.append(
            callable if object is None else partial(
                callable, object))

    def _right(right, attr, left):
        return getattr(Symbols._left(Proposition(), left), dunder(attr))(right)

    __truediv__ = map = partialmethod(_left, map, partial=partial)
    __floordiv__ = filter = partialmethod(_left, filter, partial=partial)
    __mod__ = reduce = partialmethod(_left, reduce, partial=partial)
    __matmul__ = groupby = partialmethod(_left, groupby, partial=partial)
    __add__ = __mul__ = __sub__ = __rshift__ = _append_

    def __lshift__(self, object): return self.append(Do(object))
    do = __lshift__


for attr in ['add', 'sub', 'mul', 'truediv', 'getitem', 'rshift', 'lshift']:
    setattr(
        Symbols,
        '__r' +
        dunder(attr).lstrip('__'),
        partialmethod(
            Symbols._right,
            attr))


# # Juxtapositions

class Position(Append, Conditions, Attributes, Symbols):
    ...


class Proposition(Pro, Position):
    """Evaluate the outer callable if the inner callable is ~Null."""


class Exposition(Ex, Position):
    """Pass ~Null inner function return values as input to the outer function."""


class Imposition(Im, Position):
    """Evaluate the other outer function is the inner function is Null."""


class Juxtaposition(Juxt, Position):
    """Pass arguments to all callables in all iterables."""


IfThen, IfNot = Exposition, Imposition


class Lambda:
    def append(self, object): return self().append(_partial_(self, object))

    def __bool__(self): return False


class Composition(Lambda, Proposition):
    __slots__ = Pose.__slots__ + ('args', 'kwargs')


class Simple(Composition):
    def __call__(self, *tuple, **dict):
        if tuple or dict:
            self = copy(self)
            self.args, self.kwargs = tuple, dict
            return self
        return super().__call__(dict.get('inner', None), *tuple, **dict)


composite = compositon = Composition(outer=[Proposition])
a = an = the = simple = λ = Simple(outer=[Proposition])
juxt = juxtaposition = Simple(outer=[Juxtaposition])


binop = 'add', 'sub', 'mul', 'truediv', 'floordiv', 'mod', 'getitem'
boolop = 'gt', 'ge', 'le', 'lt', 'eq', 'ne'
nop = 'abs', 'pos', 'neg', 'pow'


class Operate(Proposition):
    __wrapped__ = None
    __annotations__ = {}

    def __init__(self, *args, **kwargs):
        super().__init__(kwargs.pop('inner', Juxtaposition()), *args, **kwargs)

    def _left(self, callable, arg=None, partial=partial_object):
        return self.append(partial(callable, arg))

    def _right(self, callable, left):
        return Operate._left(Operate(), callable, left, partial=partial)

    def _bool(self, callable, *args):
        return Operate(inner=[partial_object(callable, *args)], outer=[self])

    def __getattr__(self, object):
        if object == dunder('qualname'):
            return '.'.join((__name__, type(self).__name__))
        return self._left(getattr, object, )


for attr in binop:
    op, rop = getattr(operator, attr), '__r' + dunder(attr).lstrip('__')
    setattr(Operate, dunder(attr), partialmethod(Operate._left, op))
    setattr(Operate, rop, partialmethod(Operate._right, op))

list(
    setattr(
        Operate,
        dunder(attr),
        partialmethod(
            Operate._bool,
            getattr(
                operator,
                attr))) for attr in boolop)
list(
    setattr(
        Operate,
        dunder(attr),
        partialmethod(
            Operate._left,
            getattr(
                operator,
                attr))) for attr in nop)
pass


class Operation(Lambda, Operate):
    def append(self, object): return self().append(object)


x = op = Operation(outer=[Operate])


class Star(Proposition):
    _repr_token_ = "*"

    def __call__(self, object, *tuple, **dict):
        if isinstance(object, Mapping):
            return super().__call__(*tuple, **(dict.update(object) or dict))
        return super().__call__(*object, *tuple, **dict)


star = Simple(outer=[Star])


class Do(Proposition):
    _repr_token_ = '>>'

    def __call__(self, *args, **kwargs): return (super().__call__(* \
                 args, **kwargs), null(*args, **kwargs))[-1]


do = Simple(outer=[Do])


class Preview(Proposition):
    """Eager proposition evaluation."""

    def __repr__(self): return repr(self())


preview = Simple(outer=[Preview])


class parallel(Proposition):
    """An embarassingly parallel proposition.

    >>> import joblib
    >>> def g(x): return x+10
    >>> assert parallel(4).range().map(x+10)(100)
    >>> assert parallel(4).range().map(a[range])(100)
    """
    _repr_token_ = '||'

    def __init__(self, jobs=4, *tuple, **dict):
        self.jobs = jobs
        super().__init__(*tuple, **dict)

    def map(self, object): return super().map(
        __import__('joblib').delayed(object))

    def __call__(self, *tuple, **dict):
        return __import__('joblib').Parallel(
            self.jobs)(
            super().__call__(
                *tuple, **dict))

    __truediv__ = map


class store(dict):
    @property
    def __self__(self): return self.__call__.__self__

    def __init__(self, callable=None, *tuple, **dict):
        self.callable = Proposition(
            *tuple, **dict) if callable is None else callable
        super().__init__(*tuple, **dict)

    def __call__(self, *tuple, **dict):
        self[tuple[0]] = self.callable(*tuple, **dict)
        return self[tuple[0]]


class cache(store):
    def __call__(self, *tuple, **dict):
        if tuple[0] not in self:
            return super().__call__(*tuple, **dict)
        return self[tuple[0]]


if __name__ == '__main__' and 'runtime' in sys.argv[-1]:
    print(__import__('doctest').testmod())


# # Developer

if __name__ == '__main__':
    if 'runtime' in sys.argv[-1]:
        get_ipython().system(
            'jupyter nbconvert --to python --TemplateExporter.exclude_input_prompt=True composites.ipynb')
        # Juxtaposition still wont work
        get_ipython().system('pyreverse -o png -pcomposites -fALL composites')
        get_ipython().system('autopep8 --in-place --aggressive --aggressive composites.py')
        get_ipython().system('flake8 composites.py')
    else:
        print('run from cli')
