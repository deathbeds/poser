
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

__all__ = 'a', 'an', 'the', 'simple', 'flip', 'parallel', 'star', 'do', 'preview', 'x', 'op', 'juxt', 'cache', 'store', 'Ø', 'Composition', 'Operation', 'Juxtaposition', 'Proposition', 'Exposition', 'Imposition'


binop = 'add', 'sub', 'mul', 'truediv', 'floordiv', 'mod', 'lshift', 'rshift'
boolop = 'gt', 'ge', 'le', 'lt', 'eq', 'ne'
nop = 'abs', 'pos', 'neg', 'pow'


# Composing function strictly through the Python datamodel.

def call(object, *tuple, Exception=None, **dict):
    """Call the object with an argument tuple, a keyword dict, and E

    >>> assert call(10) is 10
    >>> assert call(range, 10, 20) == range(10, 20)"""
    return (
        excepts(Exception, object, identity) if Exception else object
    )(*tuple, **dict) if callable(object) else object


def null(*tuple, **dict):
    """A null/identity function that returns the first arguments if it exists.

    >>> assert not null(**dict(foo=42))
    >>> assert null(10, 20, dict(foo=42)) is 10
    """
    return tuple[0] if tuple else None


class Self(partial):
    """Supply partial arguments to objects.

    >>> assert Self(str.replace, 'a', 'b')('abc') == 'bbc'
    """

    def __call__(
        self,
        object): return self.func(
        object,
        *self.args,
        **self.keywords)


class State:
    """State attributes for pickling and copying propositions."""

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
    """Flip the argument of a callable.

    >>> assert flip(range)(20, 10) == range(10, 20)
    """
    __slots__ = 'callable',

    def __init__(self, callable): self.callable = callable

    def __call__(self, *
                 tuple, **dict): return call(self.callable, *
                                             reversed(tuple), **dict)


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


class Ø(BaseException):
    """An Inner callable may return a Ø Exception."""

    def __bool__(self): return False


Null = Ø


def _is_isinstance(object):
    """Prepare types and types as isinstance functions.

    >>> assert _is_isinstance(int)(10) and not _is_isinstance(int)('10')
    """
    if isinstance(object, type) and object is not bool:
        object = object,
    if isinstance(object, tuple):
        object = Self(isinstance, object)
    return object


class Pose(State):
    """Pose a complex function with Inner and Outer callable. The Outer callable may accept exceptions.
    Pose is combined with the prefixes Pro, Ex, Im, and Juxt to evaluate inner call methods.
    """
    __slots__ = 'Inner', 'Outer', 'Exception'
    _repr_token_ = '^'

    def __init__(self, inner=None, outer=None, *, exception=None):
        inner = _is_isinstance(inner)
        if inner is not None and not isiterable(
                inner) or isinstance(inner, Pose):
            inner = [inner]

        for name in ('inner', 'outer'):
            object = locals().get(name)
            if object is not None and not isiterable(
                    object) or isinstance(object, Pose):
                locals()[name] = [object]

        self.Inner, self.Outer = Inner(inner), Outer(outer)
        self.Exception = exception

    def __bool__(self): return bool(self.Outer)

    def __repr__(self): return self._repr_token_.join(
        map(repr, (self.Inner, self.Outer)))

    def __call__(self, *tuple, **dict):
        return call(
            self.Outer,
            *tuple,
            **dict,
            Exception=dict.pop(
                'Exception',
                self.Exception))


class Juxt(Pose):
    """Juxtapose arguments cross callables."""

    def __new__(self, object=None, **dict):
        """Juxtapose is used generically to iterate through Sequences and Mappings. The
        new method returns callables and initializes everything else.  When called, Juxt
        will return an object with the same type as Juxt.Outer.
        """
        if callable(object):
            return object
        if object is None:
            object = list()
        if isiterable(object) and not isinstance(object, str):
            self = super().__new__(self)
            return self.__init__(**dict) or self
        return object

    def __init__(self, outer=None, **
                 dict): super().__init__(outer=outer, **dict)

    def __call__(self, *args, **kwargs):
        iter = (call(callable, *args, **kwargs) for callable in self.Outer)
        return type(
            self.Outer.callable)(iter) if isinstance(
            self.Outer.callable,
            Sized) else iter


# # Composites

class Pro(Pose):
    """Propose a non-null inner condition then evaluate the outer function."""

    def __call__(self, *tuple, **dict):
        object = self.Inner(*tuple, **dict)
        if self.Outer:
            return object if isinstance(
                object, Null) else super().__call__(
                *tuple, **dict)

        # If there is not outer function return a boolean.
        return not isinstance(object, Null)


class Ex(Pose):
    """Pipe non-null inner return values to the outer callable."""
    _repr_token_ = '&'

    def __call__(self, *tuple, **dict):
        object = self.Inner(*tuple, **dict)
        if object is True:
            object = null(*tuple, **dict)
        return object if isinstance(object, Null) else super().__call__(object)


class Im(Pose):
    """If the inner function is Null evaluate the outer function."""
    _repr_token_ = '|'

    def __call__(self, *tuple, **dict):
        object = self.Inner(*tuple, **dict)
        if object is True:
            object = null(*tuple, **dict)
        return super().__call__(*tuple, **dict) if isinstance(object, Null) else object


def _inner_(self): return [] if isinstance(self, Lambda) else [self]


class Conditions:
    # Lambda initializes propositions.
    # The [*]positions are defined later.
    def __pow__(self, object): return Proposition(
        inner=_inner_(self) + [_is_isinstance(object)])

    def __and__(
        self,
        object): return Exposition(
        inner=_inner_(self),
        outer=[object])

    def __or__(
        self,
        object): return Imposition(
        inner=_inner_(self),
        outer=[object])

    def __xor__(
        self,
        object): return setattr(
        self,
        'exceptions',
        object) or self

    then = __and__
    ifnot = __or__
    instance = ifthen = __pow__
    excepts = __xor__


class __getattr__(object):
    def __init__(self, object, callable=None, parent=None):
        self.object, self.callable, self.parent = object, callable, parent

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
                        type) and Self or partial,
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
                list(filter(Self(complement(str.__contains__), '.'), sys.modules.keys()))
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


Attributes.decorators[Self] = [__import__('fnmatch').fnmatch]
Attributes.decorators[call] = operator.attrgetter(
    'attrgetter', 'itemgetter', 'methodcaller')(operator)
Attributes.decorators[Self] += [item for item in vars(
    operator).values() if item not in Attributes.decorators[call]]


class Append:
    def append(self, object): return self.Outer.callable.append(object) or self

    def __getitem__(self, object): return self.append(object)


class Symbols:
    """Operations that operator on containers.

    >>> assert a@range == a.groupby(range)
    >>> assert a/range == a.map(range)
    >>> assert a//range == a.filter(range)
    >>> assert a%range == a.reduce(range)
    >>> assert copy(a%range) == a.reduce(range)
    """

    def _left(self, callable, object=None, partial=Self):
        return self.append(
            callable if object is None else partial(
                callable, object))

    def _right(right, attr, left):
        return getattr(Symbols._left(Proposition(), left), dunder(attr))(right)

    __truediv__ = map = partialmethod(_left, map, partial=partial)
    __floordiv__ = filter = partialmethod(_left, filter, partial=partial)
    __mod__ = reduce = partialmethod(_left, reduce, partial=partial)
    __matmul__ = groupby = partialmethod(_left, groupby, partial=partial)
    __add__ = __mul__ = __sub__ = __rshift__ = Append.__getitem__

    def __lshift__(self, object): return self.append(Do(object))
    do = __lshift__


list(
    setattr(
        Symbols,
        '__r' +
        dunder(attr).lstrip('__'),
        partialmethod(
            Symbols._right,
            attr)) for attr in binop)


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
    def append(self, object):
        tuple, dict = getattr(self, 'args', []), getattr(self, 'kwargs', {})
        return self().append(partial(object, *tuple, **dict) if tuple or dict else object)

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


class Operate(Proposition):
    __wrapped__ = None
    __annotations__ = {}

    def __init__(self, *args, **kwargs):
        super().__init__(None, *args, **kwargs)
        self.__qualname__ = '.'.join((__name__, type(self).__name__))

    def _left(self, callable, arg=None, partial=Self):
        return self.append(partial(callable, arg))

    def _right(self, callable, left):
        return Operate._left(Operate(), callable, left, partial=partial)

    def _bool(self, callable, *args):
        return Operate(inner=[Self(callable, *args)], outer=[self])


for attr in binop + ('getitem',):
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
    __qualname__ = 'Operation'


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

    def __call__(self, *tuple, **dict):
        super().__call__(*tuple, **dict)
        return null(*tuple, **dict)


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


# # Developer

if __name__ == '__main__':
    if 'runtime' in sys.argv[-1]:
        from IPython import get_ipython, display
        get_ipython().system(
            'jupyter nbconvert --to python --TemplateExporter.exclude_input_prompt=True composites.ipynb')
        # Juxtaposition still wont work
        get_ipython().system('python -m pydoc -w composites')
        get_ipython().system('pyreverse -o png -pcomposites -fALL composites')
        display.display(
            display.Image('classes_composites.png'),
            display.IFrame(
                'composites.html',
                height=600,
                width=800))
        __import__('doctest').testmod()
        get_ipython().system('ipython -m doctest  composites.py')
        get_ipython().system('python -m pydoc -w composites')

        get_ipython().system('autopep8 --in-place --aggressive --aggressive composites.py')
        get_ipython().system('flake8 composites.py --ignore E501,E704,W503')
    else:
        print('run from cli')
