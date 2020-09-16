import abc
import ast
import builtins
import fnmatch
import functools
import importlib
import inspect
import itertools
import operator
import pathlib
import re
import sys
import typing
from functools import partialmethod

import toolz
from toolz.curried import *

from . import util
from .util import (
    Ø,
    _evaluate,
    _sympy,
    arunner,
    attribute,
    context,
    dump,
    filter,
    flatten,
    fold,
    istype,
    map,
    raises,
    normal_slice,
    setterattr,
    setteritem,
    glob,
)

__all__ = "λ", "Λ", "Pose", "Self", "star", "juxt", "Ø", "Null", "a", "an", "the"


# `λ` is an `object` for fluent function composition in `"python"` based on the `toolz` library.
#
# 1.  [__Motivation__](#Motivation)
# 2. [__Source__](#Source)
# 3. [__Tests__](#Tests)
#
# #### Motivation
#
# [Function composition](https://en.wikipedia.org/wiki/Function_composition) is a common task in mathematics and modern programming.
# Object oriented function composition often breaks with conventional representations.
# The `toolz` library provides a set of functional programming objects to `compose` and `pipe` functions together.
# `compose` and `pipe` are top level composition functions that how two different typographic conventions.
#
# In the `toolz` example, both `f and g` are the same
#
# 1. `compose` mimics a symbollic function composition.
#
#         f = compose(type, len, range)
#
# 1. `pipe` allows a fluent composition.
#
#         g = lambda x: pipe(x, range, len, type)
#         def h(x: int) -> type: return pipe(x, range, len, type)
#
# The typology of the `compose` composition is destructive to the flow of literature because it must be read `reversed`.
# `pipe` on the other hand supplements the narrative providing literate compositions aligned with the direction of the literature.
#
# From a learning perspective, my experience with `poser` & its predecessors have taught me a lot about the pythonic data model.
# `Compose` expresses a near complete symbollic API for function composition.


# #### Source

builtins.Null = builtins.Ø = Null = Ø
# `Compose` augments `toolz.Compose` to provide a fluent & symbollic object for function composition in python.

# ## Function Composition base class.


class juxt(toolz.functoolz.juxt):
    """An overloaded toolz juxtaposition that works with different objects and iterables."""

    _lambdaified = {}

    def __new__(self, funcs=None):
        if funcs is None:
            self = super().__new__(self)
            return self.__init__() or self
        if isinstance(funcs, str):
            funcs = importlib.import_module("poser.poser").Forward(funcs)
        if util._sympy(funcs):
            import sympy

            if not funcs in self._lambdaified:
                self._lambdaified[funcs] = sympy.lambdify(
                    sorted(funcs.free_symbols, key=lambda x: x.name), funcs
                )
            return self._lambdaified[funcs]
        if callable(funcs) or not toolz.isiterable(funcs):
            return funcs
        self = super().__new__(self)
        return self.__init__(funcs) or self

    def __init__(self, object=None):
        self.funcs = object

    def __call__(self, *args, **kwargs):
        if isinstance(self.funcs, __import__("typing").Mapping):
            # Juxtapose a mapping object.
            object = type(self.funcs)()
            for key, value in self.funcs.items():
                if callable(key):
                    key = juxt(key)(*args, **kwargs)
                if callable(value):
                    value = juxt(value)(*args, **kwargs)
                object[key] = value
            return object
        if toolz.isiterable(self.funcs):
            # juxtapose an iterable type that returns the container type
            return type(self.funcs)(
                juxt(x)(*args, **kwargs)
                if (callable(x) or toolz.isiterable(x) or util._sympy(x))
                else x
                for x in self.funcs
            )
        if callable(self.funcs):
            # call it ya can
            return self.funcs(*args, **kwargs)
        return self.funcs


class State:
    def __getstate__(self):
        return tuple(getattr(self, x) for x in self.__slots__)

    def __setstate__(self, state):
        tuple(setattr(self, x, v) for x, v in zip(self.__slots__, state))


class Composition(State, toolz.functoolz.Compose):
    """Extensible function partial composition that excepts Exceptions.

All of position is overlay on the toolz library that supports conventions for functional programming in python.
The toolz documentation are some of the best in python.

>>> Composition([list, range], 5)(10)
[5, 6, 7, 8, 9]

λ or poser is a convenience type for building compositions.

>>> λ(5).partial(range).list()(10)
[5, 6, 7, 8, 9]

"""

    __slots__ = toolz.functoolz.Compose.__slots__ + tuple(
        "args kwargs exceptions".split()
    )

    def __init__(self, funcs=None, *args, **kwargs):
        """Compose stores args and kwargs like a partial."""
        super().__init__(funcs or (I,))
        self.args, self.exceptions, self.kwargs = (
            args,
            kwargs.pop("exceptions", tuple()),
            kwargs,
        )

    def __iter__(self):
        """Iterate over the functions in a Composition.

    >>> Composition([range], 1)
    Composition(<class 'range'>,)
    """
        yield self.first
        yield from self.funcs

    def __bool__(self):
        """Composition is inherited as a metaclass later.

    >>> assert λ() ^ λ

    Composition is True for both types and objects.

    >>> assert bool(Composition) and bool(Composition())
        """
        return not isinstance(self, type)

    def __call__(self, *args, **kwargs):
        """Call a partial composition with args and kwargs.

    >>> Composition([range, enumerate, dict], 5)
    Composition(<class 'range'>, <class 'enumerate'>, <class 'dict'>)
    """
        if not isinstance(self, star):
            # A starred method has already consumed the args and kwargs.
            args, kwargs = (
                getattr(self, "args", ()) + args,
                {**getattr(self, "kwargs", {}), **kwargs},
            )

        # Iterate over the callables in the Composition piping
        # the output across each function
        for callable in self:
            try:
                args, kwargs = (callable(*args, **kwargs),), {}
                object = args[0]
            except self.exceptions as Exception:
                # If an exception is trigger we return a Ø exception
                # that is False and can be used in logical circuits.
                return Ø(Exception)
        return object

    def __len__(self):
        """A Composition's length is measured by the number of functions.

    >>> assert len(λ()) ^ len(λ)
    """
        return (
            0 if isinstance(self, type) else (
                self.funcs and len(self.funcs) or 0) + 1
        )

    def partial(self, object=None, *args, **kwargs):
        """Append a callable-ish object with partial arguments to the composition.

    partial can be chained to compose complex functions..

    >>> assert Composition().partial(range).partial().partial(type)(1) is range

        """
        duplicate = kwargs.pop("duplicate", True)
        if object is not None:
            object = juxt(object)

        if isinstance(self, type) and issubclass(self, Composition):
            # if the object isn't instantiated, instantiate it.
            self = self()
        elif not isinstance(self, Λ):
            if duplicate:
                self = self.duplicate()

        if isinstance(object, Λ):
            # When using a fat lambda because we can't inspect it's attributes.
            pass
        else:
            if object is None:
                # None is a valid answer
                return self
            if isinstance(object, slice):
                # slices are weird
                if normal_slice(object):
                    # normal slices operate on a lists of functions.
                    return type(self)(funcs=list(self)[object])
                else:
                    # slices of callable are crazy!
                    # They provide access to the common filter, map, pipe pattern.
                    # slice(filter, map, pipe)
                    callable(object.start) and self.filter(
                        object.start, duplicate=False)
                    callable(object.stop) and self.map(
                        object.stop, duplicate=False)
                    callable(object.step) and self.partial(
                        object.step, duplicate=False)
                    return self

        # ball the object into a partial when args or kwargs are provided
        if args or kwargs:
            object = toolz.partial(object, *args, **kwargs)

        # update the functions in the composition
        if self.first is I:
            # the first function is identity we know that functions have not been added to the composition.
            # the first attribute is inherited from toolz.
            self.first = object
        else:
            # append the callable object
            self.funcs += (object,)
        return self

    def duplicate(self, *callable, deep=False):
        "Make a duplicate copy of a composition."
        import copy

        f = getattr(copy, deep and "deepcopy" or "copy")(self)
        for callable in callable:
            f += callable
        return f

    _ = dup = duplicate


def istype(object, cls):
    """A convenience function for checking if an object is a type."""
    return isinstance(object, type) and issubclass(object, cls)


def normal_slice(slice):
    return all(
        isinstance(x, (int, type(None)))
        for x in operator.attrgetter(*"start stop step".split())(slice)
    )


class Ø(BaseException):
    def __bool__(self):
        return False


def I(*tuple, **_):
    "A nothing special identity function, does pep8 peph8 me?"
    return tuple[0] if tuple else None


# ### Forward reference to `sys.modules`


class Forward(State, typing.ForwardRef, _root=False):
    """A forward reference implementation that accesses object off of the sys.modules"""

    def __new__(cls, object=None, *args, **kwargs):
        if not isinstance(object, (str, type(None))):
            return object
        if isinstance(object, str):
            try:
                ast.parse(object)
            except SyntaxError:
                return object  # if the forward reference isn't valid code...
        self = super().__new__(cls)
        if object is not None:
            self.__init__(object, *args, **kwargs)
        return self

    def __call__(self, *args, **kwargs):
        object = self._evaluate()
        return object(*args, **kwargs) if callable(object) else object

    def _evaluate(self, globalns=None, localns=None):
        self.__forward_value__ = _evaluate(self.__forward_arg__)
        self.__forward_evaluated__ = True
        return self.__forward_value__

    def __repr__(x):
        return repr(x._evaluate()) if x.__forward_evaluated__ else super().__repr__()

    @property
    def __signature__(x):
        return inspect.signature(x._evaluate())

    @property
    def __doc__(self):
        return inspect.getdoc(self._evaluate())

    def __str__(x):
        return x.__forward_arg__


class Compose(Composition):
    """An extended API for function compositions that allow contiguous functional compositions
    using a fluent and symbollic API.

    """

    def partial(self, object=None, *args, **kwargs):
        if object is Ellipsis:
            kwargs.pop('duplicate', None)
            return self()  # call when we see ellipsis.
        return Composition.partial(self, object, *args, **kwargs)

    def fold(self, object=None, *args, **kwargs):
        return self[fold(object, *args, **kwargs)]

    def method(self, object=None, *args, **kwargs):
        return self[object(*args, **kwargs)]

    def juxt(self, object):
        return self[juxt(object)]

    def star(self, object):
        return self[star[object]]

    @property
    def __doc__(x):
        """The first object is the documentation."""
        return inspect.getdoc(x.first)

    @property
    def __signature__(x):
        """Like the doc, the signature comes from the first function."""
        return inspect.signature(x.first)

    # +, -, >>, and [] append partials.
    # multiple symbols make it possible to append functions
    # in different places in python's order of operations.
    __pos__ = (
        __rshift__
    ) = __sub__ = __rsub__ = __add__ = __radd__ = __getitem__ = partial

    __isub__ = __iadd__ = __irshift__ = functools.partialmethod(
        partial, duplicate=False)
    """Mapping, Filtering, Groupby, and Reduction."""

    def map(self, callable, key=None, *, duplicate=True):
        """Append an overloaded map that works with iterables and mappings."""
        if callable is not None:
            callable = juxt(callable)
        if key is not None:
            key = juxt(key)
        return self.partial(toolz.partial(map, callable, key=key), duplicate=duplicate)

    # * appends a map to the composition
    __mul__ = __rmul__ = map

    __imul__ = functools.partialmethod(map, duplicate=False)

    # / appends a filter to the composition
    def filter(self, callable, key=None, *, duplicate=True):
        """Append an overloaded map that works with iterables and mappings."""
        if callable is not None:
            callable = juxt(callable)
        if key is not None:
            key = juxt(key)
        return self.partial(
            toolz.partial(filter, callable, key=key), duplicate=duplicate
        )

    __truediv__ = __rtruediv__ = filter
    __itruediv__ = functools.partialmethod(map, duplicate=False)

    @functools.wraps(toolz.groupby)
    def groupby(self, callable, *args, **kwargs):
        return self.partial(
            toolz.curried.groupby(juxt(callable)),
            duplicate=kwargs.pop("duplicate", True),
        )

    __matmul__ = __rmatmul__ = groupby
    __imatmul__ = functools.partialmethod(groupby, duplicate=False)

    def reduce(λ, callable, *, duplicate=True):
        return λ.partial(toolz.curried.reduce(juxt(callable)), duplicate=duplicate)

    __mod__ = __rmod__ = reduce
    __imod__ = functools.partialmethod(reduce, duplicate=False)

    """Conditionals."""

    def excepts(self, *Exceptions, duplicate=True):
        if duplicate:
            return Pose(exceptions=Exceptions)[self]
        self.exceptions += Exceptions
        return self

    __xor__ = excepts
    __ixor__ = functools.partialmethod(excepts, duplicate=False)

    def ifthen(self, callable, *, duplicate=True):
        object = IfThen(self)[callable]
        if duplicate:
            return object
        self.first, self.funcs, self.exceptions = object, (), ()
        return self

    __and__ = ifthen
    __iand__ = functools.partialmethod(ifthen, duplicate=False)

    def ifnot(self, callable, *, duplicate=True):
        object = IfNot(self)[callable]
        if duplicate:
            return object
        self.first, self.funcs, self.exceptions = object, (), ()
        return object

    __or__ = ifnot
    __ior__ = functools.partialmethod(ifnot, duplicate=False)

    raises = functools.partialmethod(fold, raises)

    def issubclass(self, object):
        return self[toolz.flip(istype)(object)]

    def isinstance(self, object):
        return self[toolz.flip(isinstance)(object)]

    def __enter__(self):
        exit = Pose[:]
        self = self.partial(context, exit)
        return self, exit

    def __exit__(*e):
        ...

    def iff(self, object, *, duplicate=True):
        # use tuple for bool types. (bool,)
        object = Iff(
            object
            if object is bool
            else self.isinstance(object)
            if isinstance(object, (tuple, type))
            else self[object]
        )
        if duplicate:
            return object
        self.first, self.funcs, self.exceptions = object, (), ()
        return self

    __pow__ = __ipow__ = iff

    def skip(self, bool: bool = True):
        return bool and self.off() or self.on()

    def on(self):
        return self

    def off(self):
        return self[:-1]

    def do(λ, callable, *, duplicate=True):
        return λ.partial(toolz.curried.do(juxt(callable)), duplicate=duplicate)

    __lshift__ = do
    __ilshift__ = functools.partialmethod(do, duplicate=False)

    def complement(self, object=None):
        if object == None:
            return Pose[toolz.complement(self)]
        return self[toolz.complement(object)]

    __neg__ = complement

    def flip(x, object=None):
        return λ[toolz.flip(x)] if object == None else x[toolz.flip(object)]

    __invert__ = flip

    def __abs__(x):
        return Composition(*(object for object in reversed((x.first,) + x.funcs)))

    def __reversed__(self):
        new = +type(self)
        for object in reversed((self.first,) + self.funcs):
            new = new[object]
        else:
            return new

    def __setattr__(self, name, object):
        if name not in self.__slots__:
            return self.partial(setterattr, object, name, duplicate=False)
        return super().__setattr__(name, object)

    def __setitem__(self, name, object):
        if name not in self.__slots__:
            return self.partial(setteritem, object, name, duplicate=False)
        return super().__setattr__(name, object)

    def compose(self, *callables):
        f = toolz.compose(*reversed(callables), *reversed(self))
        if self.args or self.kwargs:
            f = functools.partial(f, *self.args, **self.kwargs)
        return f

    comp = compose

    def load(self, format=None, init=None, data=None):
        return self.partial(
            __import__("anyconfig").load,
            ac_parser=format,
            ac_dict=init,
            ac_template=data is not None,
            ac_context=data,
        )

    def loads(self, format=None, init=None, data=None):
        return self.partial(
            __import__("anyconfig").loads,
            ac_parser=format,
            ac_dict=init,
            ac_template=data is not None,
            ac_context=data,
        )

    def dumps(self, format=None, **options):
        return self.partial(__import__("anyconfig").dumps, ac_parser=format, **options)

    def dump(self, format=None, **options):
        return self.partial(dump, format=format, **options)

    """Object tools"""

    """non symbollic functions"""

    flatten = functools.partialmethod(fold, flatten)

    def _ipython_key_completions_(self):
        object = []
        try:
            object = __import__(
                "IPython").core.completerlib.module_completion("import")
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

    glob = functools.partialmethod(partial, glob)
    for key in "read readline readlines execute compute read_text".split():
        builtins.locals()[key] = functools.partialmethod(
            method, operator.methodcaller, key
        )

    del key


rules = {
    Compose.fold: {
        "toolz": "pipe",
        "operator": """lt le eq ne ge gt not_ truth is_ is_not abs add and_ floordiv index inv invert
                    lshift mod mul matmul neg or_ pos pow rshift sub truediv xor concat contains countOf
                    delitem getitem indexOf setitem length_hint attrgetter itemgetter methodcaller iadd iand iconcat ifloordiv
                    ilshift imod imul imatmul ior ipow irshift isub itruediv ixor""",
        "fnmatch": """fnmatch fnmatchcase""",
        "builtins": """delattr eval exec format hasattr round setattr 
                    sorted sum bytes complex range slice super zip""",
        "importlib": "find_loader import_module",
        "random": """seed random uniform triangular randint choice randrange sample shuffle 
                    choices normalvariate lognormvariate expovariate vonmisesvariate gammavariate gauss 
                    betavariate paretovariate weibullvariate getstate setstate getrandbits""",
        "gc": """enable disable isenabled set_debug get_debug get_count set_threshold get_threshold collect get_objects
                    get_stats is_tracked get_referrers get_referents freeze unfreeze get_freeze_count""",
        "typing": """WrapperDescriptorType MethodWrapperType MethodDescriptorType Any NoReturn ClassVar Union Optional ForwardRef 
                    TypeVar Generic cast get_type_hints no_type_check no_type_check_decorator overload Hashable Awaitable 
                    Coroutine AsyncIterable AsyncIterator Iterable Iterator Reversible Sized Container Collection Callable 
                    AbstractSet MutableSet Mapping MutableMapping Sequence MutableSequence ByteString Tuple List Deque Set 
                    FrozenSet MappingView KeysView ItemsView ValuesView ContextManager AsyncContextManager Dict DefaultDict 
                    OrderedDict Counter ChainMap Generator AsyncGenerator Type SupportsInt SupportsFloat SupportsComplex 
                    SupportsBytes SupportsAbs SupportsRound NamedTupleMeta NamedTuple NewType Text IO BinaryIO TextIO Pattern Match""",
        "types": """FunctionType LambdaType CodeType MappingProxyType SimpleNamespace GeneratorType CoroutineType 
                    AsyncGeneratorType BuiltinFunctionType BuiltinMethodType WrapperDescriptorType 
                    MethodWrapperType MethodDescriptorType ClassMethodDescriptorType ModuleType TracebackType 
                    FrameType GetSetDescriptorType MemberDescriptorType new_class""",
        "itertools": """tee accumulate combinations combinations_with_replacement cycle islice chain compress count zip_longest permutations product repeat""",
        **(
            {
                "IPython.display": """Pretty HTML Markdown Math Latex SVG ProgressBar JSON GeoJSON Javascript Image clear_output 
                    publish_display_data DisplayHandle Video Audio IFrame YouTubeVideo VimeoVideo ScribdDocument FileLink FileLinks Code"""
            }
            if "IPython" in sys.modules
            else {}
        ),
    },
    Compose.method: {"operator": """attrgetter methodcaller itemgetter"""},
    Compose.star: {"asyncio": "gather"},
    Compose.partial: {
        "string": "Template",
        "pathlib": "Path",
        "types": "MethodType",
        "re": "match fullmatch search subn split findall finditer compile template escape",
        "random": """seed random uniform triangular randint choice randrange sample shuffle choices 
                    normalvariate lognormvariate expovariate vonmisesvariate gammavariate gauss betavariate 
                    paretovariate weibullvariate getstate setstate getrandbits""",
        "tokenize": """lookup TextIOWrapper ISTERMINAL ISNONTERMINAL ISEOF TokenInfo untokenize detect_encoding tokenize""",
        "contextlib": """wraps AbstractContextManager AbstractAsyncContextManager ContextDecorator 
                    contextmanager asynccontextmanager redirect_stdout redirect_stderr suppress 
                    ExitStack AsyncExitStack nullcontext""",
        "functools": """recursive_repr update_wrapper wraps total_ordering partialmethod lru_cache singledispatch""",
        "abc": """abstractmethod abstractclassmethod abstractstaticmethod abstractproperty get_cache_token""",
        "toolz": """accumulate apply assoc assoc_in compose_left concat
                    concatv cons count countby curried diff dissoc drop excepts frequencies
                    get get_in identity interleave interpose isdistinct isiterable itemfilter itemmap
                    iterate itertoolz join juxt keyfilter keymap last mapcat memoize merge
                    merge_sorted merge_with nth partition partition_all partitionby peek peekn
                    pluck random_sample reduceby remove second sliding_window sorted tail take
                    take_nth thread_first thread_last topk unique update_in valfilter valmap""",
        "statistics": """bisect_left bisect_right harmonic_mean mean median median_grouped median_high
                    median_low mode pstdev pvariance stdev variance""",
        "inspect": """ismodule isclass ismethod ismethoddescriptor isdatadescriptor ismemberdescriptor
                    isgetsetdescriptor isfunction isgeneratorfunction iscoroutinefunction isasyncgenfunction isasyncgen
                    isgenerator iscoroutine isawaitable istraceback isframe iscode isbuiltin isroutine isabstract
                    getmembers classify_class_attrs getmro unwrap indentsize getdoc cleandoc getfile getmodulename
                    getsourcefile getabsfile getmodule findsource getcomments getblock getsourcelines getsource
                    walktree getclasstree getargs getargspec getfullargspec getargvalues formatannotation
                    formatannotationrelativeto formatargspec formatargvalues getcallargs getclosurevars getframeinfo
                    getlineno getouterframes getinnerframes currentframe stack trace getattr_static getgeneratorstate
                    getgeneratorlocals getcoroutinestate getcoroutinelocals signature""",
        "fsspec": "open open_files",
        "importlib": "invalidate_caches reload",
        "copy": "copy deepcopy",
        "itertools": "dropwhile takewhile filterfalse starmap",
        "dataclasses": """Field field dataclass fields is_dataclass""",
        "io": """FileIO BytesIO StringIO BufferedReader BufferedWriter BufferedRWPair BufferedRandom IncrementalNewlineDecoder TextIOWrapper OpenWrapper IOBase RawIOBase BufferedIOBase TextIOBase""",
        "builtins": """abs all any ascii bin breakpoint callable chr dir dict globals hash hex id input iter
                    len locals max min next oct ord print repr sorted vars bool memoryview bytearray classmethod enumerate
                    float frozenset property int list object reversed set staticmethod str tuple type""",
    },
}

for key, value in rules.items():
    for module, value in value.items():
        module = __import__("importlib").import_module(module)
        for value in value.split():
            thing = builtins.getattr(module, value)
            setattr(
                Compose, value, functools.partialmethod(key, thing),
            )
del key, value, thing
# ## Conditional compositions


class Conditional(Compose):
    __slots__ = Compose.__slots__ + ("predicate",)

    def __init__(self, predicate, *args, **kwargs):
        self.predicate = super().__init__(*args, **kwargs) or predicate


class Iff(Conditional):
    """call a function with the sample arguments and keywords if the predicate is true."""

    def __call__(self, *args, **kwargs):
        object = self.predicate(*args, **kwargs)
        try:
            cond = bool(object)
        except ValueError:
            cond = True
        return super().__call__(*args, **kwargs) if cond else object


class IfThen(Conditional):
    """a conditional that continues only if the predicate is true."""

    def __call__(self, *args, **kwargs):
        object = self.predicate(*args, **kwargs)
        try:
            cond = bool(object)
        except ValueError:
            cond = True
        return super().__call__(object) if cond else object


class IfNot(Conditional):
    """call another function if the predicate evaluates to false."""

    def __call__(self, *args, **kwargs):
        object = self.predicate(*args, **kwargs)
        try:
            return object if object else super().__call__(*args, **kwargs)
        except ValueError:
            return object


# ## Compositional types.


class Type(abc.ABCMeta):
    __getattribute__ = Compose.__getattribute__

    partial = Compose.partial

    __getitem__ = Compose.__getitem__


for key in dir(Compose):
    if not hasattr(Type, key):
        try:
            setattr(Type, key, getattr(Compose, key))
        except TypeError:
            ...


class λ(Compose, metaclass=Type):
    def __init__(self, *args, **kwargs):
        super().__init__(kwargs.pop("funcs", None), *args, **kwargs)


# use lambda because the name is shorter in the repr
builtins.Pose = (
    builtins.the
) = builtins.an = builtins.a = builtins.λ = a = the = an = Pose = λ


class star(λ, Compose, metaclass=Type):
    def __call__(x, *object, **dict):
        args, kwargs = list(), {}
        for arg in x.args + object:
            kwargs.update(arg) if isinstance(
                arg, typing.Mapping) else args.extend(arg)
        return super().__call__(*args, **kwargs)


builtins.star = star


# ## A self referential function composition.


class ThisType(abc.ABCMeta):
    def __getitem__(x, object):
        return x().__getitem__(object)

    def __getattr__(x, str):
        return x().__getattr__(str)


class Λ(Composition, metaclass=ThisType):
    """
    >>> λ[(10 - Λ), abs(10 - Λ)](20)
    (-10, 10)
    """

    def __getitem__(x, object):
        return x.partial(operator.itemgetter(object))

    def __getattr__(x, object):
        def partial(*args, **kwargs):
            return x.partial(attribute(object, *args, **kwargs))

        return partial


for binop in "add sub mul matmul truediv floordiv mod eq lt gt ne xor".split():
    for cls in (ThisType, Λ):
        setattr(
            cls,
            f"__{binop}__",
            functools.wraps(getattr(operator, binop))(
                functools.partialmethod(
                    Composition.partial, toolz.flip(getattr(operator, binop))
                )
            ),
        )
        setattr(
            cls,
            f"__i{binop}__",
            functools.wraps(getattr(operator, binop))(
                functools.partialmethod(
                    Composition.partial, toolz.flip(getattr(operator, binop))
                )
            ),
        )
        setattr(
            cls,
            f"__r{binop}__",
            functools.wraps(getattr(operator, binop))(
                functools.partialmethod(
                    Composition.partial, getattr(operator, binop))
            ),
        )


def and_(a, b):
    return a and b


def or_(a, b):
    return a or b


for binop in (and_, or_):
    for cls in (ThisType, Λ):
        setattr(
            cls,
            f"__{binop.__name__}_",
            functools.wraps(binop)(
                functools.partialmethod(Composition.partial, toolz.flip(binop))
            ),
        )
        setattr(
            cls,
            f"__i{binop.__name__}_",
            functools.wraps(binop)(
                functools.partialmethod(Composition.partial, toolz.flip(binop))
            ),
        )
        setattr(
            cls,
            f"__r{binop.__name__}_",
            functools.wraps(binop)(
                functools.partialmethod(Composition.partial, toolz.flip(binop))
            ),
        )

for unaryop in "pos neg invert abs".split():
    setattr(
        Λ,
        f"__{unaryop}__",
        functools.wraps(getattr(operator, unaryop))(
            functools.partialmethod(Λ.partial, getattr(operator, unaryop))
        ),
    )


builtins.Self = builtins.Λ = Self = Λ
del binop, unaryop


__test__ = globals().get("__test__", {})
__test__[
    __name__
] = """
# Tests

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

Conditionals

    >>> λ[λ**int+bool, λ**str](10)
    (True, False)

Forward references.

    >>> e = Forward('builtins.range')
    >>> e
    ForwardRef('builtins.range')
    >>> e._evaluate(), e
    (<class 'range'>, <class 'range'>)
    >>> λ['random.random']()
    0...
    >>> λ['random itertools'.split()]()
    [<module 'random'...>, <module 'itertools' (built-in)>]
    >>> λ['itertools.chain.__name__']()
    'chain'

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

    >>> assert λ(λ) + dir + len + (Λ>500) + ...
    >>> (Λ[1][1]['a'].__add__(22).__truediv__(7))\\
    ...     ((None, [None, {'a': 20}]))
    6.0

    >>> assert λ.sub(10).add(3).truediv(2)(20) == 6.5
    >>> assert λ.fnmatch('abc*')('abcde')


"""


class Rules:
    partial = Pose.partialmethod(Compose.partial)
    fold = Pose.partialmethod(Compose.fold)


Type.read_csv = Compose.read_csv = Rules.fold(Pose["pandas.read_csv"])
