
# coding: utf-8

from collections import UserList
from functools import partialmethod, wraps
from inspect import signature, getdoc, getsource
from itertools import zip_longest, starmap
from operator import attrgetter, not_, eq, methodcaller, itemgetter
from toolz.curried import isiterable, identity, concat, concatv, flip, cons, merge, memoize, keymap
from toolz import map, groupby, filter, reduce
from copy import copy
dunder = '__{}__'.format
__all__ = 'a', 'an', 'the', 'star', 'do', 'λ', 'juxt', 'parallel', 'cache', 'ifthen', 'ifnot', 'excepts', 'instance', 'partial', 'dispatch'


class partial(__import__('functools').partial):
    """partial overloads functools.partial to provide equality and documentation.
    
    >>> f = partial(range, 10) 
    >>> f == partial(range, 10)
    >>> assert partial(range, 10, 20) == partial(range, 10)
    """
    def __eq__(self, other):
        return isinstance(other, partial) and all(
            (a is b) or (a == b) for a, b in zip_longest(*(cons(_.func, _.args) for _ in [self, other])))
    
    @property
    def __doc__(self): return getdoc(self.func)

class partial_attribute(partial):
    """partial_attribute is a partial for MethodType attributes.
    
    >>> f = partial_attribute(str.replace, 'x', 'y')
    >>> assert f('xy') == 'yy'
    """
    def __call__(self, object):
        if callable(self.func):
            return self.func(object, *self.args, **self.keywords) 
        return self.func


class __callable__(UserList):
    """__callable__ is a callable list."""        
    def __call__(self, *args, **kwargs):
        for value in self:
            args, kwargs = ([value(*args, **kwargs)] if callable(value) else [value]), dict()
        return args[0] if len(args) else None    


class compose(__callable__):
    __slots__ = 'data',
    _annotations_ = None

    def __init__(self, data=None):
        super().__init__(data is not None and (not isiterable(data) or isinstance(data, str)) and [data] or data or list())
        self.__qualname__ = __name__ + '.' + type(self).__name__
    
    def __getitem__(self, object):
        """Use brackets to append functions to the compose.
        >>> compose()[range][list]
        compose:[<class 'range'>, <class 'list'>]
        
        Iterable items are juxtaposed.
        >>> compose()[range][list, type]
        compose:[<class 'range'>, juxt(<class 'tuple'>)[<class 'list'>, <class 'type'>]]
        """
        if isiterable(object) and not isinstance(object, (str, compose)):
            object = juxt(object)
            
        if object in (slice(None), getdoc): return self        
        
        return self.data[object] if isinstance(object, (int, slice)) else self.append(object)
            
    def __getattr__(self, attr, *args, **kwargs):
        """extensible attribute method relying on compose.attributer
        
        >>> assert a.range().len() == a.builtins.range().builtins.len() == a[range].len()
        """
        if callable(attr): 
            args = (arg if callable(arg) else λ[arg] for arg in args)
            return self[:][partial(attr, *args, **kwargs)]
        try:
            return super().__getattr__(attr, *args, **kwargs)
        except AttributeError as e:
            return getattr(self.attributer(self[:]), attr)

    def append(self, object):
        return  self.data.append(object) or not self.data[0] and self.data.pop(0) or self        
    
    def __repr__(self):
        other = len(self.__slots__) is 2 and repr(self.__getstate__()[1-self.__slots__.index('data')])
        return type(self).__name__.replace('composition', 'λ') +(
            '({})'.format(other or '').replace('()', ':')
        )+ super().__repr__()

    @property
    def __name__(self): return type(self).__name__
            
    @property
    def __annotations__(self): 
        return self._annotations_ or self and getattr(self[0], dunder('annotations'), None) or {}
    
    @property
    def __signature__(self): return signature(self[0])
    def __exit__(self, exc_type, exc_value, traceback): pass
    def __hash__(self): return hash(tuple(self))
    def __bool__(self): return any(self.data)
    def __abs__(self): return self.__call__
    def __reversed__(self): return type(self)(list(reversed(self.data)))      
    def __getstate__(self): return tuple(getattr(self, slot) for slot in self.__slots__)
    def __setstate__(self, state):
        for attr, value in zip(reversed(self.__slots__), reversed(state)): setattr(self, attr, value)
            
    def __copy__(self, memo=None):
        new = type(self.__name__, (type(self),), {'_annotations_': self._annotations_})()
        return new.__setstate__(tuple(map(copy, self.__getstate__()))) or new
    
    def __dir__(self): return super().__dir__() + dir(self.attributer())

    copy = __enter__ = __deepcopy__ = __copy__


@partial(setattr, compose, 'attributer')
@staticmethod
class attributer(object):
    def __init__(self, composition=None, object=None, parent=None):
        self.object, self.composition, self.parent = object, composition, parent

    def __iter__(self):
        if self.object: yield self.object
        else:
            for object in self.imports: 
                yield type(object) is str and __import__(object) or object

    def __getitem__(self, item):
        objects = list(self)
        if len(objects) > 1:
            for object in objects:
                if getattr(object, dunder('name'), "") == item: return object
        for object in objects:
            dict = getattr(object, dunder('dict'), object)
            if item in dict: return dict[item]
        else: raise AttributeError(item)

    def __dir__(self):
        return (list() if self.object else [value for value in self.imports if isinstance(value, str)]) + list(concat(
            getattr(object, dunder('dict'), object).keys() for object in self))

    def __getattr__(self, item): return type(self)(self.composition, self[item], self.object)

    def __repr__(self): return repr(self.object or list(self))

    @property
    def __doc__(self):
        try: return getdoc(self.object) or inspect.getsource(self.object)
        except: pass

    def __call__(self, *args, **kwargs):
        object = self.object
        if callable(object):
            for decorator, values in self.decorators.items():
                if object in values: 
                    object = decorator(object)
                    if isinstance(object, partial):
                        object = object.func(*args, **kwargs)  
                    break
            else:
                if isinstance(self.parent, type):
                    object = partial_attribute(object, *args, **kwargs)
                elif args or kwargs:
                    object = partial(object, *args, **kwargs)
        return (λ.object() if self.composition is None else self.composition)[object]
    
    @property
    def __signature__(self): return signature(self.object or self)


compose.attributer.imports = list(['toolz', 'requests', 'builtins', 'json', 'pickle', 'io', 
        'collections', 'itertools', 'functools', 'pathlib', 'importlib', 'inspect', 'operator'])


# decorators for the operators.
import operator, fnmatch
# some of these cases fail, but the main operators work.
compose.attributer.decorators = keymap([flip, partial].__getitem__, groupby(
    attrgetter('itemgetter', 'attrgetter', 'methodcaller')(operator).__contains__, 
    filter(callable, vars(__import__('operator')).values())
))
compose.attributer.imports.insert(0, {'fnmatch': fnmatch.fnmatch})
compose.attributer.decorators[flip].append(fnmatch.fnmatch)


class factory(compose):
    """A factory of composition that works as a decorator.
    
    Create a factory
    >>> some = factory(compose)
    >>> some.range()
    compose:[<class 'range'>]
    
    Supply partial arguments
    
    >>> some(10)(20)
    10
    >>> some(10)[range](20)
    range(10, 20)
    >>> assert some(10)[range](20) == a.range(10)(20)
    """
    __slots__ = 'object', 'data', 'args', 'kwargs'
    def __init__(self, object, data=None, args=None, kwargs=None):
        super().__init__(data)
        self.object, self.args, self.kwargs = object, args, kwargs
        
    def __getitem__(self, attr):
        if isinstance(self.args, tuple) and  isinstance(self.kwargs, dict):
            attr = attr == slice(None) and abs(self) or partial(attr, *self.args, **self.kwargs)
        return self.object()[attr]
                
    def __getattr__(self, attr, *args, **kwargs):
        return self.object().__getattr__(attr, *args, **kwargs)
    
    def __call__(self, *args, **kwargs):
        return (
            next(concatv(self.args, args)) if isinstance(self.args, tuple) and  isinstance(self.kwargs, dict) 
            else factory(self.object, self.data, args, kwargs))    
    
    def __dir__(self): return dir(self.object())
    
    __mul__ = __add__ = __rshift__ = __sub__ = __getitem__


class composition(compose):
    """λ provides syntactic sugar to functions.
    
    Prefer using the factory articles `a`, `an`, `the`, and `λ` because they save
    on typography in both naming and the use of parenthesis.
    
    >>> def mul(x): return x*10
    >>> λ[range].map(range).builtins.list()
    λ:[<class 'range'>, partial(<class 'map'>, <class 'range'>), <class 'list'>]
    
    Conditional dispatching.
    >>> a[10, 'test', {10}].map(a**int&range|a**str&str.upper|type).list()()
    [range(0, 10), 'TEST', <class 'set'>]
    """
    def __lshift__(self, object):          return self[do(object)]
    def __pow__(self, object=slice(None)):
        """
        >>> f = a**int*range
        >>> a[10, '10'].map(f).list()()
        [range(0, 10), False]
        
        A dictionary sets the function attributes.
        
        >>> assert (a**{'start': int, 'returns': range}*range).__annotations__
        """
        self = self[:]
        if isinstance(object, str):
            return setattr(self, '__doc__', object) or self
        if isinstance(object, dict):
            return setattr(self, '_annotations_', object) or self
        return instance(object)[self]
    
    def __and__(self, object=slice(None)):        
        """append an ifthen statement to the composition
        
        >>> (a&range)(0), (a&range)(10)
        (0, range(0, 10))
        """
        return ifthen(self[:])[object]
    def __or__(self, object=slice(None)):  
        """append an ifnot statement to the composition
        
        >>> (a|range)(0), (a|range)(10)
        (range(0, 0), 10)
        """
        return ifnot(self[:])[object] # There is no reasonable way to make this an attribute?
    
    def __xor__(self: 'λ', object: (slice, Exception)=slice(None)) -> 'λ':             
        """append an exception to the composition
        
        >>> (a.str.upper()^TypeError)(10)
        TypeError("descriptor 'upper' requires a 'str' object but received a 'int'",)
        """
        return excepts(object)[self[:]]

    __mul__ = __add__ = __rshift__ = __sub__ = compose.__getitem__
    factory.map = map = __truediv__  = partialmethod(compose.__getattr__, map)
    factory.filter = filter = __floordiv__ = partialmethod(compose.__getattr__, filter)
    factory.groupby = groupby = __matmul__   = partialmethod(compose.__getattr__, groupby)
    factory.reduce = reduce = __mod__      = partialmethod(compose.__getattr__, reduce)
    __pos__ = partialmethod(compose.__getitem__, bool)
    __neg__ = partialmethod(compose.__getitem__, not_)
    __invert__ = compose.__reversed__    
    
    def __magic__(self, name, *, ip=None):
        ip, function = ip or __import__('IPython').get_ipython(), self.copy()
        @wraps(function)
        def magic_wrapper(line, cell=None):
            return function('\n'.join(filter(bool, [line, cell])))
        ip.register_magic_function(magic_wrapper, 'cell', name)
        
a = an = the = λ = factory(composition)


composition.__truediv__.__doc__ = """>>> λ / range
λ:[partial(<class 'map'>, <class 'range'>)]"""
composition.__floordiv__.__doc__ = """>>> λ // range
λ:[partial(<class 'filter'>, <class 'range'>)]"""
composition.__matmul__.__doc__ = """>>> λ @ range
λ:[partial(<class 'groupby'>, <class 'range'>)]"""
composition.__mod__.__doc__ = """>>> λ % range
λ:[partial(<class 'reduce'>, <class 'range'>)]"""


def right_attr(right, attr, left): 
    if isinstance(left, factory): left = left[:]
    return object.__getattribute__(λ[left], attr)(right[:])

def op_attr(self, attr, value): 
    return object.__getattribute__(self[:], attr)(value)
        

[setattr(factory, dunder(attr), getattr(composition, dunder(attr))) or
 setattr(factory, dunder('r'+attr), partialmethod(right_attr, dunder(attr)))
 for attr in ['and', 'or', 'xor', 'truediv', 'floordiv', 'matmul', 'mod', 'lshift', 'pow']]

[setattr(object, dunder('i'+attr), partialmethod(op_attr, dunder(attr))) or
 setattr(object, dunder('r'+attr), partialmethod(right_attr, dunder(attr)))
 for attr in ['mul', 'add', 'rshift' ,'sub', 'and', 'or', 'xor', 'truediv', 'floordiv', 'matmul', 'mod', 'lshift', 'pow']
 for object in [composition, factory]]

[setattr(object, key, getattr(object, dunder(other))) for key, other in zip(('do', 'excepts', 'instance'), ('lshift', 'xor', 'pow')) for object in [factory, composition]];


@factory
class do(composition):
    """
    >>> assert not λ[print](10) and do()[print](10) is 10
    10
    10
    """
    
    def __call__(self, *args, **kwargs):
        super(do, self).__call__(*args, **kwargs)
        return args[0] if args else None


class juxt(composition):
    """juxtapose functions.
    
    >>> juxt([range, type])(10)
    [range(0, 10), <class 'int'>]
    """
    __slots__ = 'data', 'object'
    
    def __init__(self, data=None, object=None):
        if isiterable(data) and not isinstance(data, composition):
            object = object or type(data)
        self.object = object or tuple
        super().__init__(list(data.items()) if isinstance(data, dict) else list(data or list()))

    def __call__(self, *args, **kwargs):
        result = list()
        for callable in self.data:
            if not isinstance(callable, (str, compose)) and isiterable(callable):
                callable = juxt(callable)
            if not isinstance(callable, compose):
                callable = compose(callable)
            result.append(callable(*args, **kwargs))
        return self.object(result)


class condition(composition):
    __slots__ = 'condition', 'data'
    def __init__(self, condition=bool, data=None):
        setattr(self, 'condition', condition) or super().__init__(data)
        
class ifthen(condition):
    """the composition is executed only if the condition is true.
    
    >>> (a[0, 10] / ifthen(bool)[range] * list)()
    [False, range(0, 10)]
    """
    def __call__(self, *args, **kwargs):
        return self.condition(*args, **kwargs) and super(ifthen, self).__call__(*args, **kwargs)

class ifnot(condition):
    """the composition is executed only if the condition is false.
    
    >>> (a[0, 10] / ifnot(bool)[range] * list)()
    [range(0, 0), True]
    """
    def __call__(self, *args, **kwargs):
        return self.condition(*args, **kwargs) or super(ifnot, self).__call__(*args, **kwargs)

class instance(ifthen):
    """a conditional composition for instances/types
    
    >>> a[instance(str)[str.upper], instance(int)[range]](10)
    (False, range(0, 10))
    """
    def __init__(self, condition=None, data=None):        
        if isinstance(condition, type): condition = condition,            
        if isinstance(condition, tuple): condition = partial(flip(isinstance), condition)
        super().__init__(condition, data)


class FalseException(object):
    """A false wrapper for an exceptions"""
    def __init__(self, exception): self.exception = exception
    def __bool__(self):  return False
    def __repr__(self): return repr(self.exception)

class excepts(composition):
    """
    >>> excepts(TypeError)[str.upper](10)
    TypeError("descriptor 'upper' requires a 'str' object but received a 'int'",)
    """
    __slots__ = 'exceptions', 'data'
    def __init__(self, exceptions=None, data=None):
        setattr(self, 'exceptions', exceptions) or super().__init__(data)
    
    def __call__(self, *args, **kwargs):
        try: return super(excepts, self).__call__(*args, **kwargs)
        except self.exceptions as e:
            return FalseException(e)


class parallel(composition):
    """An embarassingly parallel composition
    
    All map functions are delayed
    >>> parallel(jobs=4)[range].map(print) # doctest: +SKIP
    """
    def __init__(self, jobs=4, data=None):
        setattr(self, 'jobs', jobs) or super().__init__(data)
        
    def map(self, function):
        """A delay each function."""
        return super().__getattr__('map')(__import__('joblib').delayed(function))
    
    def __call__(self, *args, **kwargs):
        return __import__('joblib').Parallel(self.jobs)(super().__call__(*args, **kwargs))
        
    __truediv__ = map

class cache(composition):
    """a cached composition
    
    >>> f = cache().range()
    >>> f(42), f.object
    (range(0, 42), {((42,), None): range(0, 42)})
    """
    def __init__(self, object=None, data=None):
        self.object = dict() if object is None else getattr(data, 'object', object)
        super().__init__(data)

    @property
    def __call__(self): return memoize(super().__call__, cache=self.object)

@factory
class star(composition):
    """star sequences as arguments and containers as keywords
    
    >>> def f(*args, **kwargs): return args, kwargs
    >>> star()[f]([10, 20], {'foo': 'bar'})
    ((10, 20), {'foo': 'bar'})
    """
    def __call__(self, *inputs):
        args, kwargs = list(), dict()
        [kwargs.update(**input) if isinstance(input, dict) else args.extend(input) for input in inputs]
        return super().__call__(*args, **kwargs)


class dispatch(composition):
    """a singledispatching composition

    >>> f = dispatch((str, str.upper), (int, range), (object, type))
    >>> (a['text', 42, {10}] / f * list)()
    ['TEXT', range(0, 42), <class 'set'>]
    """
    def __init__(self, *data):
        self.dispatch = None
        super().__init__(isinstance(data[0], dict) and list(data.items()) or data)

    def __call__(self, arg):
        if not self.dispatch:
            self.dispatch = __import__('functools').singledispatch(λ[:])
            for cls, func in self.data: self.dispatch.register(cls, func)
        return self.dispatch.dispatch(type(arg))(arg)


def load_ipython_extension(ip=__import__('IPython').get_ipython()):
    if ip: ip.Completer.use_jedi = False
load_ipython_extension()


if __name__ == '__main__':
    print(__import__('doctest').testmod(verbose=False))
    get_ipython().system('jupyter nbconvert --to python --TemplateExporter.exclude_input_prompt=True articles.ipynb')

