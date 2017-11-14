
# coding: utf-8

from collections import UserList, UserDict, ChainMap
from functools import partialmethod, wraps
from inspect import getfullargspec, signature
from itertools import zip_longest, starmap
from operator import attrgetter, not_, eq, methodcaller, itemgetter
from toolz.curried import isiterable, identity, concat, concatv, flip, cons, merge, memoize
from toolz import map, groupby, filter, reduce
from pathlib import Path
from copy import copy
dunder = '__{}__'.format
__all__ = 'a', 'an', 'the', 'star', 'do', 'flip', 'compose', 'composite', 'λ', 'this', 'juxt', 'parallel', 'memo', 'Path'


class functions(UserList):
    """A composition of functions."""
    __slots__ = 'data',
        
    def __init__(self, data=None):
        if data and not isiterable(data):
            data = [data]
        super().__init__(data or list())
        self.__qualname__ = __name__ + '.' + type(self).__name__
    
    def __call__(self, *args, **kwargs):
        """Call an iterable as a function evaluating the arguments in serial."""                    
        for value in self:
            args, kwargs = (
                [value(*args, **kwargs)] if callable(value) else [value]), dict()
        return args[0] if len(args) else None    
        
    def __copy__(self):
        compose = type(self)(*map(copy, self.__getstate__()))
        compose.data = list(map(copy, self.data))
        return compose
    
    copy = __copy__

    def __exit__(self, exc_type, exc_value, traceback): pass
    
    __enter__ = __deepcopy__ = __copy__
    def __abs__(self):
        return self.__call__
    
    def __reversed__(self): 
        self.data = type(self.data)(reversed(self.data))
        return self
    
    def __repr__(self, i=0):
        return (type(self).__name__ or 'λ').replace('compose', 'λ') + '>' + ':'.join(map(repr, self.__getstate__()[i:]))   
    __name__ = property(__repr__)
        
    def __hash__(self): return hash(tuple(self))
    def __bool__(self): return any(self.data)
    
    def __getattr__(self, attr, *args, **kwargs):
        if attr in self.__slots__ or attr in self.__dict__:
            return object.__getattribute__(self, attr)
        if callable(attr):
            if args or kwargs:
                return self[partial(attr, *args, **kwargs)]
            return self[attr]
        raise AttributeError(attr)

    
    def __getitem__(self, object):
        if object == slice(None): return self        
        if isinstance(object, (int, slice)): 
            try:
                return self.data[object]
            except IndexError as e: raise e
        return self.append(object) or self
    
    @property
    def _first(self):
        out = self
        while isinstance(out, UserList): out = out[0]
        return out
    
    @property
    def __annotations__(self):
        return getattr(self._first, '__annotations__', {})

    @property
    def __signature__(self):
        return signature(self._first)
    
    def __magic__(self, name):
        from IPython import get_ipython
        ip = get_ipython()
        if ip:            
            def magic_wrapper(line, cell=None):
                if not(cell is None):
                    line += '\n'+cell
                return self(line)
            ip.register_magic_function(wraps(self)(magic_wrapper), 'line_cell', name)


class partial(__import__('functools').partial):
    def __eq__(self, other, result = False):
        if isinstance(other, partial):
            result = True
            for a, b in zip_longest(*(cons(_.func, _.args) for _ in [self, other])):
                result &= (a is b) or (a == b)
        return result


class attributes(ChainMap):
    def new_child(self, m=None):
        from importlib import  import_module
        if type(m) is str:
            m = import_module(m)
        self.maps.append(getattr(m, '__dict__', m))


class compose(functions):
    """A composition of functions."""
    _attributes_ = attributes()
    
    def __getattr__(self, attr, *args, **kwargs):
        try:
            parent = super().__getattr__(attr, *args, **kwargs)
            if parent: 
                return parent
        except AttributeError as e:
            if attr in self._attributes_:
                value = self._attributes_.get(attr, attr)
                def wrapper(*args, **kwargs):
                    nonlocal value
                    (self.data[-1] if isinstance(self, composite) else self)[
                        value.func(*args, **kwargs) if isinstance(value,  partial)
                        else partial(value, *args, **kwargs) if args or kwargs and callable(value)
                        else value]
                    return self
                return wraps(getattr(value, 'func', value))(wrapper)
        raise AttributeError(attr)
    
    def __getstate__(self):
        return tuple(getattr(self, slot) for slot in self.__slots__)
    
    def __setstate__(self, state):
        for attr, value in zip(self.__slots__, state):
            setattr(self, attr, value)
        
    __truediv__  = partialmethod(__getattr__, map)
    __floordiv__ = partialmethod(__getattr__, filter)
    __matmul__   = partialmethod(__getattr__, groupby)
    __mod__      = partialmethod(__getattr__, reduce)

    def __getitem__(self, object):
        if isiterable(object) and not isinstance(object, (str, compose)):
            object = juxt(object)
        return super().__getitem__(object)
    
    __mul__ = __add__ = __rshift__ = __sub__ = __getitem__
    
    def __lshift__(self, object):          return do(object)
    def __xor__(self, object):             return excepts(object, self)
    def __or__(self, object=None):         return ifnot(self, object)
    def __and__(self, object=None):        return ifthen(self, object)
    def __pow__(self, object=None):        return instance(object, self)
    
    __pos__ = partialmethod(__getitem__, bool)
    __neg__ = partialmethod(__getitem__, not_)
    __invert__ = functions.__reversed__
    
    def __dir__(self):
        return super().__dir__() + list(self._attributes_.keys())
    
for attrs in [
    dict(fnmatch=flip(__import__('fnmatch').fnmatch)),
    'io', 'inspect', 'builtins', 'itertools', 'collections', 'pathlib', 'json', 'requests', 'toolz',{
        k: (partial if k.endswith('getter') or k.endswith('caller') else flip)(v)
        for k, v in vars(__import__('operator')).items()}
]:
    compose._attributes_.new_child(attrs)
    
del attrs


class do(compose):
    def __call__(self, *args, **kwargs):
        super(do, self).__call__(*args, **kwargs)
        return args[0] if args else None
    
class flipped(compose):
    def __call__(self, *args, **kwargs):
        return super().__call__(*reversed(args), **kwargs)


class juxt(compose):
    __slots__ = 'data', 'type'
    """Any mapping is a callable, call each of its elements."""
    def __init__(self, data=None, type=None):
        if isiterable(data) and not isinstance(data, self.__class__.__mro__[1]):
            self.type = type or data.__class__ or tuple
        super().__init__(
            list(data.items()) if issubclass(self.type, dict) else list(data) or list())

    def __call__(self, *args, **kwargs):
        result = list()
        for callable in self.data:
            if not isinstance(callable, (str, compose)) and isiterable(callable):
                callable = juxt(callable)
            if not isinstance(callable, compose):
                callable = compose([callable])            
            result.append(callable(*args, **kwargs))
        return self.type(result)


class condition(compose):
    __slots__ = 'condition', 'data'
    def __init__(self, condition=None, data=None):
        setattr(self, 'condition', condition) or super().__init__(data)

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs) if self else True

class ifthen(condition):
    """Evaluate a function if a condition is true."""
    def __call__(self, *args, **kwargs):
        return self.condition(*args, **kwargs) and super(ifthen, self).__call__(*args, **kwargs)

class ifnot(condition):
    """Evaluate a function if a condition is false."""
    def __call__(self, *args, **kwargs):
        return self.condition(*args, **kwargs) or super(ifnot, self).__call__(*args, **kwargs)

class instance(ifthen):
    """Evaluate a function if a condition is true."""
    def __init__(self, condition=None, data=None):        
        if isinstance(condition, type):
            condition = condition,            
        if isinstance(condition, tuple):
            condition = partial(flip(isinstance), condition)
        super().__init__(condition, data or list())


class FalseException(compose):
    __slots__ = 'exception',
    def __init__(self, exception):        
        self.exception = exception
    def __bool__(self):  return False

class excepts(compose):
    __slots__ = 'exceptions', 'data'
    """Allow acception when calling a function"""
    def __init__(self, exceptions=None, data=None):
        setattr(self, 'exceptions', exceptions) or super().__init__(data)
    
    def __call__(self, *args, **kwargs):
        try:
            return super(excepts, self).__call__(*args, **kwargs)
        except self.exceptions as e:
            return FalseException(e)


class composite(compose):
    """A composite composition with push and pop methods.  It chains compositions together
    allowing a chainable api to map, filter, reduce, and groupby functions.|
    """
    def __init__(self, data=None):
        super().__init__([data or compose()])
    
    def __getattr__(self, attr):
        if isinstance(self, factory): self = composite()
        def wrapped(*args, **kwargs):
            nonlocal self, attr
            self.data[-1] = getattr(self.data[-1], attr)(*args, **kwargs)
            return self
        return wraps(super().__getattr__(attr))(wrapped)
    
    def push(self, type=compose, *args):
        self[type(*args)]
        not self.data[0] and self.pop(0)
        return self
    

    def __getitem__(self, *args, **kwargs):
        if isinstance(self, factory): self = composite()
        if args[0] == slice(None): return self
        if args and isinstance(args[0], (int, slice)): 
            try:
                return self.data[args[0]]
            except IndexError as e:
                raise e
        try:
            self.data[-1].__getitem__(*args, **kwargs)
        except AttributeError:
            self.push()
            self.data[-1].__getitem__(*args, **kwargs)
        return self    
    
    __mul__ = __add__ = __rshift__ = __sub__ = __getitem__

for cls in [ifthen, ifnot, excepts, do, instance]: 
    setattr(composite, cls.__name__, partialmethod(composite.push, cls))



def right_attr(self, attr, other):
    self = self[:]
    return op_attr(type(self)(compose(other)), attr, self)

def op_attr(self, attr, other): 
    if isinstance(self, factory): self = self[:]
    if isinstance(self, composite):
        self.data[-1] = object.__getattribute__(self.data[-1], attr)(other)
    else:
        self = object.__getattribute__(self, attr)(other)
    return self
    
for other in ['and', 'or', 'xor', 'truediv', 'floordiv', 'matmul', 'mod', 'lshift', 'pow']:
    setattr(composite, dunder(other), partialmethod(op_attr, dunder(other)))
    
for other in ['mul', 'add', 'rshift' ,'sub', 'and', 'or', 'xor', 'truediv', 'floordiv', 'matmul', 'mod', 'lshift', 'pow']:
    setattr(compose, dunder('i'+other), partialmethod(op_attr, dunder(other)))
    setattr(compose, dunder('r'+other), partialmethod(right_attr, dunder(other)))
    setattr(composite, dunder('r'+other), partialmethod(right_attr, dunder(other)))


class factory(composite):
    __slots__ = 'args', 'kwargs', 'data'
    def __init__(self):
        self.args, self.kwargs, self.data = None, None, list()
        
    def __getitem__(self, attr):
        if attr == slice(None): return compose() if isinstance(self, factory) else self
        if isinstance(self.args, tuple) and  isinstance(self.kwargs, dict):
            attr = partial(attr, *self.args, **self.kwargs)
        return super().__getitem__(attr)
        
    def __call__(self, *args, **kwargs):
        if isinstance(self.args, tuple) and  isinstance(self.kwargs, dict):
            return new[:](*concatv(self.args, args), **merge(self.kwargs, kwargs))
        self = type(self)()
        self.args, self.kwargs = args, kwargs
        return self
    
    __mul__ = __add__ = __rshift__ = __sub__ = push = __getitem__

a = an = the = λ = factory()


class memo(composite):
    def __init__(self, cache=None, data=None):
        self.cache = dict() if cache is None else cache
        super().__init__(data)

    def memoize(self): return memoize(super().__call__, cache=self.cache)
    
    __call__ = property(memoize)

    __repr__ = partialmethod(composite.__repr__, 1)


class parallel(composite):
    def __init__(self, jobs, data=None):
        self.jobs = jobs
        super().__init__(data)
        
    def map(self, function):
        return super().__getattr__('map')(__import__('joblib').delayed(function))

    __truediv__ = map
    
    def __call__(self, *args, **kwargs):
        return __import__('joblib').Parallel(self.jobs)(super().__call__(*args, **kwargs))
        
    __repr__ = partialmethod(composite.__repr__, 1)


class stargetter:
    def __init__(self, attr, *args, **kwargs):
        self.attr, self.args, self.kwargs = attr, args, kwargs

    def __call__(self, object):
        object = attrgetter(self.attr)(object)
        if callable(object):
            return object(*self.args, **self.kwargs)
        return object
    
class this(compose):
    def __getattr__(self, attr):
        def wrapped(*args, **kwargs):
            self.data.append(stargetter(attr, *args, **kwargs))
            return self
        return wrapped
    
    def __getitem__(self, attr):
        if isinstance(attr, str):
            return self[itemgetter(attr)]
        return super().__getitem__(attr)

class star(compose):
    """Call a function starring the arguments for sequences and starring the keywords for containers."""
    def __call__(self, *inputs):
        args, kwargs = list(), dict()
        for input in inputs:
            if isinstance(input, dict):
                kwargs.update(**input)
            else:
                args += list(input)
        return super(star, self).__call__(*args, **kwargs)


if __name__ == '__main__':
    get_ipython().system('jupyter nbconvert --to python --TemplateExporter.exclude_input_prompt=True articles.ipynb')

