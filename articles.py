
# coding: utf-8

from collections import UserList
from functools import partialmethod, wraps
from inspect import signature, getdoc, getsource
from itertools import zip_longest, starmap
from operator import attrgetter, not_, eq, methodcaller, itemgetter
from toolz.curried import isiterable, identity, concat, concatv, flip, cons, merge, memoize
from toolz import map, groupby, filter, reduce
from copy import copy
dunder = '__{}__'.format
IGNORE = slice(None),
__all__ = (
    'a', 'an', 'the', 'star', 'do', 'λ', 'this', 'juxt', 'compose',
    'parallel', 'memo', 'then', 'ifthen', 'ifnot', 'excepts', 'instance')


class functions(UserList):
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
    
    def __getitem__(self, object):
        if object in IGNORE: return self        
        if isinstance(object, (int, slice)): 
            return self.data[object]
        return self.append(object)
    
    def append(self, object):
        self.data.append(object)
        if not self.data[0]: self.data.pop(0)
        return  self
        
    def __abs__(self):
        return self.__call__
    
    def __reversed__(self): 
        self.data = type(self.data)(reversed(self.data))
        return self
    
    def __repr__(self, i=0):
        return (type(self).__name__ or 'λ').replace('compose', 'λ') + '>' + ':'.join(map(repr, self.__getstate__()[i:]))   

    @property
    def __name__(self): return type(self).__name__
    def __hash__(self): return hash(tuple(self))
    def __bool__(self): return any(self.data)
    
    def __getattr__(self, attr, *args, **kwargs):
        try:
            return object.__getattribute__(self, attr)
        except Exception as e:
            if callable(attr):
                if args or kwargs:
                    return self[partial(attr, *args, **kwargs)]
                return self[attr]
            raise e
            
    @property
    def __annotations__(self): return getattr(self._first, dunder('annotations'), {})
    
    @property
    def __signature__(self):
        return signature(self._first)
    
    def __getstate__(self):
        return tuple(getattr(self, slot) for slot in self.__slots__)
    
    def __setstate__(self, state):
        for attr, value in zip(self.__slots__, state): setattr(self, attr, value)
            
    def __copy__(self, memo=None):
        new = type(self)()
        return new.__setstate__(self.__getstate__()) or new
    
    def __exit__(self, exc_type, exc_value, traceback): pass

    copy = __enter__ = __deepcopy__ = __copy__


class partial(__import__('functools').partial):
    def __eq__(self, other, result = False):
        if isinstance(other, partial):
            result = True
            for a, b in zip_longest(*(cons(_.func, _.args) for _ in [self, other])):
                result &= (a is b) or (a == b)
        return result
    
    @property
    def __doc__(self): return getdoc(self.func)


class partial_attribute(partial):
    def __call__(self, object):
        if callable(self.func):
            return self.func(object, *self.args, **self.keywords)
        return self.func


class _composition_attr(object):
    def __init__(self, maps=list(), parent=None, composition=None):
        if not isiterable(maps): maps = [maps]
        self._maps, self.composition, self.parent = list(maps), composition, parent

    @property
    def _current(self): 
        object = self._maps[0] 
        return slice(None) if object is getdoc else object
    
    @property
    def maps(self):
        return [getattr(object, dunder('dict'), object) for object in self._maps]
    
    def __getitem__(self, item):
        for object in self._maps:
            if getattr(object, dunder('name'), """""") == item:
                return object, None
        for raw, object in zip(self._maps, self.maps):
            if item in object:
                return object[item], raw
        raise AttributeError(item)

    def __dir__(self):
        keys = list()
        for raw, object in zip(self._maps, self.maps):
            keys += [getattr(raw, dunder('name'), """""")] + list(object.keys())
        return list(sorted(filter(bool, keys)))
            
    def __getattr__(self, value):
        return self.__class__(*self[value], self.composition)
    
    def __repr__(self): return repr(self._current)
    
    @property
    def __doc__(self):
        try:
            return getdoc(self._current) or inspect.getsource(self._current)
        except:
            return """No docs."""
    
    def __call__(self, *args, **kwargs):
        value = self._current
        if isinstance(self.parent, type):
            value = partial_attribute(value, *args, **kwargs)
        elif callable(value):
            if isinstance(value, partial) and not (value.args or value.keywords):
                value = value.func(*args, **kwargs)
            elif args or kwargs:
                value = partial(value, *args, **kwargs)
        return self.composition[value]


class compose(functions):
    """A composition of functions."""
    _attributes_ = list(map(__import__, [
        'toolz', 'requests', 'builtins', 'json', 'pickle', 'io', 
        'collections', 'itertools', 'functools', 'pathlib', 
        'importlib', 'inspect']))    
    @property
    def __attributes__(self):
        return _composition_attr(self._attributes_, None, self[:])
    
    def __getattr__(self, attr, *args, **kwargs):
        if callable(attr):
            return self[:][partial(attr, *args, **kwargs)]
        try:
            return super().__getattr__(attr, *args, **kwargs)
        except AttributeError as e:
            return getattr(self.__attributes__, attr)
        
    __truediv__  = partialmethod(__getattr__, map)
    __floordiv__ = partialmethod(__getattr__, filter)
    __matmul__   = partialmethod(__getattr__, groupby)
    __mod__      = partialmethod(__getattr__, reduce)

    def __getitem__(self, object):
        if isiterable(object) and not isinstance(object, (str, compose)):
            object = juxt(object)
        return super().__getitem__(object)
    
    __mul__ = __add__ = __rshift__ = __sub__ = __getitem__
    
    def __lshift__(self, object):          return self[do(object)]
    def __xor__(self, object=slice(None)):             return excepts(object)[self]
    def __or__(self, object=slice(None)):         return ifnot(self)[object] # There is no reasonable way to make this an attribute?
    def __and__(self, object=slice(None)):        return ifthen(self)[object]
    def __pow__(self, object=slice(None)):        return instance(object)[self]
    
    __pos__ = partialmethod(__getitem__, bool)
    __neg__ = partialmethod(__getitem__, not_)
    __invert__ = functions.__reversed__
    
    def __dir__(self):
        return super().__dir__() + dir(_composition_attr(self._attributes_))
                
compose._attributes_.insert( 0, dict(fnmatch=partial(flip, __import__('fnmatch').fnmatch)))
compose._attributes_.append({
        k: (partial if k.endswith('getter') or k.endswith('caller') else flip)(v)
        for k, v in vars(__import__('operator')).items()})


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
    def __init__(self, condition=bool, data=None):
        setattr(self, 'condition', condition) or super().__init__(data)
        
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


class FalseException(object):
    def __init__(self, exception): self.exception = exception
    def __bool__(self):  return False
    def __repr__(self): return repr(self.exception)

class excepts(compose):
    __slots__ = 'exceptions', 'data'
    def __init__(self, exceptions=None, data=None):
        setattr(self, 'exceptions', exceptions) or super().__init__(data)
    
    def __call__(self, *args, **kwargs):
        try: return super(excepts, self).__call__(*args, **kwargs)
        except self.exceptions as e:
            return FalseException(e)


def right_attr(self, attr, object):
    return compose()[object].__getattr__(attr)(self[:])

def op_attr(self, attr, value): 
    if isinstance(self, factory): self = self[:]
    self = object.__getattribute__(self, attr)(value)
    return self
        
for other in ['mul', 'add', 'rshift' ,'sub', 'and', 'or', 'xor', 'truediv', 'floordiv', 'matmul', 'mod', 'lshift', 'pow']:
    setattr(compose, dunder('i'+other), partialmethod(op_attr, dunder(other))) 
    setattr(compose, dunder('r'+other), partialmethod(right_attr, dunder(other)))

for key, attr in zip(('do', 'excepts', 'instance'), ('lshift', 'xor', 'pow')):
    setattr(compose, key, getattr(compose, dunder(attr)))

del other, key, attr


class factory(compose):
    __slots__ = 'args', 'kwargs', 'data'
    def __init__(self, args=None, kwargs=None):
        self.args, self.kwargs, self.data = args, kwargs, list()
        
    def __getitem__(self, attr):
        if attr == slice(None): return compose()
        if attr in IGNORE: return self
        if isinstance(self.args, tuple) and  isinstance(self.kwargs, dict):
            attr = partial(attr, *self.args, **self.kwargs)
        return compose()[attr]
            
    def __call__(self, *args, **kwargs):
        if isinstance(self.args, tuple) and  isinstance(self.kwargs, dict):
            return next(concatv(self.args, args))
        return factory(args, kwargs)
    
    __mul__ = __add__ = __rshift__ = __sub__ = push = __getitem__

a = an = the = then = λ = factory(compose)


class memo(compose):
    def __init__(self, cache=None, data=None):
        self.cache = dict() if cache is None else getattr(data, 'cache', cache)
        super().__init__(data)

    def memoize(self): return memoize(super().__call__, cache=self.cache)
    __call__ = property(memoize)


class parallel(compose):
    def __init__(self, jobs=4, data=None):
        self.jobs = jobs
        super().__init__(data)
        
    def map(self, function):
        return super().__getattr__('map')(__import__('joblib').delayed(function))
    
    def __call__(self, *args, **kwargs):
        return __import__('joblib').Parallel(self.jobs)(super().__call__(*args, **kwargs))
        
    __truediv__ = map


def stargetter(attr, *args, **kwargs):
    def __call__(self, object):
        object = attrgetter(attr)(object)
        return object(*args, **kwargs) if callable(object) else object 


class this(compose):
    def __getattr__(self, attr):
        def wrapped(*args, **kwargs):
            return self.data.append(partial(stargetter, attr, *args, **kwargs)) or self
        return wrapped
    
    def __getitem__(self, attr):
        return super().__getitem__(itemgetter(attr) if isinstance(attr, str) else attr)

class star(compose):
    """Call a function starring the arguments for sequences and starring the keywords for containers."""
    def __call__(self, *inputs):
        args, kwargs = list(), dict()
        for input in inputs:
            if isinstance(input, dict): kwargs.update(**input)
            else:                       args += list(input)
        return super().__call__(*args, **kwargs)


def load_ipython_extension(ip=None):
    ip = ip or __import__('IPython').get_ipython()
    if ip: ip.Completer.use_jedi = False
load_ipython_extension()


if __name__ == '__main__':
    get_ipython().system('jupyter nbconvert --to python --TemplateExporter.exclude_input_prompt=True articles.ipynb')

