
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
__all__ = 'a', 'an', 'the', 'star', 'do', 'λ', 'juxt', 'compose', 'parallel', 'memo', 'then', 'ifthen', 'ifnot', 'excepts', 'instance'


class functions(UserList):
    __slots__ = 'data',
        
    def __init__(self, data=None):
        super().__init__(data and not isiterable(data) and [data] or data or list())
        self.__qualname__ = __name__ + '.' + type(self).__name__
    
    def __call__(self, *args, **kwargs):
        for value in self:
            args, kwargs = ([value(*args, **kwargs)] if callable(value) else [value]), dict()
        return args[0] if len(args) else None    
    
    def __getitem__(self, object):
        if object == slice(None): return self        
        return self.data[object] if isinstance(object, (int, slice)) else self.append(object)
    
    def __getattr__(self, attr, *args, **kwargs):
        try:
            return object.__getattribute__(self, attr)
        except Exception as e:
            if callable(attr):
                return self[args or kwargs and partial(attr, *args, **kwargs) or attr]
            raise e

    def append(self, object):
        return  self.data.append(object) or not self.data[0] and self.data.pop(0) or self        
    
    def __repr__(self):
        return (type(self).__name__ or 'λ').replace('compose', 'λ') + ':' + super().__repr__()

    @property
    def __name__(self): return type(self).__name__
            
    @property
    def __annotations__(self): return getattr(self[0], dunder('annotations'), {})
    
    @property
    def __signature__(self): return signature(self[0])
    
    def __exit__(self, exc_type, exc_value, traceback): pass
    def __hash__(self): return hash(tuple(self))
    def __bool__(self): return any(self.data)
    def __abs__(self): return self.__call__
    def __reversed__(self): return type(self)(list(reversed(self.data)))      
    def __getstate__(self): return tuple(getattr(self, slot) for slot in self.__slots__)
    def __setstate__(self, state):
        for attr, value in zip(self.__slots__, state): setattr(self, attr, value)
            
    def __copy__(self, memo=None):
        new = type(self)()
        return new.__setstate__(self.__getstate__()) or new

    copy = __enter__ = __deepcopy__ = __copy__


class partial(__import__('functools').partial):
    def __eq__(self, other):
        return isinstance(other, partial) and all(
            (a is b) or (a == b) for a, b in zip_longest(*(cons(_.func, _.args) for _ in [self, other])))
    
    @property
    def __doc__(self): return getdoc(self.func)

class partial_attribute(partial):
    def __call__(self, object):
        return callable(self.func) and self.func(object, *self.args, **self.keywords) or self.func


class attributer(object):
    def __init__(self, maps=list(), parent=None, composition=None):
        self._maps, self.composition, self.parent = list(not isiterable(maps) and [maps] or maps), composition, parent

    @property
    def _map_(self): return slice(None) if self._maps[0] is getdoc else self._maps[0]
    
    @property
    def maps(self): return [getattr(object, dunder('dict'), object) for object in self._maps]
    
    def __getitem__(self, item):
        for raw, object in zip(self._maps, self.maps):
            if getattr(raw, dunder('name'), """""") == item: return object, None
            if item in object: 
                return object[item], raw
        raise AttributeError(item)

    def __dir__(self):
        keys = list()
        for raw, object in zip(self._maps, self.maps):
            keys += [getattr(raw, dunder('name'), """""")] + list(object.keys())
        return list(sorted(filter(bool, keys)))
            
    def __getattr__(self, value): return self.__class__(*self[value], self.composition)
    def __repr__(self): return repr(self._map_)
    
    @property
    def __doc__(self):
        try:
            return getdoc(self._map_) or inspect.getsource(self._map_)
        except: pass
    
    def __call__(self, *args, **kwargs):
        value = self._map_
        return self.composition[
            callable(value) and (
                isinstance(self.parent, type) and partial_attribute(value, *args, **kwargs)
                or type(value) is partial and not (value.args or value.keywords) and value.func(*args, **kwargs) 
                or (args or kwargs) and partial(value, *args, **kwargs)) or value]


class compose(functions):
    attributer = list(map(__import__, [
        'toolz', 'requests', 'builtins', 'json', 'pickle', 'io', 
        'collections', 'itertools', 'functools', 'pathlib', 'importlib', 'inspect']))    
    
    @property
    def __attributes__(self): return attributer(self.attributer, None, self[:])
    
    def __getattr__(self, attr, *args, **kwargs):
        if callable(attr): return self[:][partial(attr, *args, **kwargs)]
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
    
    def __dir__(self): return super().__dir__() + dir(attributer(self.attributer))
                
compose.attributer.insert( 0, dict(fnmatch=partial(flip, __import__('fnmatch').fnmatch)))
compose.attributer.append({
        k: (partial if k.endswith('getter') or k.endswith('caller') else flip)(v)
        for k, v in vars(__import__('operator')).items()})


class do(compose):
    def __call__(self, *args, **kwargs):
        super(do, self).__call__(*args, **kwargs)
        return args[0] if args else None


class juxt(compose):
    __slots__ = 'data', 'type'
    
    def __init__(self, data=None, type=None):
        if isiterable(data) and not isinstance(data, self.__class__.__mro__[1]):
            self.type = type or data.__class__ or tuple
        super().__init__(list(data.items()) if issubclass(self.type, dict) else list(data) or list())

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
    def __call__(self, *args, **kwargs):
        return self.condition(*args, **kwargs) and super(ifthen, self).__call__(*args, **kwargs)

class ifnot(condition):
    def __call__(self, *args, **kwargs):
        return self.condition(*args, **kwargs) or super(ifnot, self).__call__(*args, **kwargs)

class instance(ifthen):
    def __init__(self, condition=None, data=None):        
        if isinstance(condition, type): condition = condition,            
        if isinstance(condition, tuple): condition = partial(flip(isinstance), condition)
        super().__init__(condition, data)


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
    return object.__getattribute__(self[:], attr)(value)
        
for other in ['mul', 'add', 'rshift' ,'sub', 'and', 'or', 'xor', 'truediv', 'floordiv', 'matmul', 'mod', 'lshift', 'pow']:
    setattr(compose, dunder('i'+other), partialmethod(op_attr, dunder(other))) 
    setattr(compose, dunder('r'+other), partialmethod(right_attr, dunder(other)))

for key, other in zip(('do', 'excepts', 'instance'), ('lshift', 'xor', 'pow')):
    setattr(compose, key, getattr(compose, dunder(other)))

del other, key


class factory(compose):
    __slots__ = 'args', 'kwargs', 'data'
    def __init__(self, args=None, kwargs=None):
        self.args, self.kwargs, self.data = args, kwargs, list()
        
    def __getitem__(self, attr):
        if isinstance(self.args, tuple) and  isinstance(self.kwargs, dict):
            attr = attr == slice(None) and abs(self) or partial(attr, *self.args, **self.kwargs)
        return compose()[attr]
            
    def __call__(self, *args, **kwargs):
        return (
            next(concatv(self.args, args)) if isinstance(self.args, tuple) and  isinstance(self.kwargs, dict) 
            else factory(args, kwargs))
    
    __mul__ = __add__ = __rshift__ = __sub__ = push = __getitem__

a = an = the = then = λ = factory(compose)


class parallel(compose):
    def __init__(self, jobs=4, data=None):
        setattr(self, 'jobs', jobs) or super().__init__(data)
        
    def map(self, function):
        return super().__getattr__('map')(__import__('joblib').delayed(function))
    
    def __call__(self, *args, **kwargs):
        return __import__('joblib').Parallel(self.jobs)(super().__call__(*args, **kwargs))
        
    __truediv__ = map

class memo(compose):
    def __init__(self, cache=None, data=None):
        self.cache = dict() if cache is None else getattr(data, 'cache', cache)
        super().__init__(data)

    @property
    def __call__(self): return memoize(super().__call__, cache=self.cache)

class star(compose):
    def __call__(self, *inputs):
        args, kwargs = list(), dict()
        [kwargs.update(**input) if isinstance(input, dict) else args.extend(input) for input in inputs]
        return super().__call__(*args, **kwargs)


def load_ipython_extension(ip=__import__('IPython').get_ipython()):
    if ip: ip.Completer.use_jedi = False
load_ipython_extension()


if __name__ == '__main__':
    get_ipython().system('jupyter nbconvert --to python --TemplateExporter.exclude_input_prompt=True articles.ipynb')

