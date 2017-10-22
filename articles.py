
# coding: utf-8

from collections import UserList
from functools import partialmethod, wraps
from inspect import getfullargspec
from itertools import zip_longest, starmap
from operator import attrgetter, not_, eq, methodcaller, itemgetter
from toolz.curried import isiterable, identity, concat, flip, cons
from toolz import map, groupby, filter, reduce
from copy import copy
dunder = '__{}__'.format
__all__ = 'a', 'an', 'the', 'star', 'do', 'flip', 'compose', 'composite', 'λ', 'this'


class functions(UserList):
    """A composition of functions."""
    def __init__(self, data=None):
        if data and not isiterable(data):
            data = [data]
        super().__init__(data or list())
    
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
    __abs__ = __call__
    
    def __reversed__(self): 
        self.data = type(self.data)(reversed(self.data))
        return self
    
    def __repr__(self):
        return (type(self).__name__ or 'λ').replace('compose', 'λ') + '>' + ':'.join(map(repr, self.__getstate__()))   
    __name__ = property(__repr__)
    
    def __hash__(self): return hash(tuple(self))
    def __bool__(self): return any(self.data)
    def __getstate__(self, state=None):
        keys = getfullargspec(type(self)).args[1:]
        if state is None:
            return tuple(map(self.__dict__.get, keys))
        for key, value in zip(keys, state):
            setattr(self, key, value)
            
    __setstate__ = __getstate__

    def __getitem__(self, object):
        if object == slice(None): return self        
        if isinstance(object, (int, slice)): 
            try:
                return self.data[object]
            except IndexError as e: raise e
        return self.append(object) or self


class partial(__import__('functools').partial):
    def __eq__(self, other, result = False):
        if isinstance(other, partial):
            result = True
            for a, b in zip_longest(*(cons(_.func, _.args) for _ in [self, other])):
                result &= (a is b) or (a == b)
        return result


class attributes(functions):
    def __getitem__(self, key):
        try:
            return super().__getitem__(type(key) is str and __import__(key) or key)
        except:
            raise AttributeError(key)

    def __call__(self, key):
        for mapping in reversed(self):
            try:
                value = getattr(mapping, '__dict__', mapping)[key]
                if callable(value):
                    return (type(mapping) is type and flipped or identity)(value)
            except KeyError as e: pass
        else: raise e

    def __dir__(self): 
        return [
            key for object in self.data 
            for key, value in getattr(object, dunder('dict'), object).items()
            if callable(value)]


class compose(functions):
    """A composition of functions."""
    _attributes_ = attributes()
    
    def __getattr__(self, attr, *args, **kwargs):
        try:
            return object.__getattr__(self, attr)
        except: pass
        
        value = callable(attr) and attr or self._attributes_(attr)
        if attr is value:
            if args or kwargs:
                return self[partial(value, *args, **kwargs)]
            return self[value]
        def wrapper(*args, **kwargs):
            nonlocal value
            (self.data[-1] if isinstance(self, composite) else self)[
                value(*args, **kwargs) if type(value) == partial
                else partial(value, *args, **kwargs) if args or kwargs
                else value]
            return self
        
        return wraps(getattr(value, 'func', value))(wrapper)
        
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
        return super().__dir__() + dir(self._attributes_)
    
compose._attributes_['inspect']['builtins']['collections']['pathlib'][__import__('pathlib').Path][{
        k: (partial if k.endswith('getter') or k.endswith('caller') else flip)(v)
        for k, v in vars(__import__('operator')).items()
}]['json']['requests'][__import__('requests').Response]['toolz'][dict(fnmatch=flip(__import__('fnmatch').fnmatch))];


class do(compose):
    def __call__(self, *args, **kwargs):
        super(do, self).__call__(*args, **kwargs)
        return args[0] if args else None
    
class flipped(compose):
    def __call__(self, *args, **kwargs):
        return super().__call__(*reversed(args), **kwargs)


class juxt(compose):
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
    def __init__(self, exception):        
        self.exception = exception
    def __bool__(self):  return False

class excepts(compose):
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
        return wraps(super(composite, self).__getattr__(attr))(wrapped)
        
    def push(self, type=compose, *args):
        self = self[:]
        if not isinstance(type, compose): type = type(*args)
        not self and self.pop()
        return self.append(type) or self
    

    def __getitem__(self, *args, **kwargs):
        if isinstance(self, factory): self = composite()
        if object == slice(None): return self
        if args and isinstance(args[0], (int, slice)): 
            try:
                return self.data[args[0]]
            except IndexError as e:
                raise e
        try:
            self[-1].__getitem__(*args, **kwargs)
        except AttributeError:
            self.push()
            self[-1].__getitem__(*args, **kwargs)
        return self    
        
    __mul__ = __add__ = __rshift__ = __sub__ = __getitem__

for cls in [ifthen, ifnot, excepts, do, instance]: 
    setattr(composite, cls.__name__, partialmethod(composite.push, cls))


def right_attr(self, attr, other):
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
    args, kwargs = tuple(), dict()

    def __getitem__(self, attr):
        if attr == slice(None): return composite()
        if self.args or self.kwargs:
            attr = partial(attr, *self.args, **self.kwargs)
        return super().__getitem__(attr)
        
    def __call__(self, *args, **kwargs):     
        self = type(self)()
        self.args, self.kwargs = args, kwargs
        return self
    
    __mul__ = __add__ = __rshift__ = __sub__ = push = __getitem__

a = an = the = λ = factory()


class stargetter:
    def __init__(self, attr, *args, **kwargs):
        self.attr, self.args, self.kwargs = attr, args, kwargs

    def __call__(self, object):
        object = attrgetter(self.attr)(object)
        if callable(object):
            return object(*self.args, **self.kwargs)
        return object

    
class this(compose):
    class this_attributes(attributes):
        def __call__(self, attr):
            return partial(stargetter, attr)

    _attributes_ = this_attributes()
    def __getitem__(self, attr):
        if isinstance(attr, str):
            return self[itemgetter(attr)]
        return super().__getitem__(attr)

class star(compose):
    """Call a function starring the arguments for sequences and starring the keywords for containers."""
    def __call__(self, *args, **kwargs):
        args = args[0] if len(args) is 1 else (args,)
        if not isiterable(args): 
            args = [(args,)]
        if isinstance(args, dict):
            args = kwargs.update(args) or tuple()
        return super(star, self).__call__(*args, **kwargs)


# !jupyter nbconvert --to python --TemplateExporter.exclude_input_prompt=True articles.ipynb
