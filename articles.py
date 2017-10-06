
# coding: utf-8

# `articles` are `callable` user defined lists in python. Use arthimetic and list operations to compose dense higher-order functions.

from functools import singledispatch, partialmethod, wraps, partial
from itertools import zip_longest
from collections import ChainMap
from operator import attrgetter
from toolz.curried import first, isiterable, identity, count, get, concat, flip, map, groupby, filter, reduce
from copy import copy
__all__ = 'a', 'an', 'the', 'then', 'f', 'star', 'flip', 'do', 
from operator import not_
from collections import UserList, OrderedDict

dunder = '__{}__'.format


# # Composition

class compose(UserList):
    """The main class for function composition."""
    
    # __kwdefaults__ contains default arguments and values
    __kwdefaults__ = ['data', list()],
        
    def __new__(cls, *args, **kwargs):
        if not isinstance(cls.__kwdefaults__, OrderedDict):
            cls.__kwdefaults__ = OrderedDict(cls.__kwdefaults__)
        cls.__slots__ = tuple(cls.__kwdefaults__.keys())
        cls.__iter__ = partialmethod(iter)
        return super().__new__(cls)
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        for i, (slot, arg) in enumerate(zip_longest(self.__slots__, args)):
            default = self.__kwdefaults__[slot]
            if i >= len(args):
                arg = copy(default)
                
            arg = kwargs.pop(slot, arg)
            
            if isiterable(default):
                if not isiterable(arg):
                    arg = type(default)([arg])
                if not isinstance(arg, type(default)):
                    arg = type(default)(arg)
            setattr(self, slot, arg)
                                
    def __getattr__(self, attr, *args, **kwargs):
        try:
            return object.__getattr__(self, attr)
        except: pass
        value = callable(attr) and attr or self._attributes_[attr]
        if attr is value:
            if args or kwargs:
                return self[partial(value, *args, **kwargs)]
            return self[value]
        def wrapper(*args, **kwargs):
            nonlocal value
            self[
                value(*args, **kwargs) if type(value) is partial
                else partial(value, *args, **kwargs) if args or kwargs
                else value]
            return self
        return wraps(getattr(value, 'func', value))(wrapper)
        
    __truediv__ = partialmethod(__getattr__, map)
    __floordiv__ = partialmethod(__getattr__, filter)
    __matmul__ = partialmethod(__getattr__, groupby)
    __mod__ = partialmethod(__getattr__, reduce)


    def __getitem__(self, object):
        # An empty slice returns self
        if object == slice(None):
            return self        
        if isinstance(object, (int, slice)): 
            try:
                return self.data[object]
            except IndexError as e:
                raise e
        # An iterable object is evaluated a callable map.
        if isiterable(object) and not isinstance(object, (str, compose)):
            object = juxt(object)
        # An other object is included in the composition.
        return self.append(object) or self


    def __dir__(self):
        """List the attributes available on the object."""
        return list(super().__dir__()) + dir(self._attributes_)
    
    def __call__(self, *args, **kwargs):
        """Call an iterable as a function evaluating the arguments in serial."""
        
        try:
            if args[0] in attrgetter('tqdm', 'tqdm_notebook')(__import__('tqdm')):
                self, args = args[0](self), args[1:]
        except: pass
            
        for value in self:
            args, kwargs = (
                # Return the value of non-callables, they are constants
                [value] if not callable(value) 
                # Otherwise call the function
                else [value(*args, **kwargs)]), dict()
        return args[0] if len(args) else None    
    
    def __lshift__(self, object): return compose([self, do(object)])
    def __xor__(self, object): return compose([excepts(object, self)])
    def __or__(self, object): return ifnot(self, object)
    def __and__(self, object): return ifthen(self, object)
    def __pow__(self, object): return instance(object)
    
    
    def __copy__(self):
        new = type(self)(*map(copy, self.__getstate__()))
        new.data = list(map(copy, self.data))
        return new

    def __exit__(self, exc_type, exc_value, traceback):
        pass
    
    __enter__ = __deepcopy__ = __copy__
    
    # Unary operations. 
    __abs__ = __call__
    
    def __pos__(self): return self[bool]
    
    def __neg__(self): return self[not_]
    
    def __reversed__(self): 
        return type(self)(super().__reversed__())
    
    __invert__ = __reversed__
    
    # State operations
    def __getstate__(self):
        return tuple(map(self.__dict__.get, self.__slots__))
    
    def __setstate__(self, state):
        for key, value in zip(self.__slots__, state):
            setattr(self, key, value)

    def __repr__(self):
        return (type(self).__name__ or 'λ').replace('compose', 'λ') + '>' + ':'.join(map(repr, self.__getstate__()))
        
    
    def __hash__(self):
        return hash(tuple(self))
    
    __name__ = property(__repr__)
    __mul__ = __add__ = __rshift__ = __sub__ = __getitem__
    
    _attributes_ = dict()
    
    def __dir__(self):
        return super().__dir__() + dir(self._attributes_)


class attributes(ChainMap):
    def __getitem__(self, key):
        for mapping in self.maps:
            try:
                value = getattr(mapping, '__dict__', mapping)[key]
                if callable(value):
                    return (type(mapping) is type and flip or identity)(value)
            except KeyError: 
                pass
        try:
            return self.new_child(type(key) is str and __import__(key) or key)
        except:
            raise AttributeError(key)
        
    def __dir__(self): 
        return concat(map(lambda x: [
            k for k, v in getattr(x, '__dict__', x).items() if callable(v)
        ], self.maps))

compose._attributes_ = attributes()['builtins']['pathlib'][__import__('pathlib').Path].new_child({
        k: (
            partial if k.endswith('getter') or k.endswith('caller')
            # some need to flip
            else flip)(v)
        for k, v in vars(__import__('operator')).items()
    })['json']['requests'][__import__('requests').Response]['toolz']


class juxt(compose):
    """Any mapping is a callable, call each of its elements."""
    __kwdefaults__ = ['data', list()], ['type', tuple]
    def __init__(self, data=None, _type=tuple):
        super().__init__()
        if isiterable(data) and not isinstance(data, type(self).__mro__[1]):
            self.type = type(data)
        self.data = list(data.items()) if issubclass(self.type, dict) else list(data) or list()

    def __call__(self, *args, **kwargs):
        result = list()
        for callable in self.data:
            if not isinstance(callable, (str, compose)) and isiterable(callable):
                callable = juxt(callable)
            if not isinstance(callable, compose):
                callable = compose([callable])            
            result.append(callable(*args, **kwargs))
        return self.type(result)


class flip(compose):
    """Call a function with the positional arguments reversed"""
    def __call__(self, *args, **kwargs):
        return super(flip, self).__call__(*reversed(args), **kwargs)


class do(compose):
    """Call a function and return input argument."""
    def __call__(self, *args, **kwargs):
        super(do, self).__call__(*args, **kwargs)
        return args[0] if args else None


class star(compose):
    """Call a function starring the arguments for sequences and starring the keywords for containers."""
    def __call__(self, *args, **kwargs):
        args = args[0] if len(args) is 1 else (args,)
        if not isiterable(args): 
            args = [(args,)]
        if isinstance(args, dict):
            args = kwargs.update(args) or tuple()
        return super(star, self).__call__(*args, **kwargs)


class condition(compose):
    condition = None
    __kwdefaults__ = ['condition', compose()], ['data', list()]
    
    def __call__(self, *args, **kwargs):
        if not self: 
            return True
        return super().__call__(*args, **kwargs)


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
    def __init__(self, object=None, data=None):        
        if isinstance(object, type):
            object = object,            
        if isinstance(object, tuple):
            object = partial(flip(isinstance), object)
        super().__init__(object, data or list())


class excepts(compose):
    """Allow acception when calling a function"""
    exceptions = None
    __kwdefaults__ = ['exceptions', tuple()], ['data', compose()]
    
    def __call__(self, *args, **kwargs):
        try:
            return super(excepts, self).__call__(*args, **kwargs)
        except self.exceptions as e:
            return e


class stack(compose):
    """A composition stack with push and pop methods.  It chains compositions together
    allowing a chainable api to map, filter, reduce, and groupby functions.|
    """
    __kwdefaults__ = ['data', list([compose()])], 
    
    def __init__(self, *args):
        super().__init__(*args)
        # nested copy
        self.data = list(map(copy, self.data))

    def push(self, type=compose, *args):
        if not isinstance(type, compose):
            type = type(*args)
        not self and self.pop()
        return self.append(type) or self
    
    def pop(self, *args):
        self.data.pop(*args)
        return self
    
    def __getitem__(self, *args, **kwargs):
        if object == slice(None):
            return self        
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
    
    __getattr__ = compose.__getattr__

    def __bool__(self):
        return any(self)
    
    stack = partialmethod(push)
    ifthen = partialmethod(push, ifthen)
    ifnot = partialmethod(push, ifnot)
    excepts = partialmethod(push, excepts)
    __pow__ = instance = partialmethod(push, instance)   
    do = partialmethod(push, do)
    __mul__ = __add__ = __rshift__ = __sub__ = __getitem__


def right_attr(self, attr, other):
    """Add the right attribute operations to the function"""
    return getattr(type(self)([other]), attr)(self)

for other in ['mul', 'add', 'rshift' ,'sub', 'and', 'or', 'xor', 'truediv', 'floordiv', 'matmul', 'mod', 'lshift']:
    setattr(compose, dunder('i'+other), getattr(compose, dunder(other)))
    setattr(compose, dunder('r'+other), partialmethod(right_attr, dunder(other)))


class call(stack):
    args, kwargs = tuple(), dict()
                                
    def __getattr__(self, attr):
        return stack().__getattr__(attr)

    def __getitem__(self, attr, *args, **kwargs):
        if attr == slice(None): return stack()
        return stack().__getitem__(attr, *args, **kwargs)
        
    def __call__(self, *args, **kwargs):     
        self = type(self)()
        self.args, self.kwargs = args, kwargs
        return self
    
    def __pow__(self, object): return stack()**object
    __mul__ = __add__ = __rshift__ = __sub__ = push = __getitem__


a = an = the = then = f = call()

