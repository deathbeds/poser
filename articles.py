
# coding: utf-8

# `articles` are `callable` user defined lists in python. Use arthimetic and list operations to compose dense higher-order functions.

from functools import singledispatch, partialmethod, wraps
from itertools import zip_longest
from collections import ChainMap
from toolz.curried import first, flip, isiterable, partial, identity, count, get, concat
from copy import copy
__all__ = 'a', 'an', 'the', 'then', 'f', 'star', 'flip', 'do', 
from collections import UserList, OrderedDict

dunder = '__{}__'.format


def methods(cls):
    # attach all list methods to the cls
    for attr in dir(UserList):
        if attr[0].islower():
            setattr(cls, attr, partialmethod(cls._list_attr_, attr))
        
    # attach the *, +, >>, - operators
    for other in ['mul', 'add', 'rshift' ,'sub']:
        setattr(cls, dunder(other), getattr(cls, 'append'))
    return cls


@methods
class compose(UserList):
    __kwdefaults__ = ['data', list()],
        
    def __new__(cls, *args, **kwargs):
        if not isinstance(cls.__kwdefaults__, OrderedDict):
            cls.__kwdefaults__ = OrderedDict(cls.__kwdefaults__)
        cls.__slots__ = tuple(cls.__kwdefaults__.keys())
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
         
    def __getattr__(self, attr):
        if hasattr(type(self), attr):
            return getattr(type(self), attr)(self)
        def wrapper(*args, **kwargs):
            callable = self._attributes_[attr]
            if args or kwargs:
                callable = partial(callable, *args, **kwargs)
            return self.append(callable)
        return wraps(self._attributes_[attr].data[0])(wrapper)
        
    def __getitem__(self, object):
        if object == slice(None):
            return self
        if isinstance(object, tuple):
            object = juxt(object)
        if callable(object):
            return self.append(object)
        return super().__getitem__(object)

    def __dir__(self):
        return list(super().__dir__()) + dir(self._attributes_)
    
    def __call__(self, *args, **kwargs):
        for callable in self:
            args, kwargs = [callable(*args, **kwargs)], dict()
        return args[0] if len(args) else None    
            
    def __getstate__(self):
        return tuple(map(partial(getattr, self), self.__slots__))
    
    def __setstate__(self, state):
        for key, value in zip(self.__slots__, state):
            setattr(self, key, value)
        
    def __copy__(self):
        new = type(self)()
        new.__setstate__(tuple(map(copy, self.__getstate__()))) or new
        new.data = list(map(copy, self))
        return new
    
    def __hash__(self):
        return hash(tuple(self))
            
    def __exit__(self, exc_type, exc_value, traceback):
        pass
                
    def __repr__(self):
        return ':'.join(map(repr, self.__getstate__()))
    
    
    def _list_attr_(self, attr, *args):
        return getattr(super(), attr)(*args) or self

    def __pow__(self, object):
        if isinstance(object, type):
            object = object,
        if not isinstance(object, tuple):
            object = partial(flip(isinstance), object)
        return self._condition_attr_(ifthen, object)

    __abs__ = __call__
    __enter__ = __deepcopy__ = __copy__
    
    __truediv__ = property(partial(flip(__getattr__), 'map'))
    __floordiv__ = property(partial(flip(__getattr__), 'filter'))
    __matmul__ = property(partial(flip(__getattr__), 'groupby'))
    __mod__ = property(partial(flip(__getattr__), 'reduce'))
    __lshift__ = property(partial(flip(__getattr__), 'do'))


class juxt(compose):
    def __call__(self, *args, **kwargs):
        return tuple((isinstance(callable, tuple) and juxt or identity)(callable)(*args, **kwargs) for callable in self)


class flip(compose):
    """Call a function with the arguments positional arguments reversed"""
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
    """Evaluate a function if a condition is true."""
    __kwdefaults__ = ['condition', compose()], ['data', list()]


class ifthen(condition):
    def __call__(self, *args, **kwargs):
        return self.condition(*args, **kwargs) and super(ifthen, self).__call__(*args, **kwargs)

class ifnot(condition):
    def __call__(self, *args, **kwargs):
        return self.condition(*args, **kwargs) or super(ifnot, self).__call__(*args, **kwargs)


class step(condition):
    def __call__(self, *args, **kwargs):
        result = self.condition(*args, **kwargs)
        return result and super(step, self).__call__(result)


class excepts(compose):
    """Allow acception when calling a function"""
    __kwdefaults__ = ['data', identity], ['exceptions', tuple()], 
    
    def __call__(self, *args, **kwargs):
        try:
            return super(excepts, self).__call__(*args, **kwargs)
        except self.exceptions as e:
            return e


@methods
class stack(compose):
    __kwdefaults__ = ['data', list([compose()])], 
    
    def __init__(self, *args):
        super().__init__(*args)
        self.data = list(map(copy, self.data))

    def _list_attr_(self, attr, *args):
        try:
            self[-1]._list_attr_(attr, *args)
        except AttributeError:
            self.push()
            self[-1]._list_attr_(attr, *args)
        return self    

    def push(self):
        self.data.append(compose())
        return self

    def __bool__(self):
        return any(map(bool, self))
    
def _pop(self, *args):
    self.data.pop(*args)
    return self

stack.pop = _pop


class call(stack):
    args, kwargs = tuple(), dict()    
    
    def append(self, object=None):
        new = type(self).__mro__[1]()
        if self.args or self.kwargs:
            object = partial(object, *self.args, **self.kwargs)
        return new.append(object)
    
    def __getitem__(self, object):
        if object == slice(None):  return type(self).__mro__[1]()
        return super().__getitem__(object)
    
    def __call__(self, *args, **kwargs):     
        self = type(self)()
        self.args, self.kwargs = args, kwargs
        return self
    


def _condition_attr_(self, callable, object):
    """"""
    return type(self)().append(callable(self, object))

def _right_attr_(self, attr, other):
    return getattr(compose([other]), attr)(self)


compose.__and__ = partialmethod(_condition_attr_, step)
compose.__or__ = partialmethod(_condition_attr_, ifnot)
compose.__xor__ = partialmethod(_condition_attr_, excepts)

for other in ['mul', 'add', 'rshift' ,'sub', 'and', 'or', 'xor', 'truediv', 'floordiv', 'matmul', 'mod', 'lshift']:
    setattr(compose, dunder('i'+other), getattr(compose, dunder(other)))
    setattr(compose, dunder('r'+other), partialmethod(_right_attr_, dunder(other)))


class attributes(ChainMap):
    def __getitem__(self, key):
        for mapping in self.maps:
            try:
                object = getattr(mapping, '__dict__', mapping)[key]
                return (
                    not isinstance(object, compose) 
                    and isinstance(mapping, type) and flip or compose
                )(object)
            except KeyError:
                pass
        raise AttributeError(key)
        
    def __dir__(self):
        return concat(map(lambda x: getattr(x, '__dict__', x).keys(), self.maps))


compose._attributes_ = attributes(*map(__import__, ['builtins', 'pathlib', 'operator', 'json', 'toolz'])).new_child(__import__('pathlib').Path)


a = an = the = then = f = call()

