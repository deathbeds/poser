
# coding: utf-8

from functools import total_ordering, singledispatch, partialmethod, wraps
from itertools import zip_longest, chain
from toolz.curried import first, isiterable, partial, identity, count, get, concat
from copy import copy
__all__ = 'a', 'an', 'the', 'then', 'f', 'star', 'flip', 'do', 
from collections import UserList, OrderedDict

dunder = '__{}__'.format


class compose(UserList):
    _attributes_ = list([
        str, __import__('six').moves.builtins, __import__('pathlib'), __import__('json'), 
        __import__('pathlib').Path, __import__('toolz').curried, __import__('operator'), __import__('collections'),
    ])

    def __getattr__(self, attr):
        if hasattr(type(self), attr):
            return getattr(type(self), attr)(self)
        for object in self._attributes_:
            decorate = isinstance(object, type) and flip or compose
            object = getattr(object, '__dict__', object)
            if attr in object:
                @wraps(object[attr])
                def wrapper(*args, **kwargs):
                    return self.append(partial(decorate(object[attr]), *args, **kwargs))
                return wrapper
        raise AttributeError(attr)
        
    def __getitem__(self, object):
        if object == slice(None):
            return self
        if isinstance(object, tuple):
            object = juxt(object)
        if callable(object):
            return self.append(object)
        return super().__getitem__(object)

        
    def __dir__(self):
        return list(super().__dir__()) + list(concat(map(lambda x: getattr(x, '__dict__', x).keys(), self._attributes_)))
    
    def __call__(self, *args, **kwargs):
        for callable in self:
            args, kwargs = [callable(*args, **kwargs)], dict()
        return args[0] if len(args) else None    
    
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
    __deepcopy__ = __copy__ 

    __abs__ = __call__
    __enter__ = __copy__
    
    def _condition_attr_(self, callable, object):
        return type(self)().append(callable(self, object))
    
    def _list_attr_(self, attr, *args):
        return getattr(super(), attr)(*args) or self
    
    def _right_attr_(self, attr, other):
        return getattr(
            compose([other]), attr.replace('__r', '__')
        )(self)

    def __pow__(self, object):
        if isinstance(object, type):
            object = object,
        if not isinstance(object, tuple):
            object = partial(flip(isinstance), object)
        return self._condition_attr_(ifthen, object)


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


class stack(compose):
    __kwdefaults__ = ['data', list([compose()])], 
    
    def __init__(self, *args):
        super().__init__(*args)
        self.data = list(map(copy, self.data))

    def _list_attr_(self, attr, *args):
        if attr == 'pop':
            self.data.pop(*args)
            return self
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


class call(stack):
    def append(self, object):
        new = type(self).__mro__[1]()
        if self.args or self.kwargs:
            object = partial(object, *self.args, **self.kwargs)
        return new.append(object)
    
    args, kwargs = tuple(), dict()    
    def __call__(self, *args, **kwargs):     
        self = type(self)()
        self.args, self.kwargs = args, kwargs
        return self


for cls in [compose, stack, call]:
    for attr in dir(UserList):
        if attr[0].islower() and call is not cls:
            setattr(cls, attr, partialmethod(cls._list_attr_, attr))
        
    for other in ['mul', 'add', 'rshift' ,'sub']:
        setattr(cls, dunder(other), getattr(cls, 'append'))

compose.__and__ = partialmethod(compose._condition_attr_, step)
compose.__or__ = partialmethod(compose._condition_attr_, ifnot)
compose.__xor__ = partialmethod(compose._condition_attr_, excepts)
compose.__truediv__ = property(partial(flip(compose.__getattr__), 'map'))
compose.__floordiv__ = property(partial(flip(compose.__getattr__), 'filter'))
compose.__matmul__ = property(partial(flip(compose.__getattr__), 'groupby'))
compose.__mod__ = property(partial(flip(compose.__getattr__), 'reduce'))
compose.__lshift__ = property(partial(flip(compose.__getattr__), 'do'))

for other in ['mul', 'add', 'rshift' ,'sub', 'and', 'or', 'xor', 'truediv', 'floordiv', 'matmul', 'mod', 'lshift']:
    setattr(compose, dunder('i'+other), getattr(compose, dunder(other)))
    setattr(cls, dunder('r'+other), partialmethod(cls._right_attr_, dunder('r'+other)))


a = an = the = then = f = call()




