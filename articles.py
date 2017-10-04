
# coding: utf-8

# `articles` are `callable` user defined lists in python. Use arthimetic and list operations to compose dense higher-order functions.

from functools import singledispatch, partialmethod, wraps
from itertools import zip_longest
from collections import ChainMap
from toolz.curried import first, isiterable, partial, identity, count, get, concat, flip
from copy import copy
__all__ = 'a', 'an', 'the', 'then', 'f', 'star', 'flip', 'do', 
from operator import not_
from collections import UserList, OrderedDict

dunder = '__{}__'.format


def append_methods(cls, ignore=set()):
    """Methods to append functions to a composition."""
    # mimic **all** list methods onto the **cls**
    for attr in dir(UserList):
        if attr not in ignore and attr[0].islower():
            setattr(cls, attr, partialmethod(list_attr, attr))
    # attach the *, +, >>, - operators to the **cls**s
    for other in ['mul', 'add', 'rshift' ,'sub']:
        setattr(cls, dunder(other), getattr(cls, 'append'))        
    return cls


def list_attr(self, attr, *args): 
    """Surrogate function to map userlist attributes to the composition"""
    return getattr(self.data, attr)(*args) or self


# # Composition

@append_methods
class compose(UserList):
    """The main class for function composition."""
    
    # __kwdefaults__ contains default arguments and values
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
         
    # getattr stays here because some operations are defined below
    def __getattr__(self, attr):
        if hasattr(type(self), attr):
            return getattr(type(self), attr)(self)
        def wrapper(*args, **kwargs):
            callable = self._attributes_[attr]
            if args or kwargs:                    
                callable = partial_attr(callable, *args, **kwargs)
            return self.append(callable)
        return wraps(self._attributes_[attr].data[0])(wrapper)
        
    def __getitem__(self, object):
        # An empty slice returns self
        if object == slice(None):
            return self
        # An iterable object is evaluated a callable map.
        if isiterable(object) and not isinstance(object, str):
            object = juxt(object)
        # An other object is included in the composition.
        if callable(object):
            return self.append(object)
        return super().__getitem__(object)

    def __dir__(self):
        """List the attributes available on the object."""
        return list(super().__dir__()) + dir(self._attributes_)
    
    def __call__(self, *args, **kwargs):
        """Call an iterable as a function evaluating the arguments in serial."""
        for value in self:
            args, kwargs = (
                # Return the value of non-callables, they are constants
                value if not callable(value) 
                # Otherwise call the function
                else [value(*args, **kwargs)]), dict()
        return args[0] if len(args) else None    
    
    
    def __pow__(self, object):
        """a**(int,) is equivalent to typing checking"""
        
        # Make sure the object is a tuple if it is in a class.
        if isinstance(object, type):
            object = object,
            
        if isinstance(object, tuple):
            object = partial(flip(isinstance), object)
        
        # Step through the function if the condition is true
        return condition_attr(self, flip(ifthen), object)
    
    def __copy__(self):
        new = type(self)()
        new.__setstate__(tuple(map(copy, self.__getstate__()))) or new
        new.data = list(map(copy, self))
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
        return tuple(map(partial(getattr, self), self.__slots__))
    
    def __setstate__(self, state):
        for key, value in zip(self.__slots__, state):
            setattr(self, key, value)

    def __repr__(self):
        return ':'.join(map(repr, self.__getstate__()))
        
    
    def __hash__(self):
        return hash(tuple(self))


def partial_attr(callable: 'compose', *args, **kwargs):
    # Attributes with partials are called immediately on the chain like `attrgetter`.
    if type(callable[0]) is __import__('functools').partial:
        return callable(*args, **kwargs) 
    return partial(callable, *args, **kwargs) 


# ## Compositions

class juxt(compose):
    """Any mapping is a callable, call each of its elements."""
    __kwdefaults__ = ['data', list()], ['type', tuple]
    def __init__(self, *args):
        super().__init__(*args)
        if isiterable(args[0]) and not isinstance(args[0], type(self).__mro__[1]):
            self.type = type(args[0])
            
    def __call__(self, *args, **kwargs):
        result = list()
        for callable in self:
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


# ## Conditional Compositions

class condition(compose):
    __kwdefaults__ = ['condition', compose()], ['data', list()]


class ifthen(condition):
    """Evaluate a function if a condition is true."""
    def __call__(self, *args, **kwargs):
        return self.condition(*args, **kwargs) and super(ifthen, self).__call__(*args, **kwargs)

class ifnot(condition):
    """Evaluate a function if a condition is false."""
    def __call__(self, *args, **kwargs):
        return self.condition(*args, **kwargs) or super(ifnot, self).__call__(*args, **kwargs)


class step(condition):
    """Evaluate a function only if a condition is true."""
    def __call__(self, *args, **kwargs):
        result = self.condition(*args, **kwargs)
        return result and super(step, self).__call__(result)


# ## Exception compositon

class excepts(compose):
    """Allow acception when calling a function"""
    __kwdefaults__ = ['data', identity], ['exceptions', tuple()], 
    
    def __call__(self, *args, **kwargs):
        try:
            return super(excepts, self).__call__(*args, **kwargs)
        except self.exceptions as e:
            return e


append_methods(condition, {'append'})


# ## Generic attributes
# 
# Append attributes to the composition object from other Python namespaces.
# 
#     'builtins', 'pathlib', 'operator', 'json', 'toolz'

@singledispatch
def resolve_attr(parent, object):
    """How to resolve decorated attribute values."""
    return (isinstance(object, compose) and identity or compose)(object)

@resolve_attr.register(partial)
def _resolve_attr(parent, object):
    return object

@resolve_attr.register(type)
def _resolve_attr(parent, object):
    return flip(object)


class attributes(ChainMap):
    def __getitem__(self, key):
        for mapping in self.maps:
            try:
                return resolve_attr(mapping, getattr(mapping, '__dict__', mapping)[key],)
            except KeyError: 
                pass
        try:
            return self.new_child(type(key) is str and __import__(key) or key)
        except:
            raise AttributeError(key)
        
    def __dir__(self):
        return concat(map(lambda x: getattr(x, '__dict__', x).keys(), self.maps))
        
compose._attributes_ = attributes()['builtins']['pathlib'][__import__('pathlib').Path].new_child(
    {
        k: (
            partial if k.endswith('getter')  
            # some need to flip
            else flip) (v)
        for k, v in vars(__import__('operator')).items()
    }
)['json']['toolz']


def condition_attr(self, callable, object):
    """Append attributes for condition compositions."""
    return type(self)().append(callable(self, object))

def right_attr(self, attr, other):
    """Add the right attribute operations to the function"""
    return getattr(compose([other]), attr)(self)

def extra_methods(cls):
    cls.__and__ = partialmethod(condition_attr, step)
    cls.__or__ = partialmethod(condition_attr, ifnot)
    cls.__xor__ = partialmethod(condition_attr, excepts)
    cls.__truediv__ = property(partial(flip(cls.__getattr__), 'map'))
    cls.__floordiv__ = property(partial(flip(cls.__getattr__), 'filter'))
    cls.__matmul__ = property(partial(flip(cls.__getattr__), 'groupby'))
    cls.__mod__ = property(partial(flip(cls.__getattr__), 'reduce'))
    cls.__lshift__ = property(partial(flip(cls.__getattr__), 'do'))

    for other in ['mul', 'add', 'rshift' ,'sub', 'and', 'or', 'xor', 'truediv', 'floordiv', 'matmul', 'mod', 'lshift']:
        setattr(cls, dunder('i'+other), getattr(compose, dunder(other)))
        setattr(cls, dunder('r'+other), partialmethod(right_attr, dunder(other)))
        
        
    return cls


extra_methods(compose)


# # Composition Stack

class stack(compose):
    """A composition stack with push and pop methods.  It chains compositions together
    allowing a chainable api to map, filter, reduce, and groupby functions.|
    """
    __kwdefaults__ = ['data', list([compose()])], 
    
    def __init__(self, *args):
        super().__init__(*args)
        # nested copy
        self.data = list(map(copy, self.data))

    def push(self, type=compose):
        self.data.append(type())
        return self
    
    def pop(self, *args):
        self.data.pop(*args)
        return self

    def append(self, *args, **kwargs):
        try:
            self.data[-1].append(*args, **kwargs)
        except AttributeError:
            self.push()
            self.data[-1].append(*args, **kwargs)
        return self    

    def __bool__(self):
        return any(map(bool, self))
    
def _pop(self, *args):
    self.data.pop(*args)
    return self

stack.pop = _pop


extra_methods(append_methods(stack, {'pop', 'append'}))


# # Composition Factory
# 
# `call` is a stack factory.  Use call to:
# 
# * Create partial - `a(*args, **kwargs)`
# * Generate new compositions.

class call(stack):
    args, kwargs = tuple(), dict()    
    
    def append(self, object=None):
        if self.args or self.kwargs:
            object = partial(object, *self.args, **self.kwargs)
        return stack().append(object)
    
    def __getitem__(self, object):
        if object == slice(None):  
            return stack()
        return super().__getitem__(object)
    
    def __pow__(self, object):
        return stack()**object
    
    def __call__(self, *args, **kwargs):     
        self = type(self)()
        self.args, self.kwargs = args, kwargs
        return self
        
extra_methods(append_methods(call, {'append'}))


a = an = the = then = f = call()

