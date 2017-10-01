
# coding: utf-8

# # `callables`
# 
# Collections of special callable objects.

# In[1]:


from functools import total_ordering, singledispatch
from toolz.curried import first, isiterable, partial, identity, count
from copy import copy
from six import PY3
from typing import Mapping, Text, Sequence, Callable
from inspect import signature, getdoc
from operator import eq

__all__ = 'functor', 'flipped', 'do', 'starred', 'ifthen', 'ifnot', 'step', 'excepts'


# In[17]:


def into(callable):
    @singledispatch
    def dispatched(self): ...

    @dispatched.register(Sequence)
    def _(self):
        return dispatched(first(self)) if count(self) else """"""

    @dispatched.register(object)
    def _(self):
        try:
            if hasattr(self, 'function') :
                return dispatched(self.function)
            return callable(self)
        except:
            return """"""
        
    return dispatched


# In[18]:


@total_ordering
class State(object):
    """Base attributes for callables and fidgets."""
    
    def __init__(self, *args, **kwargs):
        for i, slot in enumerate(self.__slots__):
            setattr(self, slot, args[i])
        self.kwargs = kwargs
        
    def __getstate__(self):
        return tuple(map(partial(getattr, self), self.__slots__))
    
    def __setstate__(self, state):
        for key, value in zip(self.__slots__, state):
            setattr(self, key, value)
        
    def __copy__(self, *args):
        new = type(self)()
        return new.__setstate__(tuple(map(copy, self.__getstate__()))) or new

    def __hash__(self):
        values = []
        for slot in self.__slots__:
            values += [hashed(getattr(self, slot))]
        return hash(tuple(values))
    
    def __eq__(self, other):
        return isinstance(other, State) and hash(self) == hash(other)
    
    def __enter__(self):
        return copy(self[:])
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass
        
    def __lt__(self, other):
        if isinstance(other, State):
            return (len(self) < len(other)) and all(eq(*i) for i in zip(self, other))
        return False

    def __len__(self):
        return sum(1 for f in self)
    
    @property
    def __signature__(self):
        try:
            return signature(
                first(self.function) if isiterable(self.function) else 
                self.function) 
        except:
            return signature(self.__call__)

   
    __deepcopy__ = __copy__
    __signature__ = property(into(signature))
    __doc__ = property(into(getdoc))


# In[19]:


@singledispatch
def hashable(object):
    """Hash mappings and sequences"""
    return object

@hashable.register(Mapping)
def _(object):
    return type(object), tuple(
        (hashable(key), hashable(value)) for key, value in object.items())

hashable.register(Text, identity)

@hashable.register(Sequence)
def _(object):
    return type(object), tuple(hashable(value) for value in object)

def hashed(object): 
    return hash(hashable(object))


# In[20]:


class functor(State):
    """A function that evaluates a callable or returns the value of a non-callable."""
    __slots__ = ('function',)
    def __init__(self, function=identity):
        super(functor, self).__init__(function)
        
    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs) if callable(self.function) else self.function
    
    def __repr__(self):
        return repr(self.function)
    
    def __abs__(self):
        return self.__call__


# In[21]:


class flipped(functor):
    """Call a function with the arguments positional arguments reversed"""
    def __call__(self, *args, **kwargs):
        return super(flipped, self).__call__(*reversed(args), **kwargs)


# In[22]:


class do(functor):
    """Call a function and return input argument."""
    def __call__(self, *args, **kwargs):
        super(do, self).__call__(*args, **kwargs)
        return args[0] if args else None


# In[23]:


class starred(functor):
    """Call a function starring the arguments for sequences and starring the keywords for containers."""
    def __call__(self, *args, **kwargs):
        args = args[0] if len(args) is 1 else (args,)
        if not isiterable(args): 
            args = [(args,)]
        if isinstance(args, dict):
            args = kwargs.update(args) or tuple()
        return super(starred, self).__call__(*args, **kwargs)


# In[24]:


class Condition(functor):
    """Evaluate a function if a condition is true."""
    __slots__ = ('condition', 'function')
    def __init__(self, condition=bool, function=identity):
        super(functor, self).__init__(condition, function)


# In[25]:


class ifthen(Condition):
    def __call__(self, *args, **kwargs):
        return functor(self.condition)(*args, **kwargs) and super(ifthen, self).__call__(*args, **kwargs)

class ifnot(Condition):
    def __call__(self, *args, **kwargs):
        return functor(self.condition)(*args, **kwargs) or super(ifnot, self).__call__(*args, **kwargs)


# In[11]:


class step(Condition):
    def __call__(self, *args, **kwargs):
        result = functor(self.condition)(*args, **kwargs)
        return result and super(step, self).__call__(result)


# In[12]:


class excepts(functor):
    """Allow acception when calling a function"""
    __slots__ = ('exceptions', 'function')
    
    def __init__(self, exceptions=tuple(), function=identity):
        super(functor, self).__init__(copy(exceptions), function)

    def __call__(self, *args, **kwargs):
        try:
            return super(excepts, self).__call__(*args, **kwargs)
        except self.exceptions as e:
            return e


# In[15]:


if PY3:
    for func in __all__:
        setattr(locals()[func], '__doc__', property(into(getdoc)))

