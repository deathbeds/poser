
# coding: utf-8

# # `callables`
# 
# Collections of special callable objects.

# In[52]:


from functools import total_ordering
from toolz.curried import compose, first, isiterable, partial, identity
from six import iteritems, PY3
from decorator import decorator, decorate
from types import LambdaType
from inspect import getsource, signature
from operator import eq

__all__ = 'functor', 'flipped', 'do', 'starred', 'ifthen', 'ifnot', 'step', 'excepts', 'Compose', 'Juxtapose'


# In[ ]:


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
            values += [hashiter(getattr(self, slot))]
        return hash(tuple(values))
    
    def __eq__(self, other):
        return isinstance(other, State) and hash(self) == hash(other)
    
    def __enter__(self):
        return copy(self[:])
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass
    
    def __abs__(self):
        return self.__call__
    
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


# In[53]:


def hashiter(object):
    """Hash an interable container."""
    if isiterable(object):
        if isinstance(object, dict):
            values = []
            for key, value in object.items():
                values +=[(key, hashiter(value))]
        else:
            values = []
            for value in object:
                values += [hashiter(value)]
        object = (type(object), tuple(values))
    return hash(object)


# In[54]:


class functor(State):
    """A function that evaluates a callable or returns the value of a non-callable."""
    __slots__ = ('function',)
    def __init__(self, function=identity):
        super(functor, self).__init__(function)
        
    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs) if callable(self.function) else self.function
    
    def __repr__(self):
        return repr(self.function)


# In[55]:


class flipped(functor):
    """Call a function with the arguments positional arguments reversed"""
    def __call__(self, *args, **kwargs):
        return super(flipped, self).__call__(*reversed(args), **kwargs)


# In[56]:


class do(functor):
    """Call a function and return input argument."""
    def __call__(self, *args, **kwargs):
        super(do, self).__call__(*args, **kwargs)
        return args[0] if args else None


# In[57]:


class starred(functor):
    """Call a function starring the arguments for sequences and starring the keywords for containers."""
    def __call__(self, *args, **kwargs):
        args = args[0] if len(args) is 1 else (args,)
        if not isiterable(args): 
            args = [(args,)]
        if isinstance(args, dict):
            args = kwargs.update(args) or tuple()
        return super(starred, self).__call__(*args, **kwargs)


# In[63]:


class Condition(functor):
    """Evaluate a function if a condition is true."""
    __slots__ = ('condition', 'function')
    def __init__(self, condition=bool, function=identity):
        super(functor, self).__init__(condition, function)


# In[64]:


class ifthen(Condition):
    def __call__(self, *args, **kwargs):
        return functor(self.condition)(*args, **kwargs) and super(ifthen, self).__call__(*args, **kwargs)

class ifnot(Condition):
    def __call__(self, *args, **kwargs):
        return functor(self.condition)(*args, **kwargs) or super(ifnot, self).__call__(*args, **kwargs)


# In[67]:


class step(Condition):
    def __call__(self, *args, **kwargs):
        result = functor(self.condition)(*args, **kwargs)
        return result and super(step, self).__call__(result)


# In[69]:


class excepts(functor):
    """Allow acception when calling a function"""
    __slots__ = ('exceptions', 'function')
    
    def __init__(self, exceptions=tuple(), function=identity):
        super(functor, self).__init__(copy(exceptions), function)

    def __call__(self, *args, **kwargs):
        try:
            return super(excepts, self).__call__(*args, **kwargs)
        except self.exceptions as e:
            return exception(e)


# In[70]:


class Functions(State):  
    """Base class for chainable functions."""
    __slots__ = ('function',)
    def __getitem__(self, object=slice(None)):   
        return object != slice(None) and self.append(object) or self 
    
    def __repr__(self):
        return repr(self.function)
    
    @property
    def append(self):
        return self.function.append
    
    __name__ = __repr__
    
    def __contains__(self, object):
        return any(object == function for function in self)
    
    def __delitem__(self, object):
        self.function = list(fn for fn in self if fn != object)
        return self
    
    def __setitem__(self, attr, object):
        self.function = list(object if fn == attr else fn for fn in self)
        return self  
    
    def __iter__(self):
        for function in self.function:
            yield function

    def __reversed__(self):
        self.function = type(self.function)(reversed(self.function))
        return self


# In[71]:


def _dispatch(object):
    """Evaluate sequences and containers as functions"""
    if isinstance(object, (dict, set, list, tuple)):
        return Juxtapose(object, type(object))
    return functor(object)


# In[72]:


class Callable(Functions):
    """Base class for chainable functions through the getitem api."""
    __slots__ = ('args', 'keywords', 'function', 'type')
            
    def __getitem__(self, object=slice(None), *args, **kwargs):
        if isinstance(object, slice):
            object, self = Compose(self.function[object]), copy(self)
            self.function = object
            return self
               
        return super(Callable, self).__getitem__(
            (args or kwargs) and partial(object, *args, **kwargs) or object)
    
    @property
    def append(self): return self.function.append
    
    @property
    def __call__(self):
        return partial(self.function, *self.args, **self.keywords)


# In[73]:


class Compose(Callable):
    """Function compose in serial."""
    
    def __init__(self, object=None, type=functor):
        if object is None:
            object = list()
        if not isiterable(object) or isinstance(object, (str,)):
            object = [object]
        
        super(Compose, self).__init__(tuple(), dict(), object, type)
        
        
    def __call__(self, *args, **kwargs):
        try:
            for i, object in enumerate(self):
                try:
                    args, kwargs = (_dispatch(object)(*args, **kwargs),), {}
                except Exception as e:
                    # Could analyze current state.
                    raise Exception('on {}th callable `{}` in {}'.format(
                        i+1, 
                        (isinstance(object, LambdaType) and compose(first, str.splitlines, getsource) or identity)(object) 
                        , self
                    )) from e
        except StopIteration:
            pass
        return first(args)


# In[74]:


class Juxtapose(Callable):
    """Composed many functions for the sample inputs."""
    def __init__(self, function=None, type=tuple):
        if isinstance(function, dict):
            function = compose(tuple, iteritems)(function)
        super(Juxtapose, self).__init__(tuple(), dict(), function or list(), type)
        
    def __call__(self, *args, **kwargs):
        return self.type([_dispatch(function)(*args, **kwargs) for function in self]) 


# In[83]:


class Partial(Callable):
    """Compose functions with parital arguments and keywords."""
    
    _composition, _wrapper = map(staticmethod, (Compose, functor))
    
    def __init__(self, *args, **kwargs):
        super(Partial, self).__init__(args, kwargs, self._composition(), self._wrapper)
        
    def __getitem__(self, object=slice(None), *args, **kwargs):
        if isinstance(object, slice):
            object, self = Compose(self.function.function[object]), copy(self)
            self.function = object
            return self
               
        return super(Partial, self).__getitem__(
            (args or kwargs) and partial(object, *args, **kwargs) or object)
    
    @property
    def append(self):
        return self.function.function.append
    
    @property
    def __call__(self):
        return partial(
            self._wrapper(self.function), *self.args, **self.keywords)


# In[76]:


if PY3:
    def doc(self):
        return getattr(first(self), '__doc__', '')
    
    for func in __all__:
        setattr(locals()[func], '__doc__', property(doc))

