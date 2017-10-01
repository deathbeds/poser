
# coding: utf-8

# # `callables`
# 
# Collections of special callable objects.

# In[10]:


try:
    from .functions import State, functor, into, getdoc
except:
    from functions import State, functor, into, getdoc
    
from functools import singledispatch
from toolz.curried import compose, first, isiterable, partial, identity, count
from copy import copy
from six import iteritems, PY3
from types import LambdaType
from typing import Mapping, Text, Sequence
from inspect import getsource

__all__ = 'Compose', 'Juxtapose'


# In[11]:


class Functions(State):  
    """Base class for chainable functions."""
    __slots__ = ('function',)
    def __getitem__(self, object=slice(None)):
        if object is getdoc:
            return self[:]
        if isinstance(object, int): 
            return self.function[object]
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


# In[12]:


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


# In[4]:


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
                    args, kwargs = (calls(object)(*args, **kwargs),), {}
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


# In[5]:


class Juxtapose(Callable):
    """Composed many functions for the sample inputs."""
    def __init__(self, function=None, type=tuple):
        if isinstance(function, dict):
            function = compose(tuple, iteritems)(function)
        super(Juxtapose, self).__init__(tuple(), dict(), function or list(), type)
        
    def __call__(self, *args, **kwargs):
        return self.type([calls(function)(*args, **kwargs) for function in self]) 


# In[6]:


@singledispatch
def calls(object): return None

calls.register(Text, functor)

@calls.register(Mapping)
@calls.register(Sequence)
def _(object): return Juxtapose(object, type(object))

calls.register(object, functor)


# In[7]:


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


# In[8]:


if PY3:
    for func in __all__:
        setattr(locals()[func], '__doc__', property(into(getdoc)))

