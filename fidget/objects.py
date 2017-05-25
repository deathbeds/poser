# coding: utf-8

# In[1]:

try:
    from .state import State, Signature
    from .callables import functor

except:
    from state import State, Signature
    from callables import functor

from copy import copy
from toolz.curried import compose, first, isiterable, partial
from six import iteritems, PY3

# In[2]:


class Append(State):
    def __getitem__(self, object=slice(None)):
        return object != slice(None) and self.append(object) or self

    def __repr__(self):
        return repr(self.function)

    @property
    def append(self):
        return self.function.append


# In[3]:


class Functions(Signature):
    __slots__ = ('function', )

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


# In[4]:


def dispatch(object):
    if isinstance(object, (dict, set, list, tuple)):
        return Juxtapose(object, type(object))
    return functor(object)


# In[5]:


class Composition(Functions, Append):
    __slots__ = ('function', 'type')

    def __init__(self, object, type=None):
        if not isiterable(object) or isinstance(object, (str, )):
            object = [object]
        super(Composition, self).__init__(object, type)

    def __call__(self, *args, **kwargs):
        try:
            for i, object in enumerate(self):
                args, kwargs = (dispatch(object)(*args, **kwargs), ), {}
        except StopIteration:
            pass
        return first(args)


# In[6]:


class Juxtapose(Composition):
    def __init__(self, function, type=first):
        if isinstance(function, dict):
            function = compose(tuple, iteritems)(function)
        super(Juxtapose, self).__init__(function, type)

    def __call__(self, *args, **kwargs):
        return self.type(
            [dispatch(function)(*args, **kwargs) for function in self])


# In[21]:


class Compose(Composition):
    def __init__(self, function=None, type=functor):
        super(Compose, self).__init__(list()
                                      if function is None else function, type)

    def __call__(self, *args, **kwargs):
        return self.type(super(Compose, self).__call__)(*args, **kwargs)


# In[18]:


class Partial(Functions, Append):
    wrapper = staticmethod(functor)
    __slots__ = ('args', 'keywords', 'function')

    def __init__(self, *args, **kwargs):
        super(Partial, self).__init__(args, kwargs, Compose(type=self.wrapper))

    def __getitem__(self, object=slice(None), *args, **kwargs):
        if isinstance(object, slice):
            object, self = self.function.function[object], copy(self)
            self.function = Compose(object)
            return self

        return super(Partial, self).__getitem__(
            (args or kwargs) and partial(object, *args, **kwargs) or object)

    @property
    def append(self):
        return self.function.function.append

    @property
    def __call__(self):
        return partial(self.function, *self.args, **self.keywords)


# In[19]:


def doc(self):
    return getattr(first(self), '__doc__', '')


if PY3:
    for func in [Compose, Juxtapose]:
        setattr(func, '__doc__', property(doc))
