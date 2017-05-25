# coding: utf-8

# In[1]:

try:
    from .state import State, Signature
except:
    from state import State, Signature
from copy import copy
from six import PY3
from toolz import isiterable, identity

__all__ = [
    'functor', 'flipped', 'do', 'starred', 'ifthen', 'ifnot', 'step', 'excepts'
]

# In[3]:


class functor(State, Signature):
    __slots__ = ('function', )

    def __init__(self, function=identity):
        super(functor, self).__init__(function)

    def __call__(self, *args, **kwargs):
        return self.function(
            *args, **kwargs) if callable(self.function) else self.function

    def __repr__(self):
        return repr(self.function)


class flipped(functor):
    def __call__(self, *args, **kwargs):
        return super(flipped, self).__call__(*reversed(args), **kwargs)


class do(functor):
    def __call__(self, *args, **kwargs):
        super(do, self).__call__(*args, **kwargs)
        return args[0] if args else None


# In[4]:


class starred(functor):
    def __call__(self, *args, **kwargs):
        args = args[0] if len(args) is 1 else (args, )
        if not isiterable(args):
            args = [(args, )]
        if isinstance(args, dict):
            args = kwargs.update(args) or tuple()
        return super(starred, self).__call__(*args, **kwargs)


# In[43]:


class condition(functor):
    __slots__ = ('condition', 'function')

    def __init__(self, condition=bool, function=identity):
        super(functor, self).__init__(condition, function)


class ifthen(condition):
    def __call__(self, *args, **kwargs):
        return functor(self.condition)(*args, **kwargs) and super(
            ifthen, self).__call__(*args, **kwargs)


class ifnot(condition):
    def __call__(self, *args, **kwargs):
        return functor(self.condition)(*args, **kwargs) or super(
            ifnot, self).__call__(*args, **kwargs)


class step(condition):
    def __call__(self, *args, **kwargs):
        result = functor(self.condition)(*args, **kwargs)
        return result and super(step, self).__call__(result)


# In[44]:


class excepts(functor):
    __slots__ = ('exceptions', 'function')

    def __init__(self, exceptions=tuple(), function=identity):
        super(functor, self).__init__(copy(exceptions), function)

    def __call__(self, *args, **kwargs):
        try:
            return super(excepts, self).__call__(*args, **kwargs)
        except self.exceptions as e:
            return exception(e)


# In[45]:


class exception(State):
    __slots__ = ('exception', )

    def __init__(self, exception=tuple()):
        super(functor, self).__init__(copy(exception))

    def __bool__(self):
        return not self.exception

    def __repr__(self):
        return repr(self.exception)


# In[46]:


def doc(self):
    return getattr(self.function, '__doc__', '')


if PY3:
    for func in map(locals().__getitem__, __all__):
        setattr(func, '__doc__', property(doc))
