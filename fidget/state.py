# coding: utf-8

# In[12]:

from copy import copy
from functools import partial, total_ordering
from operator import eq
from toolz import isiterable

# In[33]:


def hashiter(object):
    if isiterable(object):
        if isinstance(object, dict):
            values = []
            for key, value in object.items():
                values += [(key, hashiter(value))]
        else:
            values = []
            for value in object:
                values += [hashiter(value)]
        object = (type(object), tuple(values))
    return hash(object)


# In[21]:


@total_ordering
class State(object):
    def __init__(self, *args):
        for i, slot in enumerate(self.__slots__):
            setattr(self, slot, args[i])

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
            return (len(self) < len(other)) and all(
                eq(*i) for i in zip(self, other))
        return False

    def __len__(self):
        return sum(1 for f in self)

    __deepcopy__ = __copy__
