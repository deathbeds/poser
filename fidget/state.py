# coding: utf-8

from copy import copy
from functools import partial, total_ordering
from operator import eq


def hashdict(attr):
    value = []
    for item in attr.items():
        value += [[]]
        for part in item:
            if isinstance(part, dict):
                part = hashdict(part)

            value[-1] += [part]
        value[-1] = tuple(value[-1])
    return tuple(value)


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
        for attr in self.__slots__:
            if isinstance(attr, dict):
                attr = hashdict(attr)
            values += [hash(attr)]
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
