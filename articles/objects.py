
# coding: utf-8

# Special composition objects

try:
    from .composites import composite, a, compose
    from .partials import partial_attribute
except BaseException:
    from composites import composite, a, compose
    from partials import partial_attribute
from functools import partialmethod
__all__ = tuple()

from collections import UserDict, UserList, OrderedDict
from toolz import isiterable
dunder = '__{}__'.format
__all__ = 'cache', 'persist', 'parallel', 'dispatch', 'enumerated', 'store'


class enumerated(composite):
    """Return the a list of the execution results. enumerated escapes the
    computer returning the error and previous callables.

    >>> f = enumerated()[range].do(len)[type]
    >>> dict(zip(f.data, f(10)))
    {do:[<built-in function len>]: range(0, 10), <class 'type'>: <class 'range'>, <class 'range'>: range(0, 10)}

    This composition is very useful for development. The native call output is not
    the same as the absolute value of the evalution.

    >>> assert f(10) != abs(f)(10) == f(10)[-1]
    """

    def __call__(self, *args, **kwargs):
        output = list()
        result = compose.__call__(self, *args, **kwargs)
        while True:
            try:
                output.append(next(result))
            except Exception as e:
                not isinstance(e, StopIteration) and output.append(e)
                break
        return output

    __abs__ = compose.__abs__


class parallel(composite):
    """Composites with trivially parallel map operations. Each map is a joblib.delayed
    object.

    parallel requires joblib.

    >>> assert parallel(jobs=4)[range].map(range)(8) == list(map(range, range(8)))
    """

    def __init__(self, jobs=4, data=None):
        setattr(self, 'jobs', jobs) or super().__init__(data)

    def map(self, function):
        """A delay each function."""
        return super().__getattr__('map')(__import__('joblib').delayed(function))

    def __call__(self, *args, **kwargs):
        return __import__('joblib').Parallel(
            self.jobs)(
            super().__call__(
                *args, **kwargs))

    __truediv__ = map


class dispatch(OrderedDict):
    """Single dispatch callable dictionary.

    >>> f = dispatch((str, str.upper), (int, range), (object, type))
    >>> f('text'), f(42), f({10})
    ('TEXT', range(0, 42), <class 'set'>)
    """

    def __init__(self, *data):
        super().__init__(
            data[0] if len(data) is 1 and isinstance(
                data[0], (list, dict)) else data)

    def __call__(self, *args, **kwargs):
        for condition, object in self.items():
            if callable(condition) and not isinstance(condition, type):
                if condition(*args, **kwargs):
                    break
            else:
                if isinstance(args[0], condition):
                    break
        else:
            raise TypeError("No conditions matched for {}.".format(args))

        return object(*args, **kwargs)


class store(UserDict):
    """A mutable callable object that stores function calls on itself.

    >>> c = store(range)
    >>> assert c(10)(20)(30) and 10 in c and 20 in c and 30 in c
    >>> assert c[10] == range(10)
    >>> c.callable = type
    >>> assert c(10)[10] == int
    """

    def __init__(self, callable, data=None):
        self.callable = callable or composition()
        super().__init__(data)

    def __missing__(self, item):
        self(item) if not isinstance(item, tuple) else self(*item)
        return self[item]

    def __call__(self, *args, **kwargs):
        self[args[0] if args else None] = self.callable(*args, **kwargs)
        return self

    def call(self, *args, **kwargs):
        self(*args, **kwargs)
        return self.get(args[0] if args else None)

    def __abs__(self): return self.call


class cache(store):
    """A immutable callable object that caches function calls on itself.

    >>> c = cache(range)
    >>> assert c(10)[10] == range(10)
    >>> c.callable = type
    >>> assert c(10)[10] == range(10)
    """

    def __call__(self, *args, **kwargs):
        if args[0] not in self:
            super().__call__(*args, **kwargs)
        return self


class persist(__import__('shelve').DbfilenameShelf):
    """A callable object that stores function calls on disk and/or memory.

    >>> c = cache(range)
    >>> assert c(10)(20)(30) and 10 in c and 20 in c and 30 in c
    >>> p = persist('test', 'n')
    >>> assert not list(p.keys()) and p(40) and 40 in p.keys() and 30 not in p.keys()
    >>> p.update(c)
    >>> assert 30 in p
    >>> p.close()
    >>> p2 = persist('test', 'r')
    >>> assert 10 in p2 and 20 in p2 and 30 in p2 and 40 in p2
    """

    def __init__(self, callable, *args, **kwargs):
        if isinstance(callable, str):
            args = callable, *args
            callable = composite()
        super(persist, self).__init__(*args, **kwargs)
        self.callable = callable

    def __method__(self, method, item, *args):
        return getattr(super(), dunder(method))(str(item), *args)

    __getitem__ = partialmethod(__method__, 'getitem')
    __setitem__ = partialmethod(__method__, 'setitem')
    __contains__ = partialmethod(__method__, 'contains')

    __call__ = store.__call__

    __abs__, call = store.__abs__, store.call


if __name__ == '__main__':
    print(__import__('doctest').testmod(verbose=False))
    get_ipython().system(
        'jupyter nbconvert --to python --TemplateExporter.exclude_input_prompt=True objects.ipynb')
