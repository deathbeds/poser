
# coding: utf-8

# Special composition objects

try:
    from .composites import composite
except:
    from composites import composite
from functools import partialmethod
__all__ = tuple()

from collections import UserDict
dunder = '__{}__'.format
__all__ = 'cache', 'persist'


class parallel(composite):
    """An embarassingly parallel composite
    
    All map functions are delayed
    >>> parallel(jobs=4)[range].map(print) # doctest: +SKIP
    """
    def __init__(self, jobs=4, data=None):
        setattr(self, 'jobs', jobs) or super().__init__(data)
        
    def map(self, function):
        """A delay each function."""
        return super().__getattr__('map')(__import__('joblib').delayed(function))
    
    def __call__(self, *args, **kwargs):
        return __import__('joblib').Parallel(self.jobs)(super().__call__(*args, **kwargs))
        
    __truediv__ = map


class dispatch(composite):
    """a singledispatching composite

    >>> f = dispatch((str, str.upper), (int, range), (object, type))
    >>> f('text'), f(42), f({10})
    ('TEXT', range(0, 42), <class 'set'>)
    """
    def __init__(self, *data):
        self.dispatch = None
        super().__init__(isinstance(data[0], dict) and list(data.items()) or data)

    def __call__(self, arg):
        if not self.dispatch:
            self.dispatch = __import__('functools').singledispatch(λ[:])
            for cls, func in self.data: self.dispatch.register(cls, func)
        return self.dispatch.dispatch(type(arg))(arg)


class cache(UserDict):
    """cache on the first argument, use star to store tuples.
    
    >>> c = cache(range)
    >>> assert c(10)(20)(30) and 10 in c and 20 in c and 30 in c
    """
    def __init__(self, callable, data=None):
        self.callable = callable or composition()
        super().__init__(data)
        
    def __missing__(self, item):
        self(item) if not isinstance(item, tuple) else self(*item)
        return self[item]
        
    def __call__(self, *args, **kwargs): 
        if args[0] not in self:  self[args[0]] = self.callable(*args, **kwargs)
        return self


class persist(__import__('shelve').DbfilenameShelf):
    """
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
        if isinstance(callable, str): args = callable, *args
        super(persist, self).__init__(*args, **kwargs)
        if isinstance(callable, str): callable = self.get(callable, composition())
        self.callable = callable
        
    def close(self):
        try:
            self[callable] = self.callable
            while hasattr(self[callable], 'callable'):  self[callable] = self[callable].callable
        except:
            pass
        return super().close()
    
    def __method__(self, method, item, *args):
        return getattr(super(), dunder(method))(str(item), *args)
    
    __getitem__ = partialmethod(__method__, 'getitem')
    __setitem__ = partialmethod(__method__, 'setitem')
    __contains__ = partialmethod(__method__, 'contains')
    
    def __call__(self, arg, **kwargs): 
        if arg not in self:
            self[arg] = self.callable(arg, **kwargs)
        return self


if __name__ == '__main__':
    print(__import__('doctest').testmod(verbose=False))
    get_ipython().system('jupyter nbconvert --to python --TemplateExporter.exclude_input_prompt=True objects.ipynb')

