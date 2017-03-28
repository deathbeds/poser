
# coding: utf-8

# In[178]:

__all__ = ['_x', '_xx', 'call', 'defaults', 'ifthen', 'this', 'copy', 'x_',]

from copy import copy
from functools import wraps, total_ordering, partial
from importlib import import_module
from types import GeneratorType
from decorator import decorate
from six import iteritems
from toolz.curried import isiterable, first, excepts, flip, last, identity, concatv, map, valfilter, keyfilter, merge, curry, groupby, concat, get
from operator import contains, methodcaller, itemgetter, attrgetter, not_, truth, abs, invert, neg, pos, index

class State(object):
    def __getstate__(self):
        return tuple(map(partial(getattr, self), self.__slots__))

    def __setstate__(self, state):
        [setattr(self, key, value) for key, value in zip(self.__slots__, state)]
        
    def __copy__(self, *args):
        copy = self.__class__()
        return copy.__setstate__(self.__getstate__()) or copy
    
State.__deepcopy__ = State.__copy__

def _wrapper(function, caller, *args):
    for wrap in concatv((wraps,), args):
        try: return wrap(function)(caller)
        except: pass
    return caller

def functor(function):
    def caller(*args, **kwargs):
        if callable(function):
            return function(*args, **kwargs)
        return function
    return callable(function) and _wrapper(function, caller) or caller
        
class call(State):
    __slots__ = ('args', 'kwargs')
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
        
    def __call__(self, function=identity):
        def caller(*args, **kwargs):
            return functor(function)(*concatv(self.args, args), **merge(self.kwargs, kwargs))
        return callable(function) and _wrapper(function,caller) or caller

def do(function):
    def caller(*args, **kwargs):
        function(*args, **kwargs)
        return args[0] if len(args) else tuple()
    return _wrapper(function, caller, curry(decorate))

def flipped(function):
    def caller(*args, **kwargs):
        return call(*reversed(args), **kwargs)(function)()
    return _wrapper(function, caller, curry(decorate))

def stars(function):
    def caller(*args, **kwargs):
        if all(map(isiterable, args)):
            combined = groupby(flip(isinstance)(dict), args)
            args = concat(get(False, combined, tuple()))
            kwargs = merge(kwargs, *get(True, combined, {}))
            return call(*args)(function)(**kwargs)
        return call(args)(function)(**kwargs)
    return _wrapper(function, caller, curry(decorate))

def defaults(default):
    def caller(function):
        def defaults(*args, **kwargs):
            return call(*args, **kwargs)(function)() or call(*args, **kwargs)(default)()
        return _wrapper(function, defaults, curry(decorate))
    return caller

def ifthen(condition):
    def caller(function):
        def ifthen(*args, **kwargs):
            return call(*args, **kwargs)(condition)() and call(*args, **kwargs)(function)()
        return _wrapper(function, ifthen, curry(decorate))
    return caller

@total_ordering
class Logic:
    def __eq__(self, other):
        if isinstance(other, Functions):
            return hash(self) == hash(other)
        return False

    def __le__(self, other):
        if isinstance(other, Functions):
            return self == copy(other)[:len(self)]
        return False

class Functions(State, Logic):
    __slots__ = ('_functions', '_codomain')
    
    def __init__(self, functions=tuple()):        
        if not isiterable(functions) or isinstance(functions, (str, )):
            functions = (functions,)
            
        self._functions = tuple((isinstance(functions, dict) and iteritems or identity)(functions))

    def __getitem__(self, item):
        if isinstance(item, slice): 
            if item == slice(None): pass
            else: self._functions = self._functions[item]
        else:
            self._functions = tuple(concatv(self._functions, (item,)))
        return self 

    def __iter__(self):
        for function in self._functions:
            yield (isinstance(function, (dict, set, list, tuple)) and Juxtapose or identity)(function)

    def __len__(self):
        return len(self._functions)

    def __hash__(self):
        return hash(self._functions)

    def __repr__(self):
        return "{}{}".format(self.__class__.__name__, str(self._functions))
    
class Juxtapose(Functions):
    def __init__(self, functions=tuple()):
        self._codomain = isinstance(functions, GeneratorType) and identity or type(functions)
        super(Juxtapose, self).__init__(functions)
        
    def __call__(self, *args, **kwargs):
        _call = call(*args, **kwargs)
        return self._codomain(_call(function)() for function in self)
        
class Compose(Functions):
    def __init__(self, functions=tuple()):
        self._codomain = identity
        super(Compose, self).__init__(functions)
    
    def __call__(self, *args, **kwargs):
        for function in self:
            args, kwargs = (call()(function)(*args, **kwargs),), {}
        return self._codomain(args[0])
    
class Partials(Logic):    
    def __eq__(self, other):
        if isinstance(other, Functions):
            return self._args == other._args and super(Partials, self).__eq__(other)
        return False
    
    def __le__(self, other):
        if isinstance(other, Functions):
            return self._args == other._args and super(Partials, self).__le__(other)
        return False

class Callables(Partials, Functions):
    _factory_, _do, _func_ = None, False, staticmethod(identity)
    __slots__ = ('_functions', '_args', '_keywords')

    def __init__(self, *args, **kwargs):
        self._functions = kwargs.pop('functions', Compose()) 
        self._args, self._keywords = args, kwargs    
    
    def __getitem__(self, item=None):
        if self._factory_:
            self = self()
        if item is call: 
            item = call()
        if isinstance(item, call):
            return item(self)()
        self._functions[item]
        return self

    def __func__(self):
        if self._do:
            return do(self._functions)
        return (self._factory_ and identity or self._func_)(self._functions)

    def __hash__(self):
        return hash((self._functions, self._args, self._do))

    @property
    def __call__(self):
        return call(*self._args, **self._keywords)(self.__func__())

    def __lshift__(self, item):
        if self._factory_:
            return type('_Do_', (Callables,), {'_do': True})()[item]
        return self[do(item)]
                
class This(Callables):
    def __getattr__(self, attr):
        if any(attr.startswith(key) for key in ('__' , '_repr_', '_ipython_')):
            return self
        return super(This, self).__getitem__(callable(attr) and attr or attrgetter(attr))
    
    def __getitem__(self, item):
        return super(This, self).__getitem__(callable(item) and item or itemgetter(item))
    
    def __call__(self, *args, **kwargs):
        previous = last(self._functions._functions)
        if type(previous) == attrgetter:
            attrs = previous.__reduce__()[-1]
            if len(attrs) == 1:
                self._functions = self._functions[:-1]
            return self[methodcaller(attrs[0], *args, **kwargs)]  
        return super(This, self).__call__(*args, **kwargs)


class Composition(Callables): 
    @staticmethod
    def _add_attribute_(method, _partial=True):
        def caller(self, *args, **kwargs):    
            return (args or kwargs) and self[
                _partial and partial(method, *args, **kwargs) or method(*args, **kwargs)
            ] or self[method]
        return wraps(method)(caller)

    @staticmethod
    def _add_nesting_attribute_(method):
        def caller(self, item=None):    
            self = self[:]
            self._functions = Compose([method(item)(self._functions)])
            return self
        return wraps(method)(caller)
    
    def __getitem__(self, item=None, *args, **kwargs):
        return super(Composition, self).__getitem__((args or kwargs) and partial(item, *args, **kwargs)  or item)   
    
class Flipped(Composition):
    _func_ = staticmethod(flipped)
    
class Stars(Composition):
    _func_ = staticmethod(stars)

_y, _x, this, x_, _xx,  = tuple( 
    type('_{}_'.format(f.__name__), (f,), {
        '_factory_': True,
    })(functions=Compose([f])) for f in (Callables, Composition, This, Flipped, Stars))

for imports in ('toolz', 'operator'):
    for attr, method in iteritems(
        valfilter(callable, keyfilter(Compose([first, str.islower]), vars(import_module(imports))))
    ):
        opts = {}
        if getattr(Composition, attr, None) is None:
            if method in (methodcaller, itemgetter, attrgetter, not_, truth, abs, invert, neg, pos, index):
                opts.update(_partial=False)
            elif method in (contains, flip) or imports == 'toolz': 
                pass
            else: 
                method = flip(method)
            setattr(Composition, attr, getattr(
                Composition, attr, Composition._add_attribute_(method, **opts)
            ))
            
Callables.__rshift__ = Callables.__getitem__
            
for attr, method in zip(('__pow__', '__xor__', '__or__'), (ifthen, excepts, defaults)):
    setattr(Composition, attr, Composition._add_nesting_attribute_(method))
            
for attr, method in (
    ('__and__', '__getitem__'), ('call', '__call__'), ('do', '__lshift__'), ('excepts', '__or__'), 
    ('pipe',  '__getitem__'), ('validate', '__pow__'), ('__matmul__', 'groupby'),
    ('__mul__', 'map'), ('__truediv__ ', 'filter'), 
):
    setattr(Composition, attr, getattr(Composition, method))
    
del imports, attr, method, opts


# In[154]:

# ip = get_ipython()

# _x(10)[range]

# _x(10)[range].first()

# _x[range].map(_x['sadfasdf'][ip.log.warning])>>list>>call(100);

# _xx([10, 20])>>range>>call([2])

# _xx(10,20,30)>>call

# _xx([10, 20])>>call

# (_x<<range>>print)>>call(10)

# _x()

# x_(20, 10)>>range>>OrderedDict((('b', len), ('a', type)))>>call

# x_(20, 10)>>range>>list((len, type))>>call

# _x>>range>>call(10)

# (_x >> identity << (_x >> range >> print) ) >> call(10)

# _xx(10, 20)>>{len, type}>>call

# _x(2)[_x(3, 10)[range]]>>list>>call

# _x[_x[_x[_x[str.upper]]]] >> call('asdf')

# _xx([10, 20],)>>range>>call

# from toolz.curried import *

# import requests
# GET = memoize(requests.get)

# (
#     (_x >> range) ** (lambda x: isinstance(x, int)) | _x & str.upper & (lambda x: x*10)
# ) >> call('asdf')


# In[ ]:



