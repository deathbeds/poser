
# coding: utf-8

# In[79]:

__all__ = ['_x', '_xx', 'call', 'defaults', 'ifthen', 'copy', 'x_', '_y']

from copy import copy
from functools import wraps, total_ordering, partial
from importlib import import_module
from decorator import decorate
from types import GeneratorType
from six import iteritems, PY34
from toolz.curried import isiterable, first, excepts, flip, last, complement, identity, concatv, map, valfilter, keyfilter, merge, curry, groupby, concat, get
from operator import contains, methodcaller, itemgetter, attrgetter, not_, truth, abs, invert, neg, pos, index

class State(object):
    def __getstate__(self):
        return tuple(map(partial(getattr, self), self.__slots__))

    def __setstate__(self, state):
        for key, value in zip(self.__slots__, state):
            setattr(self, key, value)
        
    def __copy__(self, *args):
        new = self.__class__()
        return new.__setstate__(tuple(map(copy, self.__getstate__()))) or new
    
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
class Functions(State):
    __slots__ = ('_functions',)
    def __init__(self, functions=tuple()):        
        if not isiterable(functions) or isinstance(functions, (str, )):
            functions = (functions,)
            
        self._functions = (isinstance(functions, dict) and iteritems or identity)(functions)

    def __getitem__(self, item):
        if isinstance(item, slice): 
            if item == slice(None): pass
            else: self._functions = self._functions[item]
        else:
            self._functions = tuple(concatv(self._functions, (item,)))
        return self 

    def __iter__(self):
        for function in self._functions:
            yield (
                isinstance(function, (dict, set, list, tuple)) and call(codomain=type(function))(Juxtapose) or identity
            )(function)

    def __len__(self):
        return len(self._functions)

    def __hash__(self):
        return hash(self._functions)

    def __eq__(self, other):
        if isinstance(other, Functions):
            return hash(self) == hash(other)
        return False

    def __lt__(self, other):
        if isinstance(other, Functions):
            return (len(self) < len(other)) and all(eq(*i) for i in zip(self, copy(other)[:len(self)-1]))
        return False

    def __reversed__(self):
        self._functions = tuple(reversed(self._functions))
        return self
    
    def __repr__(self):
        return "{}{}".format(self.__class__.__name__, str(self._functions))
    
class Juxtapose(Functions):
    __slots__ = ('_functions', '_codomain')
    def __init__(self, functions=tuple(), codomain=identity):
        self._codomain = codomain
        super(Juxtapose, self).__init__(functions)
        
    def __call__(self, *args, **kwargs):
        return self._codomain(call(*args)(function)(**kwargs) for function in self)
        
class Compose(Functions):
    def __call__(self, *args, **kwargs):
        for function in self:
            args, kwargs = (call()(function)(*args, **kwargs),), {}
        return args[0]
    
    
class Callables(Functions):
    _functions_default_ = Compose
    _factory_, _do, _func_ = None, False, staticmethod(identity)
    __slots__ = ('_functions', '_args', '_keywords')

    def __init__(self, *args, **kwargs):
        self._functions = kwargs.pop('functions', self._functions_default_()) 
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
        if self._do: return do(self._functions)
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
    
    def __invert__(self):
        self = self[:]
        self._functions = reversed(self._functions)
        return self
                            
class Juxtaposition(Callables): 
    _functions_default_ = Juxtapose
    
class Composition(Callables): 
    @staticmethod
    def _add_attribute_(method, _partial=True):
        def caller(self, *args, **kwargs):    
            return (args or kwargs) and self[
                _partial and partial(method, *args, **kwargs) or method(*args, **kwargs)
            ] or self[method]
        return wraps(method)(caller)

    def __getitem__(self, item=None, *args, **kwargs):
        return super(Composition, self).__getitem__(
            (args or kwargs) and call(*args, **kwargs)(item)  or item)   
    
    def __pow__(self, item, method=ifthen):
        self = self[:]
        if isinstance(item, type):
            if issubclass(item, Exception) or isiterable(item) and all(map(flip(isinstance)(Exception), item)):
                method = excepts
            elif isiterable(item) and all(map(flip(isinstance)(type), item)):
                item = flip(isinstance)(item)
        self._functions = Compose([method(item)(self._functions)])
        return self

    def __or__(self, item):
        self = self[:]
        self._functions = Compose([defaults(item)(self._functions)])
        return self
    
    def __pos__(self):
        return self[bool]

    def __neg__(self):
        return self[complement(bool)]

for attr in ('and', 'add', 'rshift', 'sub'): 
    setattr(Callables, "__{}__".format(attr), getattr(Callables, '__getitem__'))

class Flipped(Composition):
    _func_ = staticmethod(flipped)
    
class Starred(Composition):
    _func_ = staticmethod(stars)

_y, _x, x_, _xx = tuple( 
    type('_{}_'.format(f.__name__), (f,), {'_factory_': True,})(functions=Compose([f])) 
    for f in (Juxtaposition, Composition, Flipped, Starred))

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
            
for attr, method in (
    ('call', '__call__'), ('do', '__lshift__'), ('pipe',  '__getitem__'), ('__xor__',  '__pow__'),
    ('__matmul__', 'groupby'), ('__mul__', 'map'), ('__truediv__ ', 'filter'), 
): setattr(Composition, attr, getattr(Composition, method))

if PY34:
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
        
    this = type('_This_', (This,), {'_factory_': True})
    
    __all__ += ['this']
    
del imports, attr, method, opts, PY34


# In[12]:

# from collections import OrderedDict

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



