
# coding: utf-8

# In[14]:

__all__ = ['_x', '_xx', 'call', 'defaults', 'ifthen', 'this', 'x_',]

from copy import copy
from functools import wraps
from importlib import import_module
from types import GeneratorType
from six import iteritems
from toolz.curried import isiterable, first, excepts, flip, last, identity, concatv, map, valfilter, keyfilter, merge, curry
from functools import partial
from operator import contains, methodcaller, itemgetter, attrgetter, not_, truth, abs, invert, neg, pos, index


# In[15]:

class StateMixin:
    """Mixin to reproduce state from __slots__
    """
    def __getstate__(self):
        return tuple(map(partial(getattr, self), self.__slots__))

    def __setstate__(self, state):
        [setattr(self, key, value) for key, value in zip(self.__slots__, state)]
        
    def __hash__(self):
        return hash(self._functions)
    
    def __len__(self):
        return len(self._functions)
    
    def __bool__(self):
        return True
    
    def __eq__(self, other):
        if isinstance(other, type(self)):
            return hash(self) == hash(other)
        return False

    def __ne__(self, other):
        return not(self == other)

    def __lt__(self, other):
        if isinstance(other, Functions):
            return self <= other and len(self) != len(other)
        return True

    def __le__(self, other):
        if isinstance(other, type(self)):
            return len(self) <= len(other) and self == copy(other)[:len(self)]
        return True

    def __gt__(self, other):
        return other < self

    def __ge__(self, other):
        return other <= self
    
    def __copy__(self):
        """Copy the composition.
        """
        copy = self.__class__()
        return copy.__setstate__(self.__getstate__()) or copy


# In[18]:

def functor(function):
    def caller(*args, **kwargs):
        if callable(function):
            return function(*args, **kwargs)
        return function
    try:
        return wraps(function)(caller)
    except (TypeError, AttributeError):
        return caller
        
class call(StateMixin, object):
    __slots__ = ('args', 'kwargs')
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
        
    def __call__(self, function=identity):
        return partial(functor(function), *self.args, **self.kwargs)   

def do(function):
    @wraps(function)
    def do(*args, **kwargs):
        function(*args, **kwargs)
        return len(args) and args[0] or tuple()
    return do

def flip(function):
    @wraps(function)
    def flip(*args, **kwargs):
        return function(*reversed(args), **kwargs)
    return flip

def stars(function):
    @wraps(function)
    def caller(*args, **kwargs):
        if len(args) > 1:
            args = (args,)
        else:
            if args and args[0]:
                if isinstance(args[0], dict):
                    args, kwargs = tuple(), merge(kwargs, args[0])
                elif isiterable(args[0]):
                    args = args[0]
                else:
                    args = (args,)
        return call()(function)(*args, **kwargs)
    return caller


def defaults(default):
    def caller(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            return call()(function)(*args, **kwargs) or functor(default)(*args, **kwargs)
        return wrapped
    return caller

def ifthen(condition):
    def caller(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            return call()(condition)(*args, **kwargs) and call()(function)(*args, **kwargs)
        return wrapped
    return caller


# In[19]:



class Functions(StateMixin, object):
    __slots__ = ('_functions', '_codomain')
    
    def __init__(self, functions=tuple()):        
        if not isiterable(functions) or isinstance(functions, (str, )):
            functions = (functions,)
        
        if isinstance(functions, dict):
            functions = iteritems(functions)
            
        self._functions = tuple(functions)

    def __getitem__(self, item):
        if isinstance(item, slice):
            if not item == slice(None):
                self._functions = self._functions[item]
        else:
            self._functions = tuple(concatv(self._functions, (item,)))
        return self 

                    
    def __iter__(self):
        """Generator yielding each evaluated function.
        """
        for function in self._functions:
            if isinstance(function, (dict, set, list, tuple)):
                function = Juxtapose(function)
            yield function

    def __repr__(self):
        return "{}{}".format(self.__class__.__name__, str(self._functions))
    
class Juxtapose(Functions):
    def __init__(self, functions=tuple()):
        self._codomain = isinstance(functions, Generator) and identity or type(functions)
        super(Juxtapose, self).__init__(functions)
        
    def __call__(self, *args, **kwargs):
        return self._codomain(
            call()(function)(*args, **kwargs)
            for function in self
        )
        
class Compose(Functions):
    def __init__(self, functions=tuple()):
        self._codomain = identity
        super(Compose, self).__init__(functions)
    
    def __call__(self, *args, **kwargs):
        for function in self:
            args, kwargs = (call()(function)(*args, **kwargs),), {}
        return self._codomain(args[0])
    
class Partials:
    _flip = False
    def _partial_(self, *args, **kwargs):
        """Update arguments and keywords of the current composition.
        """
        append = kwargs.pop('append', False)
        args and setattr(self, '_args', tuple(
            concatv(append and self._args or tuple(), args))) 
        kwargs and setattr(self, '_keywords', merge(
            append and self._keywords or {}, kwargs))
        return self
    
    def __eq__(self, other):
        if isinstance(other, Functions):
            return self._args == other._args
        return False

class Operations(Partials):
    def __hash__(self):
        return(self._functions, self._args, self._do, self._flip)
    
    def __eq__(self, other):
        if isinstance(other, Functions):
            return super(Operations, self).__eq__(other) and self._functions == other._functions
        return False

    def __ne__(self, other):
        return not(self == other)

    def __lt__(self, other):
        if isinstance(other, Functions):
            return super(Operations, self).__eq__(other) and self._functions < other._functions
        return True

    def __le__(self, other):
        if isinstance(other, Functions):
            return super(Operations, self).__eq__(other) and self._functions <= other._functions
        return True
    

class Callables(Operations, Functions):
    _factory_, _do = None, False
    __slots__ = ('_functions', '_args', '_keywords')

    def __init__(self, *args, **kwargs):
        self._functions = kwargs.pop('functions', False) or kwargs.pop('factory', Compose)()
        self._args, self._keywords = args, kwargs    
    
    def __getitem__(self, item=None):
        if self._factory_:
            self = self()
        if item is call or isinstance(item, call):
            return self(*(tuple() if item is call else item.args), **(dict() if item is call else item.kwargs))
        self._functions[item]        
        return self

    @property
    def __call__(self):
        """Call the composition with appending args and kwargs.
        """
        function = self._functions
        if self._do: function = do(function)
        if self._flip: function = flip(function)
        return call(*self._args, **self._keywords)(function)
    
    def __lshift__(self, item):
        if self._factory_:
            return type('_Do_', (Callables,), {'_do': True})()[item]
        return self[do(item)]
                
Callables.__rshift__ = Callables.__getitem__
Callables.__deepcopy__ = Callables.__copy__
Callables._ = Callables.__call__

class This(Callables):
    def __getattr__(self, attr):
        if any(attr.startswith(key) for key in ('_repr_', '_ipython_')):
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
    def _set_attribute_(method, _partial=True):
        @wraps(method)
        def setter(self, *args, **kwargs):    
            return (args or kwargs) and self[
                _partial and partial(method, *args, **kwargs) or method(*args, **kwargs)
            ] or self[method]
        return setter

    @staticmethod
    def _set_nested_(method):
        @wraps(method)
        def wrapped(self, item=None):    
            self = self[:]
            self._functions = Compose([method(item)(self._functions)])
            return self
        return wrapped
    
    
for attr, method in zip(
    ('__pow__', '__xor__', '__or__'), (ifthen, excepts, defaults)):
    setattr(Composition, attr, Composition._set_nested_(method))

Composition.partial = Partials._partial_
Composition.__and__ = Composition.__getitem__

Flipped = type('Flipped', (Composition,), {'_flip': True})
    
class Stars(Composition):
    @property
    def __call__(self):
        return stars(super(Stars, self).__call__)    

_y, _x, x_, _xx, this = tuple(
    type('_{}_'.format(f.__name__), (f,), {
        '_factory_': True, '_flip': False
    })(functions=Compose([f])) for f in (Callables, Composition, Flipped, Stars, This)
)

_handlers = {Composition: identity, Flipped: x_, Stars: _xx}

Composition.call = Composition.__call__
Composition.do = Composition.__lshift__
Composition.excepts = Composition.__or__
Composition.partial = Composition._partial_
Composition.pipe =  Composition.__getitem__

for imports in ('toolz', 'operator'):
    for name, function in iteritems(
        valfilter(callable, keyfilter(Compose([first, str.islower]), vars(import_module(imports))))
    ):
        if getattr(Composition, name, None) is None:
            opts = {}
            if function in (methodcaller, itemgetter, attrgetter, not_, truth, abs, invert, neg, pos, index):
                opts.update(_partial=False)
            elif function in (contains, flip) or imports == 'toolz': 
                pass
            else:
                function = flip(function)
            setattr(Composition, name, getattr(
                Composition, name, Composition._set_attribute_(function, **opts)
            ))

Composition.validate = Composition.__pow__
Composition.__matmul__ = Composition.groupby
Composition.__mul__ = Composition.map 
Composition.__truediv__  = Composition.filter

