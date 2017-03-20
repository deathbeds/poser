
# coding: utf-8

# # Higher-order functions for interactive computing

# In[713]:

__all__ = ['_x', '_xx', 'call', 'stars', 'this', 'x_',]

from copy import copy
from functools import wraps
from importlib import import_module
from collections import Generator
from six import iteritems
from toolz.curried import identity, isiterable, first, last, concatv, map, valfilter, keyfilter, merge, curry
from functools import partial
from operator import contains, methodcaller, itemgetter, attrgetter, not_, truth, abs, invert, neg, pos, index


# ## utilities

# In[199]:

def _do(function, *args, **kwargs):
    """Call function then return the first argument."""
    function(*args, **kwargs)
    return first(args)

def _flip(function, *args, **kwargs):
    """Flip arguments then apply them to function."""
    return function(*reversed(args), **kwargs)

@curry
def call(args, function, **kwargs):
    """Evaluate a function or return non-callable functions.
    
    This is a functor method."""
    if not callable(function):
        return function 
    return function(*args, **kwargs) 


# In[200]:

class Factory:
    """A mixin to generate new callables and compositions.
    """
    def __call__(self, *args, **kwargs):
        return first(self.functions)(args=args, keywords=kwargs)


# In[201]:

class StateMixin:
    """Mixin to reproduce state from __slots__
    """
    def __getstate__(self):
        return tuple(map(partial(getattr, self), self.__slots__))

    def __setstate__(self, state):
        [setattr(self, key, value) for key, value in zip(self.__slots__, state)]


# ## composition

# In[668]:

class Compose(StateMixin, object):
    """Compose a higher-order function.  Recursively evaluate iterables.
    
    Compose([range, [type, list]])
    """
    __slots__ = ('functions', 'recursive', 'returns')
    
    def __init__(self, functions, **kwargs):
        returns = isinstance(functions, Generator) and identity or type(functions)
        
        if not isiterable(functions) or isinstance(functions, str):
            functions = [functions]
        
        if isinstance(functions, dict):
            functions = iteritems(functions)
            
        self.functions = functions
        self.recursive = kwargs.pop('recursive', True)
        self.returns = self.recursive and last or kwargs.pop('returns', returns)
        
    def __call__(self, *args, **kwargs):
        if not self.functions:
            # default to an identity function
            return args[0] if args else None
        
        # call the function 
        return self.returns(self._icall_(*args, **kwargs))
        
    def _icall_(self, *args, **kwargs):
        """Generator yielding each evaluated function.
        """
        for function in self.functions:
            if isiterable(function) and not isinstance(function, (str, Callable)):
                function = Compose(function, recursive=False)                
            
            result = call(args, function, **kwargs)
            
            yield result
            
            if self.recursive: 
                args, kwargs = ((result,), {})            


# In[753]:

class Callable(StateMixin, object):
    _do = False
    _flip = False
    
    __slots__ = ('args', 'keywords', 'functions', '_annotations_', '__doc__', '_strict')

    def __init__(
        self, functions=tuple(), args=tuple(), keywords=dict(), _annotations_=None,
        doc="""""", _strict=True
    ):
        self._annotations_ = _annotations_
        self.args = args
        self.functions = functions 
        self.keywords = keywords
        self._strict = _strict
        self.__doc__ = __doc__
        
    def _partial_(self, *args, **kwargs):
        """Update arguments and keywords of the current composition.
        """
        args and setattr(self, 'args', args) 
        kwargs and setattr(self, 'keywords', kwargs)
        return self
    
    def __getitem__(self, item=None):
        """Main feature to append functions to the composition.
        
        call is special flag to execute the function.
        """
        if not isinstance(item, This): 
            if getattr(item, 'func', identity) == call.func:
                args = item._partial.args and item._partial.args[0] or tuple()
                if not isinstance(args, tuple):
                    args = (args,)
                return self(*args, **item._partial.keywords)
        
        self = isinstance(self, Factory) and self() or self
        
        return item and not(item==slice(None)) and setattr(
            self, 'functions', tuple(concatv(self.functions, (item,)))
        ) or self
        
    def __pow__(self, *args, **kwargs):
        """Append dispatching or annotation features.
        """
        self = self[:]
        args = isinstance(args[0], tuple) and args[0] or args
        
        if len(args) == 1:
            if isinstance(args[0], str):
                return setattr(self, '__doc__', args[0]) or self
            
        return self._dispatch_(*args, **kwargs)

    def _dispatch_(self, *args, **kwargs):
        if callable(args[0]) and not isinstance(args[0], type):
            return setattr(self, '_annotations_', args[0]) or self
            
        return setattr(
            self, '_annotations_', merge(
                    isinstance(self._annotations_, dict) and self._annotations_ or {}, 
                    isinstance(args[0], dict) and args[0] or
                    dict(zip(range(len(args)), args))
            )
        ) or self 
    
    def __lshift__(self, value):
        """Append a `do` composition.
        """
        composition = type('Do', (Composition,), {'_do': True})()[value]
        return isinstance(self, Factory) and composition or self[composition.__call__]
    
    def __call__(self, *args, **kwargs):
        """Call the composition with appending args and kwargs.
        """
        composition = copy(self)._partial_(*(concatv(self.args, args)), **merge(self.keywords, kwargs))
        return call(tuple(), (bool(composition) and self._strict) and composition.__func__)

    @property
    def __func__(self):
        """Compose the composition.
        """
        function, args = Compose(functions=self.functions), reversed(self.args) if self._flip else self.args
        function = self._do and partial(_do, function) or function
        return partial(function, *args, **self.keywords)
            
    def __repr__(self):
        """String representation of the composition.
        """
        return self.functions and (self.args or self.keywords) and repr(self()) or repr({
                'args': self.args, 'kwargs': self.keywords, 'funcs': self.functions
            })
        
    def __copy__(self):
        """Copy the composition.
        """
        copy = self.__class__()
        return copy.__setstate__(self.__getstate__()) or copy
    
    def __bool__(self):
        """Validate any annotations for the composition.
        """
        if not self._annotations_: return True
        
        if callable(self._annotations_):
            return bool(self._annotations_(*self.args, **self.keywords))
        
        return all(
            i in self.__annotations__ and 
            (partial(_flip, isinstance, self._annotations_[i]) 
             if isinstance(self.__annotations__[i], type) 
             else self._annotations_[i])(arg) or False
            for i, arg in concatv(enumerate(self.args), self.keywords.items())
        )
    
    @property
    def __annotations__(self):
        return isinstance(self._annotations_, dict) and self._annotations_ or {}

    @staticmethod
    def _set_attribute_(method, _partial=True):
        """Decorator a append attributes to callable class"""
        def call(self, *args, **kwargs):    
            return (args or kwargs) and self[
                partial(method, *args, **kwargs) if _partial else method(*args, **kwargs)
            ] or self[method]
        return wraps(method)(call)
    
class This(Callable):
    def __getattr__(self, attr):
        if any(attr.startswith(key) for key in ('_repr_', '_ipython_')):
            return self
        return super(This, self).__getitem__(callable(attr) and attr or attrgetter(attr))
    
    def __getitem__(self, item):
        if isinstance(item, Factory):
            composition = item.functions[0]()
            return composition.__setstate__(self.__getstate__()) or composition
        return super(This, self).__getitem__(callable(item) and item or itemgetter(item))
    
    def __call__(self, *args, **kwargs):
        if type(last(self.functions)) == attrgetter:
            names = last(self.functions).__reduce__()[-1]
            if len(names) == 1:
                setattr(self, 'functions', tuple(self.functions[:-1])) 
                self[methodcaller(names[0], *args, **kwargs)]  
            return self
        return super(This, self).__call__(*args, **kwargs)


# ## types

# In[754]:

Composition = type('Composition', (Callable,), {'partial': Callable._partial_})
Flipped = type('Flipped', (Composition,), {'_flip': True})
    
class Stars(Composition):
    @property
    def __func__(self):
        composition = super(Stars, self).__func__
        args, kwargs = composition.args and composition.args[0] or tuple(), composition.keywords
        if isinstance(args, dict):
            args = kwargs.update(args) or tuple()
        return partial(composition.func, *args, **kwargs)


# ## namespace

# In[755]:

_x, x_, _xx, this = (
    type('_{}_'.format(f.__name__), (Factory, f,), {})((f,)) for f in (Composition, Flipped, Stars, This)
)
stars = _xx


# ## Append attributes for a chainable API

# In[756]:

This.__rshift__ = This.__getitem__
Callable.__rshift__ = Callable.__getitem__
Callable.__deepcopy__ = Callable.__copy__
Callable._ = Callable.__call__
Composition.call = Composition.__call__
Composition.do = Composition.__lshift__ 
Composition.partial = Composition._partial_
Composition.pipe =  Composition.__getitem__

for imports in ('toolz', 'operator'):
    for name, function in iteritems(
        valfilter(callable, keyfilter(Compose([first, str.islower]), vars(import_module(imports))))
    ):
        if getattr(Composition, name, None) is None:
            opts = {}
            if function is contains or imports == 'toolz': 
                pass                
            elif function in (methodcaller, itemgetter, attrgetter, not_, truth, abs, invert, neg, pos, index):
                opts.update(_partial=False)
            else:
                function = partial(_flip, function)
            setattr(Composition, name, getattr(
                Composition, name, Composition._set_attribute_(function, **opts)
            ))

Composition.dispatch = Composition.__pow__
Composition.__matmul__ = Composition.groupby
Composition.__mul__ = Composition.map 
Composition.__truediv__  = Composition.filter


# In[ ]:

# import requests
# from IPython.display import display, Image
# from toolz.curried import *
# GET = memoize(requests.get)

# _x('tonyfast')>>"https://api.github.com/users/{}".format>>GET>>this.json()>>{
#     'name': get('name'),
#     'img': _x[{'url': get('avatar_url')}]>>_xx(width=100)[Image]<<display

# }


# In[796]:

# from bokeh import plotting, models
# import pandas as pd

# source, p = pd.util.testing.makeDataFrame().pipe(plotting.ColumnDataSource), plotting.figure()
# renderers = _x[[
#     models.Circle(x='A', y='B'),
#     models.Square(x='A', y='B')
# ]].map(_x(source)[p.add_glyph])[list]()


# In[763]:

# f = this[['A', 'B']][
#     _x<<len>>"The length of the dataframe is {}".format>>print
# ].describe().unstack(0).reset_index().pipe(
#     _x<<(lambda df: setattr(df, 'columns', df.columns.map(str)))
# ).pipe(
#     _x<<(lambda df: setattr(df, '0', df['0'].mul(100)))
# ).sample(2)
# f(df)

# f(pd.concat([pd.util.testing.makeDataFrame() for i in range(3)]))

