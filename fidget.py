
# coding: utf-8

# # Higher-order functions for interactive computing

# In[20]:

__all__ = ['_x', '_xx', 'call', 'stars', 'this', 'x_',]

from copy import copy
from functools import wraps
from importlib import import_module
from collections import Generator
from six import iteritems
from toolz.curried import isiterable, first, last, identity, concatv, map, valfilter, keyfilter, merge, curry
from functools import partial
from operator import contains, methodcaller, itemgetter, attrgetter, not_, truth, abs, invert, neg, pos, index


# ## utilities

# In[21]:

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


# In[22]:

class FactoryMixin:
    """A mixin to generate new callables and compositions.
    """
    def __call__(self, *args, **kwargs):
        return first(self._functions)(args=args, keywords=kwargs)


# In[23]:

class StateMixin:
    """Mixin to reproduce state from __slots__
    """
    def __getstate__(self):
        return tuple(map(partial(getattr, self), self.__slots__))

    def __setstate__(self, state):
        [setattr(self, key, value) for key, value in zip(self.__slots__, state)]


# ## composition

# In[49]:

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
        return self.returns(self.__iter_call__(*args, **kwargs))
    
    def __iter_call__(self, *args, **kwargs):
        for function in self:
            result = call(args, function, **kwargs)            
            yield result
            if self.recursive: 
                args, kwargs = ((result,), {})            
        
    def __iter__(self):
        """Generator yielding each evaluated function.
        """
        for function in self.functions:
            if isiterable(function) and not isinstance(function, (str, Callable)):
                function = Compose(function, recursive=False)                
            yield function
            
    def __len__(self):
        return len(self.functions)


# In[159]:

class Callable(StateMixin, object):
    _do = False
    _flip = False
    
    __slots__ = ('_functions', '_args', '_keywords', '_annotations_', '_doc', '_strict')

    def __init__(
        self, functions=tuple(), args=tuple(), keywords=dict(), _annotations_=None,
        doc="""""", _strict=True
    ):
        self._annotations_ = _annotations_
        self._args = args
        self._functions = functions 
        self._keywords = keywords
        self._strict = _strict
        self._doc = doc
        
    def _partial_(self, *args, **kwargs):
        """Update arguments and keywords of the current composition.
        """
        args and setattr(self, '_args', args) 
        kwargs and setattr(self, '_keywords', kwargs)
        return self
    
    def __getitem__(self, item=None):
        """Main feature to append functions to the composition.
        
        call is special flag to execute the function.
        """
        if not isinstance(item, This): 
            if getattr(item, 'func', identity) == call.func:
                args = item._partial.args and item._partial.args[0] or tuple()
                if not isinstance(args, (tuple, list)):
                    args = (args,)
                return self(*args, **item._partial.keywords)
        
        self = isinstance(self, FactoryMixin) and self() or self
        
        return item and not(item==slice(None)) and setattr(
            self, '_functions', tuple(concatv(self._functions, (item,)))
        ) or self
        
    def __pow__(self, *args, **kwargs):
        """Append dispatching or annotation features.
        """
        self = self[:]
        args = isinstance(args[0], tuple) and args[0] or args
        
        if len(args) is 1 and isinstance(args[0], str):
            return setattr(self, '_doc', args[0]) or self
            
        return self._validate_(*args, **kwargs)

    def _validate_(self, *args, **kwargs):
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
        return isinstance(self, FactoryMixin) and composition or self[composition.__call__]
    
    def __call__(self, *args, **kwargs):
        """Call the composition with appending args and kwargs.
        """
        composition = copy(self)._partial_(*(concatv(self._args, args)), **merge(self._keywords, kwargs))
        return call(tuple(), (bool(composition) and self._strict) and composition.__func__)

    @property
    def __func__(self):
        """Compose the composition.
        """
        function, args = Compose(functions=self._functions), reversed(self._args) if self._flip else self._args
        function = self._do and partial(_do, function) or function
        return partial(function, *args, **self._keywords)
            
    def __repr__(self):
        """String representation of the composition.
        """
        return self._functions and (self._args or self._keywords) and repr(self()) or repr({
                'args': self._args, 'kwargs': self._keywords, 'funcs': self._functions
            })
        
    def __copy__(self):
        """Copy the composition.
        """
        copy = self.__class__()
        return copy.__setstate__(self.__getstate__()) or copy
    
    def __bool__(self):
        """Validate any annotations for the composition.
        """
        annotations = self.__annotations__
        
        if not annotations: return True
        
        if callable(annotations):
            return bool(annotations(*self._args, **self._keywords))
                
        return all(
            i in annotations and (
                partial(_flip, isinstance, annotations[i]) 
                if isinstance(annotations[i], type) 
                else annotations[i]
            )(arg) or False
            for i, arg in concatv(enumerate(self._args), self._keywords.items())
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
    
    def __len__(self):
        return len(self.functions)

    @property
    def __doc__(self):
        return self._doc

class This(Callable):
    def __getattr__(self, attr):
        if any(attr.startswith(key) for key in ('_repr_', '_ipython_')):
            return self
        return super(This, self).__getitem__(callable(attr) and attr or attrgetter(attr))
    
    def __getitem__(self, item):
        if isinstance(item, FactoryMixin):
            composition = item.functions[0]()
            return composition.__setstate__(self.__getstate__()) or composition
        return super(This, self).__getitem__(callable(item) and item or itemgetter(item))
    
    def __call__(self, *args, **kwargs):
        if type(last(self._functions)) == attrgetter:
            names = last(self._functions).__reduce__()[-1]
            if len(names) == 1:
                setattr(self, '_functions', tuple(self._functions[:-1])) 
                self[methodcaller(names[0], *args, **kwargs)]  
            return self
        return super(This, self).__call__(*args, **kwargs)


# ## types

# In[160]:

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

# In[161]:

_x, x_, _xx, this = (
    type('_{}_'.format(f.__name__), (FactoryMixin, f,), {})((f,)) for f in (Composition, Flipped, Stars, This)
)
stars = _xx


# ## Append attributes for a chainable API

# In[162]:

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

Composition.validate = Composition.__pow__
Composition.__matmul__ = Composition.groupby
Composition.__mul__ = Composition.map 
Composition.__truediv__  = Composition.filter


# In[192]:

# _x([1, 2]).attrgetter('index').excepts(ValueError, handler=_x[_x<<print][type])()(10)


# In[191]:

# import requests
# from IPython.display import display, Image
# from toolz.curried import *
# GET = memoize(requests.get)

# _x('tonyfast')>>"https://api.github.com/users/{}".format>>GET>>this.json()>>{
#     'name': get('name'),
#     'img': _x[{'url': get('avatar_url')}]>>_xx(width=100)[Image]<<display

# }

# from bokeh import plotting, models
# import pandas as pd
# df = pd.util.testing.makeDataFrame()
# source, p = df.pipe(plotting.ColumnDataSource), plotting.figure()
# renderers = _x[[
#     models.Circle(x='A', y='B'),
#     models.Square(x='A', y='B')
# ]].map(_x(source)[p.add_glyph])[list]()


# In[58]:

# f = this[['A', 'B']][
#     _x<<len>>"The length of the dataframe is {}".format>>print
# ].describe().unstack(0).reset_index().pipe(
#     _x<<(lambda df: setattr(df, 'columns', df.columns.map(str)))
# ).pipe(
#     _x<<(lambda df: setattr(df, '0', df['0'].mul(100)))
# ).sample(2)
# f(df)

# f(pd.concat([pd.util.testing.makeDataFrame() for i in range(3)]))


# In[ ]:




# In[93]:

_x()**(int)>>range>>call([10])

