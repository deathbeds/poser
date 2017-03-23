
# coding: utf-8

# # Higher-order functions for interactive computing

# In[1]:

__all__ = ['_x', '_xx', 'call', 'stars', 'this', 'x_',]

from copy import copy
from functools import wraps
from importlib import import_module
from collections import Generator
from six import iteritems
from toolz.curried import isiterable, first, excepts, flip, last, identity, concatv, map, valfilter, keyfilter, merge, curry
from functools import partial
from operator import contains, methodcaller, itemgetter, attrgetter, not_, truth, abs, invert, neg, pos, index


# ## utilities

# In[2]:

def _do(function, *args, **kwargs):
    """Call function then return the first argument."""
    function(*args, **kwargs)
    return first(args)

@curry
def call(args, function, **kwargs):
    """Evaluate a function or return non-callable functions.
    
    This is a functor method."""
    if not callable(function):
        return function 
    return function(*args, **kwargs) 


# In[3]:

class FactoryMixin:
    """A mixin to generate new callables and compositions.
    """
    def __call__(self, *args, **kwargs):
        return first(self._functions)(args=args, keywords=kwargs)


# In[4]:

class StateMixin:
    """Mixin to reproduce state from __slots__
    """
    def __getstate__(self):
        return tuple(map(partial(getattr, self), self.__slots__))

    def __setstate__(self, state):
        [setattr(self, key, value) for key, value in zip(self.__slots__, state)]


# ## composition

# In[87]:

class FunctionsBase(StateMixin, object):
    __slots__ = ('functions', 'codomain')
    
    def __init__(self, functions):        
        if not isiterable(functions) or isinstance(functions, (str, Callable)):
            functions = [functions]
        
        if isinstance(functions, dict):
            functions = iteritems(functions)
            
        self.functions = functions
                    
    def __iter__(self):
        """Generator yielding each evaluated function.
        """
        for function in self.functions:
            if isiterable(function) and not isinstance(function, (str, Callable)):
                function = Juxtapose(function)
            yield function

    def __len__(self):
        return len(self.functions)
    
class Juxtapose(FunctionsBase):
    def __init__(self, functions):
        self.codomain = isinstance(functions, Generator) and identity or type(functions)
        super(Juxtapose, self).__init__(functions)
        
    def __call__(self, *args, **kwargs):
        return self.codomain(map(call(args, **kwargs), self))
        
class Compose(FunctionsBase):
    def __init__(self, functions):
        self.codomain = identity
        super(Compose, self).__init__(functions)
    
    def __call__(self, *args, **kwargs):
        for function in self:
            args, kwargs = (call(args, function, **kwargs),), {}
        return self.codomain(args[0])


# In[88]:

from inspect import getfullargspec


# In[183]:

class Callable(StateMixin, object):
    _do = False
    _flip = False
    
    __slots__ = ('_functions', '_args', '_keywords', '_annotations_', '_doc', '_strict')

    def __init__(
        self, functions=tuple(), args=tuple(), keywords=dict(), _annotations_=None,
        doc="""""", _strict=True
    ):
        self._annotations_ = _annotations_
        self._functions = functions 
        self._strict = _strict
        self._doc = doc
        self._args, self._keywords = tuple(), {}
        self._partial_(*args, **keywords)
        
    def _partial_(self, *args, **kwargs):
        """Update arguments and keywords of the current composition.
        """
        append = kwargs.pop('append', False)
        args and setattr(self, '_args', tuple(
            concatv(append and self._args or tuple(), args)
        )) 
        kwargs and setattr(self, '_keywords', merge(
            append and self._keywords or {}, kwargs
        ))
        return self
    
    def __getitem__(self, item=None):
        """Main feature to append functions to the composition.
        
        call is special flag to execute the function.
        """
        if not isinstance(item, This): 
            if getattr(item, 'func', False) == call.func:
                args = item._partial.args and item._partial.args[0] or tuple()
                if not isinstance(args, (tuple, list)):
                    args = (args,)
                return self(*args, **item._partial.keywords)
        
        self = isinstance(self, FactoryMixin) and self() or self
        
        item != slice(None) and setattr(
            self, '_functions', tuple(concatv(self._functions, (item,)))
        )
        
        return self
        
    def __pow__(self, *args, **kwargs):
        """Append dispatching or annotation features.
        """
        self = self[:]
        args = isinstance(args[0], tuple) and args[0] or args
        
        if len(args) is 1 and isinstance(args[0], str):
            return setattr(self, '_doc', args[0]) or self
            
        return self._validate_(*args, **kwargs)

    def _validate_(self, *args, **kwargs):
        if not isinstance(args[0], type):
            return setattr(self, '_annotations_', args[0]) or self
        annotations = self._annotations_ or {}
        return setattr(
            self, '_annotations_', merge(annotations, dict(enumerate(args)), kwargs)
        ) or self 
    
    def __lshift__(self, value):
        """Append a `do` composition.
        """
        composition = type('Do', (Composition,), {'_do': True})()[value]
        return isinstance(self, FactoryMixin) and composition or self[composition.__call__]
    
    def __call__(self, *args, **kwargs):
        """Call the composition with appending args and kwargs.
        """
        composition = copy(self)._partial_(
            *args, **kwargs, append=True
        )
        passes = self._strict and bool(composition)
        return call(
            tuple(), 
            passes and composition.__func__
        )

    @property
    def __func__(self):
        """Compose the composition.
        """
        function, args = Compose(self._functions), reversed(self._args) if self._flip else self._args
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
        annotations = self._annotations_ or dict()
        if not annotations: return True
        
        if callable(annotations):
            return bool(
                annotations(*self._args, **self._keywords)
            ) if self._args or self._keywords else True
                
        return all(
            (
                isinstance(annotations[i], type)
                and flip(isinstance)(annotations[i]) 
                or annotations[i]
            )(arg) for i, arg in concatv(
                enumerate(self._args), self._keywords.items()
            ) if i in annotations
        )
    
    def __eq__(self, other):
        return self._functions == other._functions and self._args == other._args
    
    @property
    def __annotations__(self):
        kwargs = {}
        if self._functions:
            try:
                argspec = getfullargspec(self._functions[0])
                kwargs = {
                    arg: argspec.annotations[arg]
                    for arg in argspec.args
                    if arg in argspec.annotations
                }
            except:
                pass
        return merge(kwargs, isinstance(self._annotations_, dict) and self._annotations_ or {})

    @staticmethod
    def _set_attribute_(method, _partial=True):
        """Decorator a append attributes to callable class"""
        def call(self, *args, **kwargs):    
            return (args or kwargs) and self[
                _partial and partial(method, *args, **kwargs) or method(*args, **kwargs)
            ] or self[method]
        return wraps(method)(call)
    
    def __len__(self):
        return len(self._functions)

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
            attrs = last(self._functions).__reduce__()[-1]
            if len(attrs) == 1:
                setattr(self, '_functions', tuple(self._functions[:-1])) 
                self[methodcaller(attrs[0], *args, **kwargs)]  
            return self
        return super(This, self).__call__(*args, **kwargs)


# ## types

# In[184]:

class Composition(Callable): 
    partial = Callable._partial_
    
    def __or__(self, item):
        self = self[:]
        return setattr(self, '_functions', [excepts(
                item, Compose(self._functions), handler=partial(call, tuple())
            )]) or self
    
Flipped = type('Flipped', (Composition,), {'_flip': True})
    
class Stars(Composition):
    def _partial_(self, *args, **kwargs):
        """Update arguments and keywords of the current composition.
        """
        if len(args) > 1:
            args = (args,)
        elif args:
            if isiterable(args[0]):
                args = args[0] or tuple()
                if isinstance(args[0], dict):
                    args = kwargs.update(args) or tuple()
        return super(Stars, self)._partial_(*args, **kwargs)


# ## namespace

# In[185]:

_x, x_, _xx, this = (
    type('_{}_'.format(f.__name__), (FactoryMixin, f,), {})((f,)) for f in (Composition, Flipped, Stars, This)
)
stars = _xx


# ## Append attributes for a chainable API

# In[161]:

This.__rshift__ = This.__getitem__
Callable.__rshift__ = Callable.__getitem__
Callable.__deepcopy__ = Callable.__copy__
Callable._ = Callable.__call__
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


# In[156]:

# _x**(_x[_x.gt(5), _x**float]>>all)>>[[identity]*3]>>call(10.)


# In[158]:

# _xx([10, 10, 30])>>range


# In[592]:

# !jupyter nbconvert --to script fidget.ipynb


# In[81]:

# import requests
# from IPython.display import display, Image
# from toolz.curried import *
# GET = memoize(requests.get)

# _x('tonyfast')>>"https://api.github.com/users/{}".format>>GET>>this.json()>>{
#     'name': get('name'),
#     'img': _x[{'url': get('avatar_url')}]>>_xx(width=100)[Image]<<display

# }


# In[82]:

# from bokeh import plotting, models
# import pandas as pd
# df = pd.util.testing.makeDataFrame()
# source, p = df.pipe(plotting.ColumnDataSource), plotting.figure()
# renderers = _x[[
#     models.Circle(x='A', y='B'),
#     models.Square(x='A', y='B')
# ]].map(_x(source)[p.add_glyph])[list]()
# renderers


# In[83]:

# f = this[['A', 'B']][
#     _x<<len>>"The length of the dataframe is {}".format>>print
# ].describe().unstack(0).reset_index().pipe(
#     _x<<(lambda df: setattr(df, 'columns', df.columns.map(str)))
# ).pipe(
#     _x<<(lambda df: setattr(df, '0', df['0'].mul(100)))
# ).sample(2)
# f(df)

# f(pd.concat([pd.util.testing.makeDataFrame() for i in range(3)]))


# In[84]:

# _x()**(int)>>range>>call([10])


# In[85]:

# _x[list].excepts(TypeError)([10])


# In[86]:

# (_x[list] | TypeError) >> call(10)

