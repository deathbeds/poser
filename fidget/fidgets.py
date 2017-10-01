
# coding: utf-8

# # fidgets
# 
# * pipes - compose functions in serial to pipe arguments through.
# * juxts - juxtapose the same arguments on many functi
# * does - compose a function that returns the input arguments
# * flips - compose a function that flips the input arguments before evaluation
# * stars - compose a function starring sequence arguments and container keywords 
# * maps - 
# * filters
# * groups
# * reduces

# In[84]:


try:
    from .functions import flipped, do, starred, step, ifthen, ifnot, excepts
    from .composites import Compose, Partial, Juxtapose, calls
except Exception as e:
    from functions import flipped, do, starred, step, ifthen, ifnot, excepts
    from composites import Compose, Partial, Juxtapose, calls
    
from functools import wraps, partialmethod
from operator import attrgetter
from toolz.curried import groupby, compose, merge, reduce, filter, map, partial, merge, isiterable, flip, complement, interpose, concat, identity
from inspect import signature
from importlib import import_module
from six import PY3
_attribute_ = "__{}{}__".format
_isinstance_ = flip(isinstance)


# In[85]:


__all__ = ['flips', 'stars', 'does', 'maps', 'filters', 'groups', 'reduces']
functions = (flipped, starred, do, map, filter, groupby, reduce)


# In[86]:


def _composed(callable):
    def composed(*args, **kwargs):
        args = (calls(args[0]), *args[1:])
        return callable(*args, **kwargs)
    return wraps(callable)(composed)


def _curried(callable):
    def curried(*args):
        function = callable
        for arg in args:
            function = function(arg)
        return function
    return wraps(callable)(curried)


# In[119]:


class Operations(object):    
    _attributes = list()

    def _and_(self, callable, object):
        self = self[:]
        self.function = Composes()[Compose([callable(self.function, Compose([object]))])]
        return self
    
    __or__ = partialmethod(_and_, ifnot)
    __and__ = partialmethod(_and_, step)
    
    def __xor__(self, object):
        object = type(object) is type and (object,) or object
        self = self[:]
        self.function = Composes()[
            Compose([excepts(object, self.function)])]
        return self

    def __pow__(self, object):
        object = type(object) is type and (object,) or object
        self = self[:]
        self.function = Composes()[
            Compose([ifthen(callable(object) and object or _isinstance_(object), self.function)])]
        return self

    
    def __pos__(self):
        return self[bool]

    def __neg__(self):
        return self[complement(bool)]
            
    def __round__(self, n):
        self.function.function = list(interpose(n, self.function.function))
        return self
    
    _attributes = list()

    def __getattr__(self, attribute):
        for dict in reversed(self._attributes):
            dict = getattr(dict, '__dict__', dict)
            if attribute in dict:
                callable = dict[attribute]
                
                doc = callable.__doc__
                try:
                    sig = signature(callable)
                except:
                    sig=None

                callable = partial(self.__getitem__, callable)
                
                PY3 and setattr(callable, '__doc__', doc)
                sig and setattr(callable, '__signature__', sig)
                
                return callable
        raise AttributeError("No attribute {}".format(attribute))

    def __dir__(self):
        return set(concat(map(dir, self._attributes)))


# In[120]:


class Factory(Partial):
    @property
    def _factory_(self):
          return type(self).__name__.startswith('_') and type(self).__name__.endswith('_')
        
    def __getitem__(self, object=slice(None), *args, **kwargs):
        self = self.function() if self._factory_ else self
        
        return super(Factory, self).__getitem__(
            object() if isinstance(object, Factory) and object._factory_ else object, 
            *args, **kwargs)
    
    def __lshift__(self, object):
        return Does()[object] if self._factory_ else self[do(object)]

    __mul__ = __add__ = __rshift__ = __sub__ = __getitem__
    __invert__= Partial.__reversed__
 


# In[121]:


class Composes(Factory, Operations): 
    @property
    def __doc__(self):
        return '\n'.join([
            (getattr(function, 'func', function).__doc__ or "") + "\n---\n" 
            for function in self])

class Juxts(Factory):
    _wrapper, _composition = map(staticmethod, (tuple, Juxtapose))


# In[122]:


for op, func in (('matmul', 'groupby'), ('truediv', 'map'), ('floordiv', 'filter'), ('mod', 'reduce')):
    setattr(Composes, _attribute_('', op), property(Compose(attrgetter(func))))
Composes.__div__  = Composes.__truediv__ 

def _right_fallback(attr):
    def fallback(right, left):
        return getattr(Composes()[left], attr)(Composes()[right])
    return wraps(getattr(Composes, attr))(fallback)

for attr in ['add', 'sub', 'mul', 'matmul','div', 'truediv', 'floordiv', 'mod', 'lshift', 'rshift', 'and', 'xor', 'or', 'pow']:
    setattr(Composes, _attribute_('i', attr), getattr(Composes, _attribute_('', attr)))
    setattr(Composes, _attribute_('r', attr), _right_fallback(_attribute_('', attr)))

Operations._attributes = list()
Operations._attributes.append(__import__('pathlib'))
from pathlib import Path
Operations._attributes.append({
     k: flipped(getattr(Path, k)) for k in dir(Path) if k[0]!='_' and callable(getattr(Path, k))
})
Operations._attributes.append(__import__('json'))
Operations._attributes.append(__import__('itertools'))
Operations._attributes.append(__import__('collections'))
Operations._attributes.append(__import__('six').moves.builtins)
Operations._attributes.append({
    key: _curried(value) if key in ['attrgetter', 'methodcaller', 'itemgetter'] else flipped(value)
    for key, value in vars(__import__('operator')).items() if key[0].islower()
})
Operations._attributes.append({
    key: _composed(value) if any(map(key.endswith, ('filter', 'map'))) else value
    for key, value in vars(__import__('toolz')).items() if key[0].islower()
})


# In[123]:


for name, function in zip(__all__, functions):
    locals().update({name.capitalize(): type(name.capitalize(), (Composes,), {'_wrapper': staticmethod(function)})})

__all__ += ['composes', 'juxts']

for fidget in __all__:
    _callable = locals()[fidget.capitalize()]
    locals()[fidget] = type('_{}_'.format(fidget.capitalize()), (_callable,), {})()
    locals()[fidget].function = Compose([_callable])


# In[124]:


Composes._attributes.append({
    f.__name__: _composed(f) for f in (groupby, reduce, filter, map)})
Composes._attributes.append({
    key: getattr(Composes, _attribute_('', value)) 
    for key, value in [['call']*2, ['do', 'lshift'], ['pipe',  'getitem'], ['ifthen','xor'], ['step', 'and'], ['ifnot', 'or']]})


#         from pandas import *
# 
#         (
#             composes.Path('/Users/tonyfast/gists/')
#             .rglob('*.ipynb').take(3)
#             / [str, composes.read_text() * composes(as_version=4)[__import__('nbformat').reads]]
#             * composes.dict().valmap(composes.get('cells')[DataFrame]) 
#             * concat
#         )()
# 