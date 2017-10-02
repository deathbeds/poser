
# coding: utf-8

# 
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

# In[1]:


try:
    from .functions import flipped, do, starred, step, ifthen, ifnot, excepts, into, getdoc
    from .composites import Compose, Partial, Juxtapose, calls
except Exception as e:
    from functions import flipped, do, starred, step, ifthen, ifnot, excepts, into, getdoc
    from composites import Compose, Partial, Juxtapose, calls
    
from functools import wraps, partialmethod
from operator import attrgetter
from toolz.curried import groupby, compose, merge, reduce, filter, map, partial, merge, isiterable, flip, complement, interpose, concat, identity
from importlib import import_module
from six import PY3
_attribute_ = "__{}{}__".format
_isinstance_ = flip(isinstance)


# In[2]:


__all__ = ['flips', 'stars', 'does', 'maps', 'filters', 'groups', 'reduces']
functions = (flipped, starred, do, map, filter, groupby, reduce)


# In[3]:


def _composed(callable):
    def composed(*args, **kwargs):
        args = (calls(args[0]), *args[1:])
        return callable(*args, **kwargs)
    return wraps(callable)(composed)


def _classed(callable):
    def classed(*args):
        function = callable
        for arg in args:
            function = function(arg)
        return function
    return wraps(callable)(classed)


# In[4]:


class Operations(object):    
    _getattr_ = list()

    def __and__(self, callable, object):
        self = self[:]
        self.function = Composes()[Compose([step(self.function, Compose([object]))])]
        return self

    def __or__(self, callable, object):
        self = self[:]
        self.function = Composes()[Compose([ifnot(self.function, Compose([object]))])]
        return self
    
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

    def __getattr__(self, attribute):
        for dict in reversed(self._getattr_):
            dict = getattr(dict, '__dict__', dict)
            if attribute in dict:
                def wrapper(*args, **kwargs):
                    return self.__getitem__(dict[attribute], *args, **kwargs)
                return wraps(dict[attribute])(wrapper)
        raise AttributeError("No attribute {}".format(attribute))

    def __dir__(self):
        return set(concat((dict.keys if isinstance(attr, dict) else dir)(attr) for attr in self._getattr_))


# In[5]:


class Factory(Partial):
    @property
    def _factory_(self):
          return type(self).__name__.startswith('_') and type(self).__name__.endswith('_')
        
    def __getitem__(self, object=slice(None), *args, **kwargs):
        self = self.function() if self._factory_ else self
        
        return super(Factory, self).__getitem__(object, *args, **kwargs)
    
    def __lshift__(self, object):
        return Does()[object] if self._factory_ else self[do(object)]

    __mul__ = __add__ = __rshift__ = __sub__ = __getitem__
    __invert__= Partial.__reversed__
 


# In[6]:


class Composes(Factory, Operations): 
    @property
    def __doc__(self):
        return '\n---\n'.join(filter(bool, map(getdoc, self)))

class Juxts(Factory):
    _wrapper, _composition = map(staticmethod, (tuple, Juxtapose))


# In[7]:


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

Operations._getattr_ = list()
Operations._getattr_.append(__import__('pathlib'))
from pathlib import Path
Operations._getattr_.append({
     k: flipped(getattr(Path, k)) for k in dir(Path) if k[0]!='_' and callable(getattr(Path, k))
})
Operations._getattr_.append(__import__('json'))
Operations._getattr_.append(__import__('itertools'))
Operations._getattr_.append(__import__('collections'))
Operations._getattr_.append(__import__('six').moves.builtins)
Operations._getattr_.append({
    key: _classed(value) if key in ['attrgetter', 'methodcaller', 'itemgetter'] else flipped(value)
    for key, value in vars(__import__('operator')).items() if key[0].islower()
})
Operations._getattr_.append({
    key: _composed(value) if any(map(key.endswith, ('filter', 'map'))) else value
    for key, value in vars(__import__('toolz')).items() if key[0].islower()
})


# In[8]:


for name, function in zip(__all__, functions):
    locals().update({name.capitalize(): type(name.capitalize(), (Composes,), {'_wrapper': staticmethod(function)})})

__all__ += ['composes', 'juxts']

for fidget in __all__:
    _callable = locals()[fidget.capitalize()]
    locals()[fidget] = type('_{}_'.format(fidget.capitalize()), (_callable,), {})()
    locals()[fidget].function = Compose([_callable])


# In[9]:


Composes._getattr_.append({
    f.__name__: _composed(f) for f in (groupby, reduce, filter, map)})
Composes._getattr_.append({
    key: getattr(Composes, _attribute_('', value)) 
    for key, value in [['call']*2, ['do', 'lshift'], ['pipe',  'getitem'], ['ifthen','xor'], ['step', 'and'], ['ifnot', 'or']]})


# In[10]:


if PY3:    
    for func in __all__:
        setattr(locals()[func], '__doc__', property(into(getdoc)))


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

# In[ ]:




