
# coding: utf-8

# In[1]:

try:
    from .objects import  Compose, Partial
    from .callables import flipped, do, step, starred, excepts, ifnot, ifthen
    from .namespaces import Namespaces, composed
except Exception as e:
    from objects import  Compose, Partial
    from callables import flipped, do, step, starred, excepts, ifnot, ifthen
    from namespaces import Namespaces, composed
    
from functools import partial, wraps
from six import PY3
from copy import copy
from operator import attrgetter
from toolz.curried import (isiterable, flip, complement, interpose, groupby, compose, merge, reduce, filter, map)
_attribute_ = "__{}{}__".format


# In[2]:

__all__ = ['flips', 'stars', 'does', 'maps', 'filters', 'groups', 'reduces']
functions = (flipped, starred, do, map, filter, groupby, reduce)

_mro_ = compose(dict.values, merge, map(vars), attrgetter('__mro__'), type)
_isinstance_ = flip(isinstance)


# In[3]:

class Syntax(object):    
    def __xor__(self, object):
        self = self[:]  # noqa: F823
        if not isiterable(object) and isinstance(object, type):
            object = (object,)
            
        if isiterable(object):
            if all(map(_isinstance_(type), object)) and all(map(flip(issubclass)(BaseException), object)):
                self.function = Compose(excepts(object, self.function))
                return self
            
            if all(map(_isinstance_(BaseException), object)):
                object = tuple(map(type, object))
                
            if all(map(_isinstance_(type), object)):
                object = _isinstance_(object)

        self.function = Compose([ifthen(Compose([object]), self.function)])
        return self

    def __or__(self, object):
        self = self[:]
        self.function = Compose([ifnot(self.function, Compose([object]))])
        return self
    
    def __and__(self, object):
        self = self[:]
        self.function = Compose([step(self.function, Compose([object]))])
        return self
    
    def __pos__(self):
        return self[bool]

    def __neg__(self):
        return self[complement(bool)]
        
    def __lshift__(self, object):
        return Does()[object] if self._factory_ else self[do(object)]
    
    def __round__(self, n):
        self.function.function = list(interpose(n, self.function.function))
        return self
    
    __invert__, __pow__ = Partial.__reversed__, __xor__


# In[4]:

class Models(Partial, Syntax, Namespaces):
    @property
    def _factory_(self):
          return type(self).__name__.startswith('_') and type(self).__name__.endswith('_')
        
    def __getitem__(self, object=slice(None), *args, **kwargs):
        self = self.function() if self._factory_ else self
        
        return super(Models, self).__getitem__(
            object() if isinstance(object, Models) and object._factory_ else object, 
            *args, **kwargs)
    
    def __call__(self, *args, **kwargs):
        return super(Models, self).__call__(*args, **kwargs)
        
    __mul__ = __add__ = __rshift__ = __sub__ = __getitem__
    
    @property
    def __doc__(self):
        string=""
        for function in self:
            string += (getattr(function, 'func', function).__doc__ or "") + "\n---\n"
        return string


# In[5]:

for name, function in zip(__all__, functions):
    locals().update({name.capitalize(): type(name.capitalize(), (Models,), {'type': staticmethod(function)})})

__all__ += ['models']

for fidget in __all__:
    callable = locals()[fidget.capitalize()]
    locals()[fidget] = type('_{}_'.format(fidget.capitalize()), (callable,), {})()
    locals()[fidget].function = Compose([callable])
    


for op, func in (('matmul', 'groupby'), ('truediv', 'map'), ('floordiv', 'filter'), ('mod', 'reduce')):
    setattr(Models, _attribute_('', op), property(Compose(attrgetter(func))))
Models.__div__  = Models.__truediv__ 


# In[11]:

def fallback(attr):
    def fallback(right, left):
        return getattr(Models()[left], attr)(right[:])
    return wraps(getattr(Models, attr))(fallback)


# In[7]:

for attr in ['add', 'sub', 'mul', 'matmul','div', 'truediv', 'floordiv', 'mod', 'lshift', 'rshift', 'and', 'xor', 'or', 'pow']:
    setattr(Models, _attribute_('i', attr), getattr(Models, _attribute_('', attr)))
    setattr(Models, _attribute_('r', attr), fallback(_attribute_('', attr)))


# In[8]:

Models.namespaces['fidget'].update({
    f.__name__: composed(f) for f in (groupby, reduce, filter, map)})
Models.namespaces['fidget'].update({
    key: getattr(Models, _attribute_('', value)) 
    for key, value in [['call']*2, ['do', 'lshift'], ['pipe',  'getitem'], ['ifthen','xor'], ['step', 'and'], ['ifnot', 'or']]})

