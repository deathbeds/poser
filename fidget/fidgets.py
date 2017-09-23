
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

# In[1]:


try:
    from .callables import flipped, do, starred, Compose, Partial, Juxtapose
    from .attributes import Attributes, Operators, composed
except Exception as e:
    from callables import flipped, do, starred, Compose, Partial, Juxtapose
    from attributes import Attributes, Operators, composed
    
from functools import wraps

from operator import attrgetter
from toolz.curried import groupby, compose, merge, reduce, filter, map
_attribute_ = "__{}{}__".format


# In[2]:


__all__ = ['flips', 'stars', 'does', 'maps', 'filters', 'groups', 'reduces']
functions = (flipped, starred, do, map, filter, groupby, reduce)


# In[3]:


class Factory(Partial):
    @property
    def _factory_(self):
          return type(self).__name__.startswith('_') and type(self).__name__.endswith('_')
        
    def __getitem__(self, object=slice(None), *args, **kwargs):
        self = self.function() if self._factory_ else self
        
        return super(Factory, self).__getitem__(
            object() if isinstance(object, Factory) and object._factory_ else object, 
            *args, **kwargs)

    __mul__ = __add__ = __rshift__ = __sub__ = __getitem__
    __invert__, __pow__ = Partial.__reversed__, Operators.__xor__
    
    def __lshift__(self, object):
        return Does()[object] if self._factory_ else self[do(object)]


# In[4]:


class Pipes(Factory, Operators, Attributes): 
    @property
    def __doc__(self):
        return '\n'.join([
            (getattr(function, 'func', function).__doc__ or "") + "\n---\n" 
            for function in self])

class Juxts(Factory):
    _wrapper, _composition = map(staticmethod, (tuple, Juxtapose))


# In[5]:


for name, function in zip(__all__, functions):
    locals().update({name.capitalize(): type(name.capitalize(), (Pipes,), {'_wrapper': staticmethod(function)})})

__all__ += ['pipes', 'juxts']

for fidget in __all__:
    callable = locals()[fidget.capitalize()]
    locals()[fidget] = type('_{}_'.format(fidget.capitalize()), (callable,), {})()
    locals()[fidget].function = Compose([callable])
    
__all__ += ['a', 'an', 'the']; a = an = the = pipes

for op, func in (('matmul', 'groupby'), ('truediv', 'map'), ('floordiv', 'filter'), ('mod', 'reduce')):
    setattr(Pipes, _attribute_('', op), property(Compose(attrgetter(func))))
Pipes.__div__  = Pipes.__truediv__ 


# In[6]:


def fallback(attr):
    def fallback(right, left):
        return getattr(Pipes()[left], attr)(Pipes()[right])
    return wraps(getattr(Pipes, attr))(fallback)


# In[7]:


for attr in ['add', 'sub', 'mul', 'matmul','div', 'truediv', 'floordiv', 'mod', 'lshift', 'rshift', 'and', 'xor', 'or', 'pow']:
    setattr(Pipes, _attribute_('i', attr), getattr(Pipes, _attribute_('', attr)))
    setattr(Pipes, _attribute_('r', attr), fallback(_attribute_('', attr)))


# In[8]:


Pipes.attributes['fidget'] = {
    f.__name__: composed(f) for f in (groupby, reduce, filter, map)}
Pipes.attributes['fidget'].update({
    key: getattr(Pipes, _attribute_('', value)) 
    for key, value in [['call']*2, ['do', 'lshift'], ['pipe',  'getitem'], ['ifthen','xor'], ['step', 'and'], ['ifnot', 'or']]})

