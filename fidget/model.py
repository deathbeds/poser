
# coding: utf-8

# In[1]:

try:
    from .objects import  Compose, Partial, Juxtapose
    from .callables import flipped, do, starred
    from .attributes import Attributes, Operators, composed
except Exception as e:
    from objects import  Compose, Partial, Juxtapose
    from callables import flipped, do, starred
    from attributes import Attributes, Operators, composed
    
from functools import wraps
from operator import attrgetter
from toolz.curried import groupby, compose, merge, reduce, filter, map
_attribute_ = "__{}{}__".format


# In[2]:

__all__ = ['flips', 'stars', 'does', 'maps', 'filters', 'groups', 'reduces']
functions = (flipped, starred, do, map, filter, groupby, reduce)

_mro_ = compose(dict.values, merge, map(vars), attrgetter('__mro__'), type)


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


# In[4]:

class Append(Factory):    
    __mul__ = __add__ = __rshift__ = __sub__ = Factory.__getitem__
    __invert__, __pow__ = Partial.__reversed__, Operators.__xor__
    
    def __lshift__(self, object):
        return Does()[object] if self._factory_ else self[do(object)]


# In[5]:

class Pipes(Append, Operators, Attributes): 
    @property
    def __doc__(self):
        return '\n'.join([
            (getattr(function, 'func', function).__doc__ or "") + "\n---\n" 
            for function in self
        ])

class Juxts(Append):
    wrapper, composition = map(staticmethod, (tuple, Juxtapose))


# In[6]:

for name, function in zip(__all__, functions):
    locals().update({name.capitalize(): type(name.capitalize(), (Pipes,), {'wrapper': staticmethod(function)})})

__all__ += ['pipes', 'juxts']

for fidget in __all__:
    callable = locals()[fidget.capitalize()]
    locals()[fidget] = type('_{}_'.format(fidget.capitalize()), (callable,), {})()
    locals()[fidget].function = Compose([callable])

for op, func in (('matmul', 'groupby'), ('truediv', 'map'), ('floordiv', 'filter'), ('mod', 'reduce')):
    setattr(Pipes, _attribute_('', op), property(Compose(attrgetter(func))))
Pipes.__div__  = Pipes.__truediv__ 


# In[7]:

def fallback(attr):
    def fallback(right, left):
        return getattr(Pipes()[left], attr)(Pipes()[right])
    return wraps(getattr(Pipes, attr))(fallback)


# In[8]:

for attr in ['add', 'sub', 'mul', 'matmul','div', 'truediv', 'floordiv', 'mod', 'lshift', 'rshift', 'and', 'xor', 'or', 'pow']:
    setattr(Pipes, _attribute_('i', attr), getattr(Pipes, _attribute_('', attr)))
    setattr(Pipes, _attribute_('r', attr), fallback(_attribute_('', attr)))


# In[9]:

Pipes.attributes['fidget'] = {
    f.__name__: composed(f) for f in (groupby, reduce, filter, map)}
Pipes.attributes['fidget'].update({
    key: getattr(Pipes, _attribute_('', value)) 
    for key, value in [['call']*2, ['do', 'lshift'], ['pipe',  'getitem'], ['ifthen','xor'], ['step', 'and'], ['ifnot', 'or']]})

