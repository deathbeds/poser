
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
    from .callables import flipped, do, starred, Compose, Partial, Juxtapose, calls
except Exception as e:
    from callables import flipped, do, starred, Compose, Partial, Juxtapose, calls
    
from functools import wraps

from operator import attrgetter
from toolz.curried import groupby, compose, merge, reduce, filter, map, partial, merge, isiterable, flip, complement, interpose
from collections import OrderedDict
from functools import wraps
from inspect import signature
from importlib import import_module
from six import PY3
_attribute_ = "__{}{}__".format
_isinstance_ = flip(isinstance)


# In[21]:


__all__ = ['flips', 'stars', 'does', 'maps', 'filters', 'groups', 'reduces']
functions = (flipped, starred, do, map, filter, groupby, reduce)


# In[22]:


def composed(callable):
    def composed(*args, **kwargs):
        args = (calls(args[0]), *args[1:])
        return callable(*args, **kwargs)
    return wraps(callable)(composed)


# In[23]:


def curried(callable):
    def curried(*args):
        function = callable
        for arg in args:
            function = function(arg)
        return function
    return wraps(callable)(curried)


# In[24]:


class Imports(OrderedDict):
    def __missing__(self, key):
        self[key] = vars(import_module(key))
        return self
    
    def __setitem__(self, key, value):
        if isinstance(value, str):
            value = vars(import_module(value))
        return super(Imports, self).__setitem__(key, value) or self


# In[25]:


class Attributes(object):
    _attributes = Imports()

    def __getattr__(self, attribute):
        for section in reversed(self._attributes.values()):
            if attribute in section:
                callable = section[attribute]
                doc = callable.__doc__
                try:
                    sig = signature(callable)
                except:
                    sig=None

                if callable in merge(map(vars, type(self).__mro__)).values():
                    callable = partial(callable, self)
                else:
                    callable = partial(self.__getitem__, callable)
                
                PY3 and setattr(callable, '__doc__', doc)
                sig and setattr(callable, '__signature__', sig)
                
                return callable
        raise AttributeError("No attribute {}".format(attribute))

    def __dir__(self):
        return list(super(Attributes, self).__dir__()) + list(
            merge(self._attributes.values()).keys())


# In[26]:


Attributes._attributes['itertools']
Attributes._attributes['collections']
Attributes._attributes['builtins'] = 'six.moves.builtins'
Attributes._attributes['operator'] = {
    key: curried(value) if key in ['attrgetter', 'methodcaller', 'itemgetter'] else flipped(value)
    for key, value in vars(__import__('operator')).items() if key[0].islower()
}
Attributes._attributes['toolz'] = {
    key: composed(value) if any(map(key.endswith, ('filter', 'map'))) else value
    for key, value in vars(__import__('toolz')).items() if key[0].islower()
}


# In[27]:


class Operators(object):    
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
            
    def __round__(self, n):
        self.function.function = list(interpose(n, self.function.function))
        return self


# In[28]:


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


# In[29]:


class Pipes(Factory, Operators, Attributes): 
    @property
    def __doc__(self):
        return '\n'.join([
            (getattr(function, 'func', function).__doc__ or "") + "\n---\n" 
            for function in self])

class Juxts(Factory):
    _wrapper, _composition = map(staticmethod, (tuple, Juxtapose))


# In[2]:


for name, function in zip(__all__, functions):
    locals().update({name.capitalize(): type(name.capitalize(), (Pipes,), {'_wrapper': staticmethod(function)})})

__all__ += ['pipes', 'juxts']

for fidget in __all__:
    callable = locals()[fidget.capitalize()]
    locals()[fidget] = type('_{}_'.format(fidget.capitalize()), (callable,), {})()
    locals()[fidget].function = Compose([callable])
    
__all__ += ['a', 'an', 'the', 'then']; a = an = the = then = pipes

for op, func in (('matmul', 'groupby'), ('truediv', 'map'), ('floordiv', 'filter'), ('mod', 'reduce')):
    setattr(Pipes, _attribute_('', op), property(Compose(attrgetter(func))))
Pipes.__div__  = Pipes.__truediv__ 


# In[31]:


def fallback(attr):
    def fallback(right, left):
        return getattr(Pipes()[left], attr)(Pipes()[right])
    return wraps(getattr(Pipes, attr))(fallback)


# In[32]:


for attr in ['add', 'sub', 'mul', 'matmul','div', 'truediv', 'floordiv', 'mod', 'lshift', 'rshift', 'and', 'xor', 'or', 'pow']:
    setattr(Pipes, _attribute_('i', attr), getattr(Pipes, _attribute_('', attr)))
    setattr(Pipes, _attribute_('r', attr), fallback(_attribute_('', attr)))


# In[33]:


Pipes._attributes['fidget'] = {
    f.__name__: composed(f) for f in (groupby, reduce, filter, map)}
Pipes._attributes['fidget'].update({
    key: getattr(Pipes, _attribute_('', value)) 
    for key, value in [['call']*2, ['do', 'lshift'], ['pipe',  'getitem'], ['ifthen','xor'], ['step', 'and'], ['ifnot', 'or']]})


# In[34]:


(pipes.range(2) @ (lambda x: x//2))(10)


# In[ ]:




