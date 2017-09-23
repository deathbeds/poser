
# coding: utf-8

# In[1]:


try:

    from .callables import flipped, step, excepts, ifnot, ifthen, Compose, _dispatch
except Exception as e:
    from callables import flipped, step, excepts, ifnot, ifthen, Compose, _dispatch
    
from collections import OrderedDict
from functools import wraps
from inspect import signature
from importlib import import_module
from toolz.curried import map, partial, merge, isiterable, flip, complement, interpose
from six import PY3
_attribute_ = "__{}{}__".format
_isinstance_ = flip(isinstance)


# In[20]:


def composed(callable):
    def composed(*args, **kwargs):
        args = (_dispatch(args[0]), *args[1:])
        return callable(*args, **kwargs)
    return wraps(callable)(composed)


# In[21]:


def curried(callable):
    def curried(*args):
        function = callable
        for arg in args:
            function = function(arg)
        return function
    return wraps(callable)(curried)


# In[22]:


class Imports(OrderedDict):
    def __missing__(self, key):
        self[key] = vars(import_module(key))
        return self
    
    def __setitem__(self, key, value):
        if isinstance(value, str):
            value = vars(import_module(value))
        return super(Imports, self).__setitem__(key, value) or self


# In[23]:


class Attributes(object):
    attributes = Imports()

    def __getattr__(self, attribute):
        for section in reversed(self.attributes.values()):
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
            merge(self.attributes.values()).keys())


# In[24]:


Attributes.attributes['itertools']
Attributes.attributes['collections']
Attributes.attributes['builtins'] = 'six.moves.builtins'
Attributes.attributes['operator'] = {
    key: curried(value) if key in ['attrgetter', 'methodcaller', 'itemgetter'] else flipped(value)
    for key, value in vars(__import__('operator')).items() if key[0].islower()
}
Attributes.attributes['toolz'] = {
    key: composed(value) if any(map(key.endswith, ('filter', 'map'))) else value
    for key, value in vars(__import__('toolz')).items() if key[0].islower()
}


# In[25]:


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

