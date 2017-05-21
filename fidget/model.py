# coding: utf-8


try:
    from .objects import  Functions, Compose, Calls, Composer
    from .callables import flipped, do, step, starred, excepts, ifnot, ifthen
except Exception as e:
    from objects import  Functions, Compose, Calls, Composer
    from callables import flipped, do, step, starred, excepts, ifnot, ifthen
    
from collections import OrderedDict
from functools import partial, wraps
from six import PY3
from operator import attrgetter
from toolz.curried import (isiterable, flip, complement, interpose, groupby, merge, reduce, filter, map)
_attribute_ = "__{}{}__".format

__all__ = ['flips', 'stars', 'does', 'maps', 'filters', 'groups', 'reduces']
functions = (flipped, starred, do, map, filter, groupby, reduce)

class Namespaces(Calls):
    namespaces = OrderedDict({'fidget': {}})
    def __getattr__(self, attr):
        for namespace in self.namespaces.values():
            if attr in namespace:
                callable = namespace[attr]
                doc = callable.__doc__
                if callable in type(self).__dict__.values():
                    callable = partial(callable, self)
                else:
                    callable = partial(self.__getitem__, callable)
                return PY3 and setattr(callable, '__doc__',  doc) or callable
        raise AttributeError("No attribute {}".format(attr))
        
    def __dir__(self):
        return list(super(Namespaces, self).__dir__()) + list(merge(self.namespaces.values()).keys())

class Models(Namespaces):    
    def __xor__(self, object):
        self, _isinstance = self[:], flip(isinstance)  # noqa: F823
        if not isiterable(object) and isinstance(object, type):
            object = (object,)
        if isiterable(object):
            if all(map(_isinstance(type), object)) and all(map(flip(issubclass)(BaseException), object)):
                self.function = Compose(excepts(object, self.function))
                return self
            
            if all(map(_isinstance(BaseException), object)):
                object = tuple(map(type, object))
                
            if all(map(_isinstance(type), object)):
                object = _isinstance(object)

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
    
    __invert__, __pow__ = Functions.__reversed__, __xor__
    __mul__ = __add__ = __rshift__ = __sub__ = Composer.__getitem__

for name, function in zip(__all__, functions):
    locals().update({name.capitalize(): type(name, (Models,), {'_decorate_': staticmethod(function)})})

__all__ += ['models']

for fidget in __all__:
    callable = locals()[fidget.capitalize()]
    locals()[fidget] = type('_{}_'.format(fidget.capitalize()), (callable,), {})(function=Compose([callable]))

for op, func in (('matmul', 'groupby'), ('truediv', 'map'), ('floordiv', 'filter'), ('mod', 'reduce')):
    setattr(Models, _attribute_('', op), property(Compose(attrgetter(func))))
Models.__div__  = Models.__truediv__

def fallback(attr):
    def fallback(right, left):
        right = right[:]
        return getattr(type(right)()[left], attr)(right)
    return wraps(getattr(Models, attr))(fallback)

for attr in ['add', 'sub', 'mul', 'matmul','div', 'truediv', 'floordiv', 'mod', 'lshift', 'rshift', 'and', 'xor', 'or', 'pow']:
    setattr(Models, _attribute_('i', attr), getattr(Models, _attribute_('', attr)))
    setattr(Models, _attribute_('r', attr), fallback(_attribute_('', attr)))
