
# coding: utf-8

# An extensible attribute system for the composites.

try:
    from .composites import composite, flip, factory
    from .partials import partial, partial_attribute
except:
    from composites import composite, flip, factory
    from partials import partial, partial_attribute


from functools import partialmethod, wraps, WRAPPER_ASSIGNMENTS
from inspect import signature, getdoc
from operator import attrgetter
from toolz.curried import identity, concat, concatv, keymap
from toolz import map, groupby, filter, reduce
import sys
dunder = '__{}__'.format
__all__ = 'shortcuts',


class wrapped(object):
    """attribute uses wrapped to chain nested atrributes.
    This would not need to exist if it weren'ty for IPython
    
    >>> f = composite()
    >>> w = wrapped(f) 
    >>> assert w.range.__doc__ == range.__doc__"""
    def __init__(self, composite): 
        self.composite = composite
        for key in WRAPPER_ASSIGNMENTS: 
            hasattr(self.composite.object, key) and setattr(self, key, getattr(self.composite.object, key))
        try: self.__signature__ = signature(self.composite.object)
        except: pass
            
    @property
    def __call__(self): return wraps(self.composite.object)(self.composite)
    def __getattr__(self, attr): return self.composite.__getattr__(attr)
    def __repr__(self): return repr(self.composite.object)


class attribute(object):
    """a class to assist in function composition with attributes.  This class
    goes to extra lengths to improve complextion of attributes and arguments in
    the IPython context. It provides composites will full access to sys.modules
    in a chainable API context.
    
    >>> assert composite().range() == composite()[range]
    """
    def __init__(self, composite=None, object=None, parent=None):
        self.object, self.composite, self.parent = object, composite, parent
        
    def __iter__(self):
        if self.object: 
            yield self.object
        else:
            for object in self.shortcuts: 
                yield type(object) is str and sys.modules.get(object, __import__(object)) or object
            yield sys.modules

    def __getitem__(self, item):
        objects = list(self)
        if len(objects) > 1:
            for object in objects:
                if getattr(object, dunder('name'), "") == item: return object
        for object in objects:
            dict = getattr(object, dunder('dict'), object)
            if item in dict:  return dict[item]
        if item in sys.modules: return sys.modules[item]
        raise AttributeError(item)

    def __dir__(self):
        return list(concat(getattr(object, dunder('dict'), object).keys() for object in self))

    def __getattr__(self, item): 
        new = type(self)(self.composite, self[item], self.object)
        return wrapped(new) if callable(new.object)  else new

    def __repr__(self): return repr(self.object or list(self))
    
    def __call__(self, *args, **kwargs):
        object = self.object
        if callable(object):
            for decorator, values in self.decorators.items():
                if object in values: 
                    new = decorator(object)
                    object = object(*args, **kwargs) if new is object else partial(new, *args, **kwargs)
                    break
            else:
                if isinstance(self.parent, type):
                    object = partial_attribute(object, *args, **kwargs)
                elif args or kwargs:
                    object = partial(object, *args, **kwargs)
        return (composite() if self.composite is None else self.composite)[object]

shortcuts = attribute.shortcuts = list(['statistics', 'toolz', 'requests', 'builtins', 'json', 'pickle', 'io', 
        'collections', 'itertools', 'functools', 'pathlib', 'importlib', 'inspect', 'operator'])

# decorators for the operators.
import operator, fnmatch
# some of these cases fail, but the main operators work.
attribute.decorators = keymap([flip, identity].__getitem__, groupby(
    attrgetter('itemgetter', 'attrgetter', 'methodcaller')(operator).__contains__, 
    filter(callable, vars(__import__('operator')).values())
))
attribute.shortcuts.insert(0, {'fnmatch': fnmatch.fnmatch})
attribute.decorators[flip].append(fnmatch.fnmatch)


def __dir__(self): return list(self.__dict__.keys()) + dir(self.attribute())
composite.__dir__ = __dir__


def __getattr__(self, attr, *args, **kwargs):
    """extensible attribute method relying on compose.attributer

    >>> assert composite().range().len() == composite().builtins.range().builtins.len() == composite()[range].len()
    """
    if callable(attr): 
        args = (arg if callable(arg) else composite()[arg] for arg in args)
        return self[:][partial(attr, *args, **kwargs)]
    return getattr(self.attribute(self[:]), attr)

composite.attribute = staticmethod(attribute)
composite.__getattr__ = __getattr__
composite.map = factory.map = partialmethod(__getattr__, map)
composite.filter = factory.filter = partialmethod(__getattr__, filter)
composite.reduce = factory.reduce = partialmethod(__getattr__, reduce)
composite.groupby = factory.groupby = partialmethod(__getattr__, groupby)


def __magic__(self, name, *, ip=None):
    ip, function = ip or __import__('IPython').get_ipython(), self.copy()
    @wraps(function)
    def magic_wrapper(line, cell=None):
        return function('\n'.join(filter(bool, [line, cell])))
    ip.register_magic_function(magic_wrapper, 'cell', name)

composite.__magic__ = __magic__


if __name__ == '__main__':
    print(__import__('doctest').testmod(verbose=False))
    get_ipython().system('jupyter nbconvert --to python --TemplateExporter.exclude_input_prompt=True attributes.ipynb')

