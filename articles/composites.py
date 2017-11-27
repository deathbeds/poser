
# coding: utf-8

# # Composites
# 
# Composites are functional programming-style objects in Python.  Composites rely on `toolz` and Python builtins to compose higher-order functions. Effectively, composites are lists and dicts with call methods.

try:
    from .partials import partial
except:
    from partials import partial

__all__ = 'a', 'an', 'the', 'star', 'do', 'λ', 'flip', 'excepts', 'composite'

from collections import UserList, deque
from inspect import signature, getdoc
import toolz
from toolz.curried import isiterable, identity, concatv, last
from typing import Iterator, Any
from copy import copy
dunder = '__{}__'.format


def null(*args, **kwargs): 
    return args[0] if args else None


# CallableList subclasses `UserList` to manipulate compositions with a familar `api`; this approach provides total ordering.

class CallableList(UserList):
    """CallableList is a callable that chains a list of functions through a composite. 
    
    A CallableList returns a generator; each value corresponds to a function the list.
    
    >>> f = CallableList([range, len, type])
    >>> g = f(10)
    >>> next(g)
    range(0, 10)
    >>> next(g)
    10
    >>> next(g)
    <class 'int'>
    >>> assert list(f(10)) == [range(0, 10), 10, int]
    
    Callable sequence objects use abs to return the 
    value of the function.  
    
    >>> assert abs(f)(10) == int
    """
    def __iter__(self) -> Iterator: 
        yield from iter(self.data or [null])

    def __abs__(self) -> Any: 
        return toolz.compose(last, partial(self))
    
    def __call__(self, *args, **kwargs) -> Iterator:
        for object in self:
            args, kwargs = [object(*args, **kwargs)] if callable(object) else [object], dict()    
            yield null(*args, **kwargs)


# * Use a better error message when calling composites.
# * Attach attributes to copy, pickle, print, and other data model attributes.

class compose(CallableList):
    """compose adds attributes from the python data model to the callable list.
    
    compose objects are immutable lists that use [brackets] to append functions to a composition.
    
    >>> f = compose()[range]
    >>> f[type]
    compose:[<class 'range'>, <class 'type'>]

    All compose objects work as decorators. 
    >>> @compose
    ... def f(a): return range(a)
    >>> assert isinstance(f, compose) and abs(f)(10) == range(10) == f[0](10)
    
    compose objects can be copied and pickled.
    
    >>> from pickle import loads, dumps
    >>> assert loads(dumps(compose()[range][type])) == compose()[range][type]
    """
    __slots__ = 'data',   
    
    def __call__(self, *args, **kwargs) -> Iterator:
        callables, result = iter(self), super().__call__(*args, **kwargs)
        while True:
            try:
                callable = next(callables) 
                yield next(result)
            except Exception as e:
                if isinstance(e, StopIteration): break
                callable = repr(callable)
                if not any(line in callable for line in e.args[0].splitlines()):
                    e.args = callable+'\n'+e.args[0], *e.args[1:]
                raise e


    def __init__(self, data=None):
        super().__init__(data is not None and (not isiterable(data) or isinstance(data, str)) and [data] or data or list())
        self.__qualname__ = __name__ + '.' + type(self).__name__
    
    def __getitem__(self, object):
        return self.data[object] if isinstance(object, (int, slice)) else self.append(object)
            
        raise AttributeError(object)
    def append(self, object):
        return  self.data.append(object) or not self.data[0] and self.data.pop(0) or self        
    
    @property
    def __name__(self): return type(self).__name__
                
    @property
    def __signature__(self): return signature(self[0])
    def __hash__(self): return hash(tuple(self))
    def __bool__(self): return any(self.data)
    def __reversed__(self): 
        self.data = list(reversed(self.data))
        return self
    def __getstate__(self): return tuple(getattr(self, slot) for slot in self.__slots__)
    def __setstate__(self, state):
        for attr, value in zip(reversed(self.__slots__), reversed(state)): setattr(self, attr, value)
            
    def __copy__(self, memo=None):
        new = type(self.__name__, (type(self),), {'_annotations_': self._annotations_})()
        return new.__setstate__(tuple(map(copy, self.__getstate__()))) or new
    _annotations_ = None
    @property
    def __annotations__(self): 
        return self._annotations_ or self and getattr(self[0], dunder('annotations'), None) or {}

    def __repr__(self):
        other = len(self.__slots__) is 2 and repr(self.__getstate__()[1-self.__slots__.index('data')])
        return type(self).__name__.replace('composite', 'λ') +(
            '({})'.format(other or '').replace('()', ':')
        )+ super().__repr__()
    
    copy = __deepcopy__ = __copy__


# * A factory skips parenthesis when initializing a composition.

class factory(compose):
    """A factory of composition that works as a decorator.
    
    Create a factory
    >>> some = factory(compose)


    Supply partial arguments
    
    >> some(10)(20)
    10
    >> some(10)[range](20)
    range(10, 20)
    >> assert some(10)[range](20) == a.range(10)(20)
    """
    __slots__ = 'object', 'data', 'args', 'kwargs'
    def __init__(self, object, data=None, args=None, kwargs=None):
        super().__init__(data)
        self.object, self.args, self.kwargs = object, args, kwargs
        
    def __getitem__(self, attr):
        if isinstance(self.args, tuple) and  isinstance(self.kwargs, dict):
            attr = attr == slice(None) and abs(self) or partial(attr, *self.args, **self.kwargs)
        return self.object()[attr]
                
    def __getattr__(self, attr, *args, **kwargs):
        return self.object().__getattr__(attr, *args, **kwargs)
    
    def __call__(self, *args, **kwargs):
        return (
            next(concatv(self.args, args)) if isinstance(self.args, tuple) and  isinstance(self.kwargs, dict) 
            else factory(self.object, self.data, args, kwargs))    
    
    def __dir__(self): return dir(self.object())
    
    @property
    def __doc__(self): return getdoc(self.object)


# __composite__ is the core object. This class returns the executed function when called, rather than a generator.  The getitem method transforms non-callable iterable objects into juxtaposed functions.

class composite(compose):
    """An object for symbolically generating higher-order function compositions. Calling
    composite objects returns the evaluated function.
    
    abs is not required to evaluate a composition.
    >>> composite()[range]
    λ:[<class 'range'>]

    >>> composite()[range, type]
    λ:[juxt(<class 'tuple'>)[<class 'range'>, <class 'type'>]]
    
    Literal and symbolic attributes are append in attributes and operations.
    """
    def __call__(self, *args, **kwargs): 
        return deque(super().__call__(*args, **kwargs), maxlen=1).pop()
    
    def __abs__(self): return self.__call__
    
    def __getitem__(self, object):
        """Use brackets to append functions to the compose.
        >>> compose()[range][list]
        compose:[<class 'range'>, <class 'list'>]
        """        
        # from ipython prediction
        if object in (slice(None), getdoc): return self
        return super().__getitem__([identity, juxt][isiterable(object) and not isinstance(object, (str, compose)) and not callable(object)](object))


# ## Toolz inspired compositions
# 
# __juxt, flip, excepts, and do__ are composite versions of the `toolz` equivalents.
# 
#     from toolz import juxt, flip, excepts, do

# __juxt__ applies the same arguments to a list of functions.  The composite version adds the ability use mapping objects and control the return class.

class juxt(composite):
    """Juxtapose functions in python.  Juxtaposition applies the same input
    arguments to a list of functions.
    
    >>> juxt((range, type))(10)
    (range(0, 10), <class 'int'>)
    >>> juxt([range, type])(10)
    [range(0, 10), <class 'int'>]
    >>> juxt({range, type})(10)
    {range(0, 10), <class 'int'>}
    >>> juxt({'foo': range, 'bar': type})(10)  # doctest: +SKIP
    {'foo': range(0, 10), 'bar': <class 'int'>}
    """
    __slots__ = 'data', 'object'
    
    def __init__(self, data=None, object=None):
        if isiterable(data) and not isinstance(data, composite):
            object = object or type(data)
        self.object = object or tuple
        super().__init__(list(data.items()) if isinstance(data, dict) else list(data or list()))

    def __iter__(self):
        for callable in self.data:
            if not isinstance(callable, (str, compose)) and isiterable(callable):
                callable = juxt(callable)
            if not isinstance(callable, compose):
                callable = abs(compose(callable))
            yield callable
            
    def __call__(self, *args, **kwargs): 
        return self.object(callable(*args, **kwargs) for callable in self)


# __flip__ reverses the arguments.

class flip(composite):
    """Call a composite with the arguments reversed.
    
    >>> assert flip()[range](20, 10) == range(10, 20)
    """
    def __call__(self, *args, **kwargs):
        return super().__call__(*reversed(args), **kwargs)


# __excepts__ returns rather than raises an exception.

class excepts(composite):
    """A composition that returns exceptions as values.
    
    >>> excepts(TypeError)[str.upper](10)
    TypeError("<method 'upper' of 'str' objects>\\ndescriptor 'upper' requires a 'str' object but received a 'int'",)
    """
    __slots__ = 'exceptions', 'data'
    def __init__(self, exceptions=None, data=None):
        setattr(self, 'exceptions', exceptions) or super().__init__(data)
    
    def __call__(self, *args, **kwargs):
        try: return super(excepts, self).__call__(*args, **kwargs)
        except self.exceptions as e: return e


@factory
class do(composite):
    """Evaluate a function without modifying the input arguments.
    
    >>> assert not λ[print](10) and do()[print](10) is 10
    10
    10
    """
    
    def __call__(self, *args, **kwargs):
        super().__call__(*args, **kwargs)
        return null(*args)


@factory
class star(composite):
    """Supply iterables and dictionaries as starred arguments to a function.
    
    >>> def f(*args, **kwargs): return args, kwargs
    >>> star[f]([10, 20], {'foo': 'bar'})
    ((10, 20), {'foo': 'bar'})
    """
    def __call__(self, *inputs):
        args, kwargs = list(), dict()
        [kwargs.update(**input) if isinstance(input, dict) else args.extend(input) for input in inputs]
        return super().__call__(*args, **kwargs)


# __a__, __an__, __the__, and __λ__ are the main __articles__ used for function composition.  They seemed like uncommon namespace choices.

a = an = the = λ = factory(composite)


# Operations adds a bunch of attributes and symbols to compositions.

try:
    from . import operations
except:
    import operations


if __name__ == '__main__':
    print(__import__('doctest').testmod(verbose=False))
    get_ipython().system('jupyter nbconvert --to python --TemplateExporter.exclude_input_prompt=True composites.ipynb')
    get_ipython().system('autopep8 --in-place --aggressive --aggressive composites.py')

