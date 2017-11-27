
# coding: utf-8

# Append symbollic operations to functin compositions

try:
    from .composites import composite, do, excepts, factory
    from .conditions import ifthen, instance, ifnot
    from .attributes import *
except:
    from composites import composite, do, excepts, factory
    from conditions import ifthen, instance, ifnot
    from attributes import *
    
__all__ = tuple()


from functools import partialmethod
from operator import not_
dunder = '__{}__'.format


factory.__mul__ =factory.__add__ = factory.__rshift__ = factory.__sub__ = factory.__getitem__
composite.__mul__ =composite.__add__ = composite.__rshift__ = composite.__sub__ = composite.__getitem__


def __lshift__(self, object):          
    return self[do[object]]
def __pow__(self, object=slice(None)):
    """
    >>> f = a**int*range
    >>> a[10, '10'].map(f).list()()
    [range(0, 10), False]

    A dictionary sets the function attributes.

    >>> assert (a**{'start': int, 'returns': range}*range).__annotations__
    """
    self = self[:]
    if isinstance(object, str):
        return setattr(self, '__doc__', object) or self
    if isinstance(object, dict):
        return setattr(self, '_annotations_', object) or self
    return instance(object)[self]

def __and__(self, object=slice(None)):        
    """append an ifthen statement to the composite

    >>> (a&range)(0), (a&range)(10)
    (0, range(0, 10))
    """
    return ifthen(self[:])[object]
def __or__(self, object=slice(None)):  
    """append an ifnot statement to the composite

    >>> (a|range)(0), (a|range)(10)
    (range(0, 0), 10)
    """
    return ifnot(self[:])[object] # There is no reasonable way to make this an attribute?

def __xor__(self: 'λ', object: (slice, Exception)=slice(None)) -> 'λ':             
    """append an exception to the composite

    >>> (a.str.upper()^TypeError)(10)
    TypeError("partial_attribute(<method 'upper' of 'str' objects>)\\ndescriptor 'upper' requires a 'str' object but received a 'int'",)
    """
    return excepts(object)[self[:]]

composite.__lshift__ = __lshift__
composite.__pow__ = __pow__
composite.__and__ = __and__
composite.__or__ = __or__
composite.__xor__ = __xor__


composite.__truediv__  = composite.map
composite.__floordiv__ = composite.filter
composite.__matmul__   = composite.groupby
composite.__mod__      = composite.reduce
composite.__pos__ = partialmethod(composite.__getitem__, bool)
composite.__neg__ = partialmethod(composite.__getitem__, not_)
composite.__invert__ = composite.__reversed__    


def right_attr(right, attr, left): 
    if isinstance(left, factory): left = left[:]
    return object.__getattribute__(composite()[left], attr)(right[:])

def op_attr(self, attr, value): 
    return object.__getattribute__(self[:], attr)(value)
        
[setattr(factory, dunder(attr), getattr(composite, dunder(attr))) or
 setattr(factory, dunder('r'+attr), partialmethod(right_attr, dunder(attr)))
 for attr in ['and', 'or', 'xor', 'truediv', 'floordiv', 'matmul', 'mod', 'lshift', 'pow']]

[setattr(object, dunder('i'+attr), partialmethod(op_attr, dunder(attr))) or
 setattr(object, dunder('r'+attr), partialmethod(right_attr, dunder(attr)))
 for attr in ['mul', 'add', 'rshift' ,'sub', 'and', 'or', 'xor', 'truediv', 'floordiv', 'matmul', 'mod', 'lshift', 'pow']
 for object in [composite, factory]]

[setattr(object, key, getattr(object, dunder(other))) for key, other in zip(('do', 'excepts', 'instance'), ('lshift', 'xor', 'pow')) for object in [factory, composite]];


if __name__ == '__main__':
#     print(__import__('doctest').testmod(verbose=False))
    get_ipython().system('jupyter nbconvert --to python --TemplateExporter.exclude_input_prompt=True operations.ipynb')
    get_ipython().system('flake8 operations.py')

