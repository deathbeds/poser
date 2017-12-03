
# coding: utf-8

# Conditional composites

try:
    from .partials import partial
    from .composites import composite
except:
    from partials import partial
    from composites import composite
from collections import UserList, deque
from inspect import signature, getdoc
import toolz
from toolz.curried import isiterable, identity, last, flip
from copy import copy
dunder = '__{}__'.format
__all__ = 'ifthen', 'ifnot', 'instance'


class condition(composite):
    __slots__ = 'condition', 'data'
    def __init__(self, condition=bool, data=None):
        setattr(self, 'condition', condition) or super().__init__(data)
        
class ifthen(condition):
    """The composition evaluates if condition is True.
    
    >>> f = ifthen(bool)[range]
    >>> f(0), f(10)
    (False, range(0, 10))
    """
    def __call__(self, *args, **kwargs):
        return self.condition(*args, **kwargs) and super(ifthen, self).__call__(*args, **kwargs)

class ifnot(condition):
    """The composition evaluates if condition is False.
    
    >>> f = ifnot(bool)[range]
    >>> f(0), f(10)
    (range(0, 0), True)
    """
    def __call__(self, *args, **kwargs):
        return self.condition(*args, **kwargs) or super(ifnot, self).__call__(*args, **kwargs)

class instance(ifthen):
    """The composition evaluates if the arguments are an instance of condition.
    
    >> a[instance(str)[str.upper], instance(int)[range]](10)
    (False, range(0, 10))
    """
    def __init__(self, condition=None, data=None):        
        if isinstance(condition, type): condition = condition,            
        if isinstance(condition, tuple): condition = partial(flip(isinstance), condition)
        super().__init__(condition, data)


if __name__ == '__main__':
    print(__import__('doctest').testmod(verbose=False))
    get_ipython().system('jupyter nbconvert --to python --TemplateExporter.exclude_input_prompt=True conditions.ipynb')
    get_ipython().system('flake8 conditions.py')

