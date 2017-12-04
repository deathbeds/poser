
# coding: utf-8

try:
    from .composites import the, partial, composite, flip
    from .conditions import ifthen
except:
    from composites import the, partial, composite, flip
    from conditions import ifthen
import operator
from functools import partialmethod
dunder = "__{}__".format

__all__ = 'canonical', 'x'


class canonical(object):
    """
    >>> x = y = z = canonical
    >>> f = 10 < x() < 100
    >>> assert callable(f) and f(50) and not f(0) and not f(200)
    >>> assert (x() + 10 * 100)(20) == 20 + 10 * 100
    """
    def __init__(self, object=None):
        self.composite = object or composite()
        
    @property
    def __call__(self): return self.composite.__call__

    def __getitem__(self, object): 
        self.composite = self.composite[object]
        return self
    
    def __getattr__(self, object): 
        self.composite = self.composite.__getattr__(object)
        return self

    def __repr__(self): return repr(self.composite)
    
x = canonical


def __attr__(self, attr, object):
    attr = getattr(operator, attr)
    self.composite = (
        composite()[self.composite, object][star[attr]]
        if callable(object) else self.composite[flip(object)[attr]])
    return self

def __rattr__(self, attr, object):
    attr = getattr(operator, attr)
    self.composite = (
        composite()[object, self.composite][star[attr]]
        if callable(object) else composite()[partial(attr, object)][self.composite or slice(None)])
    return self

def __battr__(self, attr, object):
    attr = getattr(operator, attr)
    if callable(object):
        self.composite = a[self.composite, object][star[attr]]
    else:
        object = flip(object)[attr]
        self.composite = ifthen(self.composite)[object] if self.composite else object
    return self


for attr in ['add', 'sub', 'mul', 'floordiv', 'truediv', 'mod', 'matmul']:
    setattr(canonical, dunder(attr), partialmethod(__attr__, attr))
    setattr(canonical, "__r{}__".format(attr), partialmethod(__rattr__, attr))

for attr in ['lt', 'le', 'gt', 'ge', 'eq']:
    setattr(canonical, dunder(attr), partialmethod(__battr__, attr))    


if __name__  == '__main__':
    print(__import__('doctest').testmod())
    get_ipython().system('jupyter nbconvert --to python --TemplateExporter.exclude_input_prompt=True canonical.ipynb')

