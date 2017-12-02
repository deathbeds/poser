
# coding: utf-8

from itertools import zip_longest
from toolz import cons, concatv
from inspect import getdoc
__all__ = 'partial', 'flipped'


class partial(__import__('functools').partial):
    """partial overloads functools.partial to provide equality and documentation.
    
    >>> f = partial(range, 10) 
    >>> assert f == partial(range, 10)
    >>> assert partial(range, 10, 20) == partial(range, 10)
    """
    def __eq__(self, other):
        return isinstance(other, partial) and all(
            (a is b) or (a == b) for a, b in zip_longest(*(cons(_.func, _.args) for _ in [self, other])))
    
    @property
    def __doc__(self): return getdoc(self.func)

class partial_attribute(partial):
    """partial_attribute is a partial for MethodType attributes.
    
    >>> f = partial_attribute(str.replace, 'x', 'y')
    >>> assert f('xy') == 'yy'
    """
    def __call__(self, object):
        if callable(self.func):
            return self.func(object, *self.args, **self.keywords) 
        return self.func


class flipped(partial):
    """partial overloads functools.partial to provide equality and documentation.
    
    >>> assert flipped(range, 10)(20) == range(20, 20)
    """
    def __call__(self, *args, **kwargs):
        return self.func(*reversed(self.args+args), **self.keywords)


if __name__ == '__main__':
    print(__import__('doctest').testmod(verbose=False))
    get_ipython().system('jupyter nbconvert --to python --TemplateExporter.exclude_input_prompt=True partials.ipynb')

