
# coding: utf-8

try:
    from .composites import the, partial, composite, flip, star, factory
    from .conditions import ifthen
    from .partials import partial_attribute
except BaseException:
    from composites import the, partial, composite, flip, star, factory
    from conditions import ifthen
    from partials import partial_attribute
import operator
from functools import partialmethod
dunder = "__{}__".format

__all__ = 'canonical', 'x'


class canonical(object):
    """Canonical function compositions using operator.

    >>> x = y = z = canonical
    >>> f = 10 < x() < 100
    >>> assert callable(f) and f(50) and not f(0) and not f(200)
    >>> assert (x() + 10 * 100)(20) == 20 + 10 * 100
    """

    def __init__(self, object=None):
        self.__wrapped__ = object or composite()

    @property
    def __call__(self): return self.__wrapped__.__call__

    def __getattr__(self, object):
        if object == '_ipython_canary_method_should_not_exist_':
            return self
        self.__wrapped__ = self.__wrapped__[partial_attribute(getattr, object)]
        return self

    def __repr__(self): return repr(self.__wrapped__)


class factory(object):
    """A factory for canonical compositions.

    >>> x = factory()
    >>> assert 5 < x
    >>> assert x + 5
    >>> assert 5 < x() < 100
    """

    def __getitem__(self, item): return self()[item]

    def __getattr__(self, item): return getattr(self(), item)

    def __call__(self): return canonical()


def __attr__(self, attr, *object):
    if isinstance(self, factory):
        self = canonical()
    attr = getattr(operator, attr)
    object = object[0] if object else None
    self.__wrapped__ = (composite()[self.__wrapped__, object][star[attr]] if object and callable(
        object) else self.__wrapped__[partial_attribute(attr, object) if object else attr])
    return self


def __rattr__(self, attr, object):
    if isinstance(self, factory):
        self = canonical()
    attr = getattr(operator, attr)
    self.__wrapped__ = (composite()[object, self.__wrapped__][star[attr]] if callable(
        object) else composite()[partial(attr, object)][self.__wrapped__ or slice(None)])
    return self


def __battr__(self, attr, object):
    if isinstance(self, factory):
        self = canonical()
    attr = getattr(operator, attr)
    if callable(object):
        self.__wrapped__ = the[self.__wrapped__, object][star[attr]]
    else:
        object = partial_attribute(attr, object)
        self.__wrapped__ = ifthen(self.__wrapped__)[
            object] if self.__wrapped__ else the[object]
    return self


for cls in [canonical, factory]:
    for attr in [
        'add',
        'sub',
        'mul',
        'floordiv',
        'truediv',
        'mod',
        'matmul',
        'and',
        'or',
        'pow',
        'lshift',
            'rshift']:
        setattr(cls, dunder(attr), partialmethod(__attr__, attr))
        setattr(cls, "__i{}__".format(attr), partialmethod(__attr__, attr))
        setattr(cls, "__r{}__".format(attr), partialmethod(__rattr__, attr))

    for attr in ['abs', 'neg', 'pos', 'invert', 'getitem', 'delitem']:
        setattr(cls, dunder(attr), partialmethod(__attr__, attr))

    for attr in ['lt', 'le', 'gt', 'ge', 'eq']:
        setattr(cls, dunder(attr), partialmethod(__battr__, attr))


x = factory()


if __name__ == '__main__':
    print(__import__('doctest').testmod())
    get_ipython().system(
        'jupyter nbconvert --to python --TemplateExporter.exclude_input_prompt=True canonical.ipynb')
