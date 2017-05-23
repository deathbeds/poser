# coding: utf-8

try:
    from .callables import flipped
    from .objects import Compose
except:
    from callables import flipped
    from objects import Compose

from collections import OrderedDict
from functools import wraps
from toolz.curried import map, partial, merge
from six import PY3
_attribute_ = "__{}{}__".format


def composed(callable):
    def composed(*args, **kwargs):
        args = (Compose._dispatch_(args[0]), *args[1:])
        return callable(*args, **kwargs)

    return wraps(callable)(composed)


def curried(callable):
    def curried(*args):
        function = callable
        for arg in args:
            function = function(arg)
        return function

    return wraps(callable)(curried)


class Namespaces(object):
    namespaces = OrderedDict({'fidget': {}})

    def __getattr__(self, attr):
        for namespace in self.namespaces.values():
            if attr in namespace:
                callable = namespace[attr]
                doc = callable.__doc__
                if callable in merge(map(vars, type(self).__mro__)).values():
                    callable = partial(callable, self)
                else:
                    callable = partial(self.__getitem__, callable)
                return PY3 and setattr(callable, '__doc__', doc) or callable
        raise AttributeError("No attribute {}".format(attr))

    def __dir__(self):
        return list(super(Namespaces, self).__dir__()) + list(
            merge(self.namespaces.values()).keys())


Namespaces.namespaces['toolz'] = {
    key: composed(value)
    if any(map(key.endswith, ('filter', 'map'))) else value
    for key, value in vars(__import__('toolz')).items() if key[0].islower()
}
Namespaces.namespaces['itertools'] = vars(__import__('itertools'))
Namespaces.namespaces['operator'] = {
    key: curried(value)
    if key in ['attrgetter', 'methodcaller', 'itemgetter'] else flipped(value)
    for key, value in vars(__import__('operator')).items() if key[0].islower()
}
Namespaces.namespaces['builtins'] = vars(
    __import__('builtins', fromlist=['six.moves']))
Namespaces.namespaces['collections'] = vars(__import__('collections'))
