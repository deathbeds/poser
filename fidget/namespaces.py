# coding: utf-8

try:
    from .callables import flipped
    from .objects import Compose
    from .model import Models, _attribute_
except:
    from callables import flipped
    from objects import Compose
    from model import Models, _attribute_

from functools import wraps
from toolz.curried import groupby, reduce, filter, map


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


Models.namespaces['fidget'].update(
    {f.__name__: composed(f)
     for f in (groupby, reduce, filter, map)})
Models.namespaces['fidget'].update({
    key: getattr(Models, _attribute_('', value))
    for key, value in [['call'] * 2, ['do', 'lshift'], ['pipe', 'getitem'],
                       ['ifthen', 'xor'], ['step', 'and'], ['ifnot', 'or']]
})
Models.namespaces['toolz'] = {
    key: composed(value)
    if any(map(key.endswith, ('filter', 'map'))) else value
    for key, value in vars(__import__('toolz')).items() if key[0].islower()
}
Models.namespaces['itertools'] = vars(__import__('itertools'))
Models.namespaces['operator'] = {
    key: curried(value)
    if key in ['attrgetter', 'methodcaller', 'itemgetter'] else flipped(value)
    for key, value in vars(__import__('operator')).items() if key[0].islower()
}
Models.namespaces['builtins'] = vars(
    __import__('builtins', fromlist=['six.moves']))
Models.namespaces['collections'] = vars(__import__('collections'))
