# coding: utf-8

try:
    from .callables import call
    from .classes import Compose
    from .calls import *
except:
    from callables import call
    from classes import Compose
    from calls import *

__all__ = [
    'calls', 'does', 'filters', 'flips', 'maps', 'stars', 'reduces', 'groups'
]  # noqa: F822

for name in __all__:
    func = locals()[name.capitalize()]
    locals()[name] = type('_{}_'.format(func.__name__), (func, ),
                          {})(function=Compose([func]))

del func, name
__all__ += ['call']

Calls.namespaces['toolz'] = vars(__import__('toolz'))
