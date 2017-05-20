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

try:
    from IPython import get_ipython

    class Magic(Calls):
        def _decorate_(self, function):
            return get_ipython().register_magic_function(
                calls[does[stars.second()[function.function]]][None],
                magic_kind='cell',
                magic_name=self.args[0])

    __all__.append('Magic')
except:
    pass


def load_ipython_extension(ip):
    """%%fidget magic that displays code cells as markdown, then runs the cell.
    """
    from IPython import display
    Magic('fidget')[does[display.Markdown][display.display], ip.run_cell]()
