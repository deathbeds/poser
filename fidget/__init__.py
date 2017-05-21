# coding: utf-8

try:
    from .callables import call
    from . import model
    from .model import *
    from . import namespaces
except:
    from callables import call
    import model
    from model import *
    import namespaces
__all__ = model.__all__

__all__ += ['call']

try:
    from IPython import get_ipython

    class Magic(model.Model):
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
