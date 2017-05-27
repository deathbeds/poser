# coding: utf-8

# In[1]:

try:
    from . import model
    from .model import *
except:
    import model
    from model import *
__all__ = model.__all__

# In[2]:

try:
    from IPython import get_ipython

    class Magic(model.Models):
        def _decorate_(self, function):
            return get_ipython().register_magic_function(
                models[does[stars.second()[function.function]]][None],
                magic_kind='cell',
                magic_name=self.args[0])

    __all__.append('Magic')
except:
    pass

# In[3]:


def load_ipython_extension(ip):
    """%%fidget magic that displays code cells as markdown, then runs the cell.
    """
    from IPython import display
    Magic('fidget')[does[display.Markdown][display.display], ip.run_cell]()
