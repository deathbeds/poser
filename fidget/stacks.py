
# coding: utf-8

# In[2]:


from toolz.curried import *
from functools import partialmethod

try:
    from .fidgets import *
    from .fidgets import Compose, Does, Groups, Reduces, Maps, Filters, Stars, Flips, Composes, calls
except:
    from fidgets import *
    from fidgets import Compose, Does, Groups, Reduces, Maps, Filters, Stars, Flips, Composes, calls

__all__ = 'a', 'an', 'the', 'then', 'stack'


# In[16]:


class Stacks(Composes): 
    """A stack of composite functions."""
    def push(self, callable=Composes):
        self.function[callable()]
        return self
        
    def __getitem__(self, item=None):
        if self._factory_:
             self = super().__getitem__()
        assert not self._factory_
        len(self) is 0 and self.push()
        self.function[-1][item]
        return self
    
    def __getattr__(self, attr):
        self = self[:]
        def _wrapped_attr(*args, **kwargs):
            self.function[-1].__getattr__(attr)(*args, **kwargs)
            return self
        return _wrapped_attr   
    
    def pop(self, index=-1):
        self.function = self.function[:-1]
        return self

    groups = partialmethod(push, Groups)
    filters = partialmethod(push, Filters)
    stars = partialmethod(push, Stars)    
    does = partialmethod(push, Does)    
    reduces = partialmethod(push, Reduces)
    maps = partialmethod(push, Maps)


# In[13]:


stacks = type('_Stacks_', (Stacks,), dict())()
stacks.function = Compose([Stacks])


# In[14]:


a = an = the = then= stacks

for article in __all__: setattr(Stacks, article, property(identity))

