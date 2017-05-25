# coding: utf-8

# In[1]:

try:
    from .callables import flipped
    from .objects import dispatch
except:
    from callables import flipped
    from objects import dispatch

from collections import OrderedDict
from functools import wraps
from inspect import signature
from toolz.curried import map, partial, merge
from six import PY3
_attribute_ = "__{}{}__".format

# In[14]:


def composed(callable):
    def composed(*args, **kwargs):
        args = (dispatch(args[0]), *args[1:])
        return callable(*args, **kwargs)

    return wraps(callable)(composed)


# In[6]:


def curried(callable):
    def curried(*args):
        function = callable
        for arg in args:
            function = function(arg)
        return function

    return wraps(callable)(curried)


# In[7]:


class Namespaces(object):
    namespaces = OrderedDict()

    def __getattr__(self, attr):
        for namespace in reversed(self.namespaces.values()):
            if attr in namespace:
                callable = namespace[attr]
                doc = callable.__doc__
                try:
                    sig = signature(callable)
                except:
                    sig = None

                if callable in merge(map(vars, type(self).__mro__)).values():
                    callable = partial(callable, self)
                else:
                    callable = partial(self.__getitem__, callable)

                PY3 and setattr(callable, '__doc__', doc)
                sig and setattr(callable, '__signature__', sig)

                return callable
        raise AttributeError("No attribute {}".format(attr))

    def __dir__(self):
        return list(super(Namespaces, self).__dir__()) + list(
            merge(self.namespaces.values()).keys())


# In[8]:

Namespaces.namespaces['itertools'] = vars(__import__('itertools'))
Namespaces.namespaces['collections'] = vars(__import__('collections'))
Namespaces.namespaces['builtins'] = vars(
    __import__('builtins', fromlist=['six.moves']))
Namespaces.namespaces['operator'] = {
    key: curried(value)
    if key in ['attrgetter', 'methodcaller', 'itemgetter'] else flipped(value)
    for key, value in vars(__import__('operator')).items() if key[0].islower()
}
Namespaces.namespaces['toolz'] = {
    key: composed(value)
    if any(map(key.endswith, ('filter', 'map'))) else value
    for key, value in vars(__import__('toolz')).items() if key[0].islower()
}
