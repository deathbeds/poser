
# coding: utf-8

# In[6]:

from fidget import *
from fidget import _x, x_, _comp_, _pmoc_
from pickle import dumps


# In[7]:

def test_equivalence():
    assert _comp_ is _x
    assert _pmoc_ is x_


# In[3]:

def test_some_chain():
    assert _x[range, list, len](10) == 10
    assert _x[:][range][list][len](10) == 10
    assert _x(10)[range][list][len]() == 10
    assert _x(10)[range][list][len] >> identity == 10
    assert _x ** 10 >> range >> list >> len >> identity == 10
    assert dumps(_x ** 10 >> range >> list >> len)


# In[ ]:

def test_pickle():
    assert dumps(_x ** 10 >> range >> list >> len)
    assert dumps(_x ** 10 >> range >> list >> [len, type])


# In[5]:

def test_some_chain():
    assert _x[range, list, [len, type]](10) == [10, list]
    assert _x[:][range][list][len, type](10) == (10, list)
    assert _x[:][range][list][[len, type]](10) == [10, list]
    assert _x(10)[range][list][len, type]() == (10, list)
    assert _x(10)[range][list][[len, type]]() == [10, list]
    assert _x(10)[range][list][[len, type]] >> identity == [10, list]
    assert _x ** 10 >> range >> list >> [len, type] >> identity == [10, list]


# In[ ]:



