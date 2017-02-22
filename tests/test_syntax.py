
# coding: utf-8

# In[3]:

from fidget import *
import pytest


# In[14]:

list_parameters = [_list_, _tuple_, _set_, _comp_]
container_parameters = [_dict_, _comp_, _condition_]
parameters = pytest.mark.parametrize("_xx", concatv(list_parameters, container_parameters))


# In[15]:

@parameters
def test_init(_xx):
    assert isinstance(_x[:], type(_x()))


# In[16]:

@parameters
def test_arguments(_xx):
    # A function with arguments and keywords
    truth = _xx('test', 42, 'foo', 'bar', keyword='value')
    
    # Append arguments
    callable = _xx[:] ** ('test', 42, 'foo', 'bar')
    assert callable.args == truth.args
    
    # Append keywords
    callable ** {'keyword': 'value'}
    assert callable.kwargs == truth.kwargs and callable.args == truth.args
    
    # Change Arguments
    callable ** ('new',)
    assert callable.args == ('new',)
    
    # Change Keywords
    callable ** {'new': 'object', 'keyword': 'updated'}
    with pytest.raises(AssertionError):
        assert callable.kwargs == {'new': 'object'} 
    assert callable.kwargs == merge(
        {'new': 'object'}, {'keyword': 'updated'}
    ) and callable.args == ('new',)    


# In[18]:

@pytest.mark.parametrize("_xx", list_parameters)
def test_pickle_sequence(_xx):
    assert _xx[range, type]._pickable
    
@pytest.mark.parametrize("_xx", container_parameters)
def test_pickle_container(_xx):
    assert _xx[[str, range], [identity, type]]._pickable


# In[13]:

@pytest.mark.parametrize("_xx", list_parameters)
def test_copy(_xx):
    composition = _xx[range, type]
    assert composition is not composition.copy()


# In[26]:

@pytest.mark.parametrize("_xx", list_parameters)
def test_context(_xx):
    g = _xx(10)[range][type]
    funcs = g.funcs
    with g as F:
        q = F >> str >> type >> str >> type
    assert g.funcs == funcs
    assert g is not q


# In[19]:

@pytest.mark.parametrize("_xx", list_parameters)
def test_reverse(_xx):
    composition = _xx[range, type]
    assert reversed(composition)

