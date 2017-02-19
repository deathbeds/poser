
# coding: utf-8

# # Callables syntax
# 
# ## Generic Sugar
# 
# > In order of binding.
# 
# * Add normal or keyword arguments `**`
#     * `_xx[:] ** dict()` defines keywords arguments, keywords are merged and keys are considered, immutable(not enforced)
#     * `_xx[:] ** tuple()` defines normal arguments, arguments are replaced
# * Append to value `>>`
#     * `_xx[:][range] >> list == _xx[:][range][list] == _xx[range, list]` 
# * Call a callable 
#     * `identity` returns the evaluated callable from `_xx.args` and `_xx.kwargs`.
#     * `compose` returns the function composition
#     * Otherwise call the chain with the item as the file function.

# In[2]:

from fidget import *

import pytest


# In[ ]:

iterable_params = [_list_, _tuple_, _chain_, _set_]
container_params = [_dict_, _conditional_]


# In[35]:

parameters = pytest.mark.parametrize("_xx", [
    _list_, _tuple_, _set_, _dict_, _chain_, _conditional_
])


# In[34]:

@parameters
def test_init(_xx):
    assert isinstance(_x[:], type(_x()))


# In[37]:

from toolz.curried import merge


# In[36]:

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

# def test_arguments():
#     for callable in [
#         _set_, _list_, _tuple_, _dict_, _chain_, _multipledispatch_, _conditional_
#     ]:
#         assert (callable[:] ** ('test', 42)).args == callable('test', 42).args


# In[ ]:



