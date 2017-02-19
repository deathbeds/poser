
# coding: utf-8

# * Everything is function
# * Functions are immutable, Arguments and Keywords are Muttable

# # `chain`
# 
# Typographically dense function programming in Python.

# ## Getting Started
# 

# This project is largely inspired by heavy use of [`toolz`](http://toolz.readthedocs.io/en/latest/index.html) for functional programming in `python`. Object-oriented functional programming looks weird.

# In[2]:

get_ipython().magic('reload_ext autoreload')
get_ipython().magic('autoreload 2')


# In[3]:

from chain import *
from IPython import get_ipython
ip = get_ipython()
ip.ast_node_interactivity = 'all'


# ### Chain `_x`

# In[8]:

_x(10)[range][reversed][list]
_x(10) >> range >> reversed >> list


# In[12]:

[:] ** (20, ) >> ('r', str) >> 


# In[4]:

_x(10)[range][list][filter(lambda x: x > 5)][tuple]
_x(10)[range][list] % (lambda x: x > 5) > tuple


# In[7]:

_x('Whatever Forever') | str.upper
type(_)
_x('Whatever Forever') > str.upper
type(_)


# ### Discussion
# 
# * Lazy by default
# * There were no modifications to the `AST`.
# * Consistent Grammar's for 
# * Common operations converted to symbols.
# 

# In[6]:

get_ipython().run_cell_magic('file', 'requirements.txt', 'mypy\nsix\ntoolz\ntraitlets\ntyping')

