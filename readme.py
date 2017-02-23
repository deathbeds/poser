
# coding: utf-8

# # `fidget` - A Python and Functional Programming Pidgin Syntax
# 
# `fidget` helps you get started.

# In[1]:

from fidget import _x
_x(97, 97+26)[range][_x(chr)[map]][''.join][str.upper]['It is as easy as {}'.format]


# In[2]:

from fidget import *


# In[3]:

(_x(10) << str >> print >> compose)


# In[6]:

_x(10) >> (_x << str >> type >> print) >> mul(42)


# In[5]:

from jinja2 import Template
Template("""It is import to mix narrative and code for a {{
_x[:][range][add][list][', '.join](3)
}} 👊.""").render(
    list=list, **globals()
)


# ## Composite Functions
# 
# |short|long|desc|
# |------|--------------|------------------------|
# |  _x  | _comp_       |  A composite function. |
# |  x_  | _pmoc_       |  A composite function with the argument order reversed. | 
# 
# ## Juxtapose Functions
# 
# |short|long|desc|
# |------|--------------|------------------------|
# | none | _sequence_   |  A composition from a generator that yields a generator. |
# |  _t  | _tuple_      |  A composition from a tuple that yields a tuple. |
# |  _l  | _list_       |  A composition from a list that yields a list. | 
# |  _s  | _set_        |  A composition from a set that yields a dict with the set functions as keys. |
# |  _d  | _dict_       |  A composition from a dict that yields a dict with the keys and values evaluated. |
# |  _f  | _condition_  |  A composition that can handle logic.
# 
# # Object Functions
# 
# |long|desc|
# |------|--------------|------------------------|
# | _self | An chainable object.  |
# | _this | An chainable object with the value recursively evaluated. |
# 
# # Tokens
# 
# | key | desc |
# |----|------|
# | compose | Compose a higher-order function |
# | copy    | Copy a function |
# | identity | Reveal the identity of a composition. |

# ---
# 
# `fidget` provides typographically dense interfaces to functional programming objects in Python.  `fidget` is inspired by literate programming, it attempts to reduce the text required to displayed programming logic in software & data-driven narratives.
# 
# All `fidget` objects, or `fidgets`, are pure python objects that do not modify the Abstract Syntax Tree.  All of the typographic features of `fidget` are extended from the Python Data Model.  The syntactic sugar for each `fidget` is designed with the [Operator Precedence](https://docs.python.org/3/reference/expressions.html#operator-precedence) in mind. 
# 
# `fidget` relies on `toolz` and `traitlets`.  `toolz` provides a python 2 and 3 compatible interface to functional programmming in pure Python.  `toolz` was designed with parallel computing in mind.  Consequently, `fidget` objects can be pickled and used in parallel computing tasks like `joblib`, `luigi`, or `dask`.  `traitlets` provides an method to interactively type check composite functions.
# 
# The `fidget` namespace is design for a "programming as UI" experience.  In the minimal namespace, function methods and type can be modified with a single text character.  For example, `_l`, `_t`, and `_s` produce a `list`, `tuple`, and `set`, respectively.
# 
# 
# 
# 
# Other notes:
# 
# * lazy
# * Everything is a function
# * Functions are immutable, Arguments and Keywords are Muttable
# 
# More later:
# * interactive type checking
# * extensions
