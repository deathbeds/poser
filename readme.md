
# `fidget` - A Python and Functional Programming Pidgin Syntax

`fidget` helps you get started.


```python
from fidget import _x
_x(97, 97+26)[range][_x(chr)[map]][''.join][str.upper]['It is as easy as {}'.format]
```




    'It is as easy as ABCDEFGHIJKLMNOPQRSTUVWXYZ'




```python
add = _x(_x[lambda x: x+1, str])[map]
```


```python
from jinja2 import Template
Template("""It is import to mix narrative and code for a {{
_x[:][range][add][list][', '.join](3)
}} 👊.""").render(
    list=list, **globals()
)
```




    'It is import to mix narrative and code for a 1, 2, 3 👊.'



---

`fidget` provides typographically dense interfaces to functional programming objects in Python.  `fidget` is inspired by literate programming, it attempts to reduce the text required to displayed programming logic in software & data-driven narratives.

All `fidget` objects, or `fidgets`, are pure python objects that do not modify the Abstract Syntax Tree.  All of the typographic features of `fidget` are extended from the Python Data Model.  The syntactic sugar for each `fidget` is designed with the [Operator Precedence](https://docs.python.org/3/reference/expressions.html#operator-precedence) in mind. 

`fidget` relies on `toolz` and `traitlets`.  `toolz` provides a python 2 and 3 compatible interface to functional programmming in pure Python.  `toolz` was designed with parallel computing in mind.  Consequently, `fidget` objects can be pickled and used in parallel computing tasks like `joblib`, `luigi`, or `dask`.

The `fidget` namespace is design for a "programming as UI" experience.  In the minimal namespace, function methods and type can be modified with a single text character.  For example, `_l`, `_t`, and `_s` produce a `list`, `tuple`, and `set`, respectively.




Other notes:

* lazy
* Everything is a function
* Functions are immutable, Arguments and Keywords are Muttable

More later:
* interactive type checking
* extensions
