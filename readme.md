
# `fidget` - A literate syntax for functional programming

`fidget` provides symbollic & pythonic expressions that compose complex functions.  It uses the Python data model & [operator precedence](https://docs.python.org/3/reference/expressions.html#operator-precedence) to quickly prototype literate code with flexible design choices, _no `AST` was harmed in the making of `fidget`_. Markdown is our primary styleguide for monospaced type.  With `fidget`, Markdown styled lists can also execute code, even executing a `fidget` resembles a link `__x__ [requests.get](url)` - __x__ [requests.get](url).


```python
from fidget import _x as __x__

(__x__
 * range
 * [len, sum, 
    __x__
    .filter(__x__.mod(2))
    [__x__.count(), sum]
   ]
)(100)
```




    [100, 4950, (50, 0)]



`fidget` includes _most_ of the Python model attributes to provide a flexible symbollic syntax 
to compose functions.  It is also loaded with function programming attributes from `toolz` and `operator` including:


```python
__x__[dir].filter(-__x__.first().eq('_')).random_sample(.1)[list][str](__x__)
```




    "['dissoc', 'get_in', 'iand', 'irshift', 'matmul', 'merge_sorted', 'mod', 'partial', 'random_sample', 'reduceby', 'sliding_window']"




```python
_x(10) >> (_x << str >> type >> print) >> __x__.mul(42) >> call()
```

    <class 'str'>





    420



## [Examples](https://github.com/tonyfast/fidget/blob/master/test/data_model.ipynb)

---

## More

This project is developed for interactive computing & uses the Jupyter notebooks to derive formatted Python code.

`toolz` provides a python 2 and 3 compatible interface to functional programmming in pure Python.  `fidget` objects can be pickled and used in parallel computing tasks like `joblib`, `luigi`, or `dask`.  

The `fidget` namespace is design for a "programming as UI" experience.  In the minimal namespace, function methods and type can be modified with a single text character.  


Other notes:

* lazy
* Everything is a function
* Functions are immutable, Arguments and Keywords are Muttable

More later:
* interactive type checking
* extensions


```python
%%bash 
jupyter nbconvert --to custom --Exporter.file_extension .py --TemplateExporter.template_file docify.tpl fidget.ipynb
yapf -i fidget.py
```

    [NbConvertApp] Converting notebook fidget.ipynb to custom
    [NbConvertApp] Writing 14881 bytes to fidget.py



```python
%%bash
cd test
coverage erase
coverage run -a testing.py
coverage run -a data_model.py
coverage report
```

    <class 'NoneType'>
    <class 'range'>
    <class 'range'>
    <class 'NoneType'>
    <class 'str'>
    <IPython.core.display.Markdown object>
    <IPython.core.display.Markdown object>
    attributes for [<class 'fidget.Juxtaposition'>]
    <IPython.core.display.HTML object>
    attributes for [<class 'fidget.Composition'>]
    <IPython.core.display.HTML object>
    <IPython.core.display.Markdown object>
    <IPython.core.display.Markdown object>
    <IPython.core.display.Markdown object>
    <IPython.core.display.Markdown object>
    <IPython.core.display.Markdown object>
    <IPython.core.display.Markdown object>
    [10, 45]
    <IPython.core.display.Markdown object>
    <IPython.core.display.Markdown object>
    <IPython.core.display.Markdown object>
    <IPython.core.display.Markdown object>
    [10, 45, '0.001268148422241211 sec']
    Name                               Stmts   Miss  Cover
    ------------------------------------------------------
    /Users/tonyfast/fidget/fidget.py     241      8    97%
    data_model.py                         69      0   100%
    testing.py                            86      0   100%
    ------------------------------------------------------
    TOTAL                                396      8    98%



```python
%%bash 
jupyter nbconvert --to markdown readme.ipynb
```

    [NbConvertApp] Converting notebook readme.ipynb to markdown
    [NbConvertApp] Writing 3847 bytes to readme.md



```python

```
