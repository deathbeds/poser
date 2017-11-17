
# `articles` compose functions

`articles` provides a generic chainable model for function compositions.  `articles` relies on 
a large portion of the python data model to create a typographically dense interface to function 
compositions.

                pip install git+https://github.com/tonyfast/articles
                
---
            
        

> `articles` is inspired by the `d3js`, `pandas`, `underscorejs`, and `toolz` apis.

   
               


```python
    from articles import *; assert a is an is the is λ
```

## compose functions using brackets

Each composition begins evaluation at the first element in the list.


```python
    f = the[range][list]; f
```




    λ:[<class 'range'>, <class 'list'>]



Brackets juxtapose iterable objects.


```python
    the[range, type], the[[range, type]], the[{range, type}], the[{'x': range, 'y': type}]
```




    (λ:[juxt((<class 'tuple'>,))[<class 'range'>, <class 'type'>]],
     λ:[juxt((<class 'list'>,))[<class 'range'>, <class 'type'>]],
     λ:[juxt((<class 'set'>,))[<class 'type'>, <class 'range'>]],
     λ:[juxt((<class 'dict'>,))[('x', <class 'range'>), ('y', <class 'type'>)]])



Each each composition is immutable.


```python
    assert f[len] is f; f
```




    λ:[<class 'range'>, <class 'list'>, <built-in function len>]



But it is easy to copy a composition.


```python
    g = f.copy() 
    assert g is not f and g == f and g[type] > f
```

# compose functions with attributes

Each composition has an extensible attribution system.  Attributes can be accessed in a shallow or verbose way.


```python
    a.range() == a.builtins.range() == a[range]
```




    True




```python
    a.dir().len()["""articles begins with {} attributes from the modules""".format].print()(a)
    (a//a**__import__('types').ModuleType / (lambda x: getattr(x, '__name__', "")) // a[bool] * ", ".join * print)(a.attributer)
```

    articles begins with 1095 attributes from the modules
    toolz, requests, builtins, json, pickle, io, collections, itertools, functools, pathlib, importlib, inspect


# compose functions with symbols


```python
    assert a /  range == a.map(range)
    assert a // range == a.filter(range)
    assert a @  range == a.groupby(range)
    assert a %  range == a.reduce(range)
```

#### combine item getters, attributes, symbols, and other compositions to express complex ideas.


```python
    f = a['test', 5, {42}] \
     / (a**str&[str.upper, str.capitalize]|a**int&a.range().map(
         a.range(2).len()
     ).list()|a**object&type) \
     * list
    f()
```




    [['TEST', 'Test'], [0, 0, 0, 1, 2], set]



#### use compositions recursively


```python
    f = a[:]
    f[a**a.gt(5)*range | a**a.le(5)*a.add(1)[f]](4)
```




    range(0, 6)



## `articles` structure

![](classes_No_Name.png)

# Development


```python
    !jupyter nbconvert --to markdown readme.ipynb
    !pyreverse -o png -bmy -fALL articles
    !jupyter nbconvert --to python --TemplateExporter.exclude_input_prompt=True articles.ipynb
    !python -m doctest articles.py
```

    [NbConvertApp] Converting notebook readme.ipynb to markdown
    [NbConvertApp] Writing 2875 bytes to readme.md
    parsing /Users/tonyfast/fidget/articles.py...
    [NbConvertApp] Converting notebook articles.ipynb to python
    [NbConvertApp] Writing 17954 bytes to articles.py

