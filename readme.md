
# `articles` compose functions

`articles` are cracked out Python lists that are `callable`. 

> Inspired by toolz and underscore.chain.


```python
    from articles import *    
    assert a == an == the
```

## Usage

### itemgetter

Use the `getitem` method to append functions togethers.


```python
    the[range][list](10)
```




    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]



### attrgetter

Use the `getattr` method to append `the._attributes` objects togethers.


```python
    the.range().list()(10)
```




    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]



### partial


```python
    the(3).range().list()(10), the.range(3).list()(10)
```




    ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 4, 5, 6, 7, 8, 9])



### operators

`articles` include the full python data model including incremental and right operators.


```python
    for i in ['__add__', '__sub__', '__mul__', '__truediv__', '__floordiv__', '__matmul__', '__mod__']:
        print(i, getattr(a, i))
```

    __add__ <bound method call.__getitem__ of call>[λ>[]]>
    __sub__ <bound method call.__getitem__ of call>[λ>[]]>
    __mul__ <bound method call.__getitem__ of call>[λ>[]]>
    __truediv__ functools.partial(<bound method compose.__getattr__ of call>[λ>[]]>, <class 'map'>)
    __floordiv__ functools.partial(<bound method compose.__getattr__ of call>[λ>[]]>, <class 'filter'>)
    __matmul__ functools.partial(<bound method compose.__getattr__ of call>[λ>[]]>, <function groupby at 0x109190950>)
    __mod__ functools.partial(<bound method compose.__getattr__ of call>[λ>[]]>, <built-in function reduce>)


### classes


```python
    (then**(int,))(10), (then**(int,))('10')
```




    (True, False)




```python
    from pandas import *

    df = (
        the.Path('/Users/tonyfast/gists/').rglob('*.ipynb')
        .map(the[a.identity(), a.read_text().loads()^Exception])
        .filter(-the.second()**Exception).dict()
        .do(a.len()["""{} notebooks""".format].print())
        .valmap(a.get('cells', default=[]) * DataFrame)
    )[concat].do(a.len()["""{} cells""".format].print())()

#     (the * globals * dict.items @ the.second().type() * then.valmap(len))()
```

    
    KeyboardInterrupt
    



```python
    from pandas import *

    df = (
        the.Path('/Users/tonyfast/gists/').rglob('*.ipynb')
        .map(the[a.identity(), a.read_text().loads()^Exception])
        .filter(the.second()**dict).take(10).dict()
        .valmap(a.get('cells', default=[]) * DataFrame)
    )[concat]()

    (the * globals * dict.items @ the.second().type() * then.valmap(len))()
```




    {_frozen_importlib.ModuleSpec: 1,
     abc.ABCMeta: 15,
     articles.call: 5,
     builtin_function_or_method: 2,
     type: 4,
     str: 5,
     dict: 1,
     _frozen_importlib_external.SourceFileLoader: 1,
     NoneType: 1,
     tuple: 1,
     toolz.functoolz.curry: 6,
     function: 9}



## Development

Convert the single Jupyter notebook to a python script.

## `articles` structure

![](classes_No_Name.png)


```python
    !jupyter nbconvert --to markdown readme.ipynb
    !pyreverse -o png -bmy -fALL articles
```


```bash
    %%bash 
    jupyter nbconvert --to python --TemplateExporter.exclude_input_prompt=True articles.ipynb
```

    [NbConvertApp] Converting notebook articles.ipynb to python
    [NbConvertApp] Writing 12219 bytes to articles.py

