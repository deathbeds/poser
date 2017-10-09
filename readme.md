
# `determiners` compose functions

`determiners` are cracked out Python lists that are `callable`. 

> Inspired by toolz and underscore.chain.


```python
    from determiners import *    
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

`determiners` include the full python data model including incremental and right operators.


```python
    for i in ['__add__', '__sub__', '__mul__', '__truediv__', '__floordiv__', '__matmul__', '__mod__']:
        print(i, getattr(a, i))
```

    __add__ <bound method call.__getitem__ of call>[λ>[]]>
    __sub__ <bound method call.__getitem__ of call>[λ>[]]>
    __mul__ <bound method call.__getitem__ of call>[λ>[]]>
    __truediv__ functools.partial(<bound method compose.__getattr__ of call>[λ>[]]>, <class 'map'>)
    __floordiv__ functools.partial(<bound method compose.__getattr__ of call>[λ>[]]>, <class 'filter'>)
    __matmul__ functools.partial(<bound method compose.__getattr__ of call>[λ>[]]>, <function groupby at 0x10a20fc80>)
    __mod__ functools.partial(<bound method compose.__getattr__ of call>[λ>[]]>, <built-in function reduce>)


### classes


```python
    (an**(int,))(10), (an**(int,))('10')
```




    (True, False)




```python
(a**int*range | a**str*str.upper)("1")
```




    '1'




```python
(a**int*range | a**str*str.upper)('10asdf')
```




    '10ASDF'




```python
    from pandas import *

    df = (
        the.Path('/Users/tonyfast/gists/').rglob('*.ipynb')
        .map(the[a.identity(), a.read_text().loads()^Exception])
        .filter(the.second()**dict).take(10).dict()
        .valmap(a.get('cells', default=[]) * DataFrame)
    )[concat]()

    (the * globals * dict.items @ the.second().type() * a.valmap(len))()
```




    {_frozen_importlib.ModuleSpec: 1,
     function: 10,
     abc.ABCMeta: 18,
     builtin_function_or_method: 4,
     tuple: 1,
     str: 5,
     type: 10,
     NoneType: 1,
     toolz.functoolz.curry: 3,
     _frozen_importlib_external.SourceFileLoader: 1,
     dict: 1,
     determiners.call: 4}



## Development

Convert the single Jupyter notebook to a python script.

## `determiners` structure

![](classes_No_Name.png)


```python
    !jupyter nbconvert --to markdown readme.ipynb
    !pyreverse -o png -bmy -fALL determiners
```

    [NbConvertApp] Converting notebook readme.ipynb to markdown
    [NbConvertApp] Writing 3180 bytes to readme.md
    parsing /Users/tonyfast/fidget/determiners.py...



```bash
    %%bash 
    jupyter nbconvert --to python --TemplateExporter.exclude_input_prompt=True determiners.ipynb
```

    [NbConvertApp] Converting notebook determiners.ipynb to python
    [NbConvertApp] Writing 13964 bytes to determiners.py



```python

```
