
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

    __add__ <bound method factory.__getitem__ of factory>[λ>[]]>
    __sub__ <bound method factory.__getitem__ of factory>[λ>[]]>
    __mul__ <bound method factory.__getitem__ of factory>[λ>[]]>
    __truediv__ functools.partial(<bound method op_attr of factory>[λ>[]]>, '__truediv__')
    __floordiv__ functools.partial(<bound method op_attr of factory>[λ>[]]>, '__floordiv__')
    __matmul__ functools.partial(<bound method op_attr of factory>[λ>[]]>, '__matmul__')
    __mod__ functools.partial(<bound method op_attr of factory>[λ>[]]>, '__mod__')


### classes


```python
    f = a**int*range | a**str*str.upper
    f(10), f('abc'), f(10.)
```




    (range(0, 10), 'ABC', False)




```python
    from pandas import *

    df = (
        the.Path('/Users/tonyfast/gists/').rglob('*.ipynb')
        .map(the[a.identity(), a.read_text().loads()^Exception])
        .filter(the.second()).take(10).dict()
        .valmap(a.get('cells', default=[]) * DataFrame)
    )[concat]()

    (the * globals * dict.items @ the.second().type() * a.valmap(len))()
```




    {_frozen_importlib.ModuleSpec: 1,
     function: 9,
     abc.ABCMeta: 19,
     dict: 1,
     tuple: 1,
     str: 5,
     type: 10,
     builtin_function_or_method: 4,
     toolz.functoolz.curry: 2,
     _frozen_importlib_external.SourceFileLoader: 1,
     NoneType: 1,
     determiners.factory: 4}



## Development

Convert the single Jupyter notebook to a python script.

## `determiners` structure

![](classes_No_Name.png)


```bash
    %%bash 
    jupyter nbconvert --to python --TemplateExporter.exclude_input_prompt=True determiners.ipynb
```

    [NbConvertApp] Converting notebook determiners.ipynb to python
    [NbConvertApp] Writing 12644 bytes to determiners.py



```python
    !jupyter nbconvert --to markdown readme.ipynb
    !pyreverse -o png -bmy -fALL determiners
```

    [NbConvertApp] Converting notebook readme.ipynb to markdown
    [NbConvertApp] Writing 4342 bytes to readme.md
    parsing /Users/tonyfast/fidget/determiners.py...

