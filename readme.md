
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

    (the * globals * dict.items @ the.second().type() * then.valmap(len))()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-11-1754bbbcfb0c> in <module>()
          8 )[concat]()
          9 
    ---> 10 (the * globals * dict.items @ the.second().type() * then.valmap(len))()
    

    NameError: name 'then' is not defined


## Development

Convert the single Jupyter notebook to a python script.

## `determiners` structure

![](classes_No_Name.png)


```python
    !jupyter nbconvert --to markdown readme.ipynb
    !pyreverse -o png -bmy -fALL determiners
```


```bash
    %%bash 
    jupyter nbconvert --to python --TemplateExporter.exclude_input_prompt=True determiners.ipynb
```

    [NbConvertApp] Converting notebook determiners.ipynb to python
    [NbConvertApp] Writing 13069 bytes to determiners.py



```python

```
