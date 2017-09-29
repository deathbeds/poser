

```python
    from fidget import *
```


    from fidget import *



```python
    %reload_ext literacy
```


    %reload_ext literacy



```python
    f = a.range(3) @ a.floordiv(3)
    f (10)
```


    f = a.range(3) @ a.floordiv(3)
    f (10)





    {1: [3, 4, 5], 2: [6, 7, 8], 3: [9]}




```bash
    %%bash 
    jupyter nbconvert --to python fidget/*.ipynb
```


    %%bash 
    jupyter nbconvert --to python fidget/*.ipynb


    [NbConvertApp] Converting notebook fidget/__init__.ipynb to python
    [NbConvertApp] Writing 167 bytes to fidget/__init__.py
    [NbConvertApp] Converting notebook fidget/callables.ipynb to python
    [NbConvertApp] Writing 9878 bytes to fidget/callables.py
    [NbConvertApp] Converting notebook fidget/fidgets.ipynb to python
    [NbConvertApp] Writing 7481 bytes to fidget/fidgets.py



```python
    !jupyter nbconvert --to markdown readme.ipynb
    !pyreverse -o png fidget
```


    !jupyter nbconvert --to markdown readme.ipynb
    !pyreverse -o png fidget


    [NbConvertApp] Converting notebook readme.ipynb to markdown
    [NbConvertApp] Writing 1023 bytes to readme.md
    parsing fidget/__init__.py...
    parsing /Users/tonyfast/fidget/fidget/__init__.py...
    parsing /Users/tonyfast/fidget/fidget/callables.py...
    parsing /Users/tonyfast/fidget/fidget/fidgets.py...



```python

```
