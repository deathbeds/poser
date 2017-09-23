

```python
from fidget import *
```


```python
the[range].partition_all(4).enumerate().dict().valmap(list)(10)
```




    {0: [0, 1, 2, 3], 1: [4, 5, 6, 7], 2: [8, 9]}




```bash
%%bash 
jupyter nbconvert --to python fidget/*.ipynb
```

    [NbConvertApp] Converting notebook fidget/__init__.ipynb to python
    [NbConvertApp] Writing 167 bytes to fidget/__init__.py
    [NbConvertApp] Converting notebook fidget/attributes.ipynb to python
    [NbConvertApp] Writing 4204 bytes to fidget/attributes.py
    [NbConvertApp] Converting notebook fidget/callables.ipynb to python
    [NbConvertApp] Writing 9712 bytes to fidget/callables.py
    [NbConvertApp] Converting notebook fidget/fidgets.ipynb to python
    [NbConvertApp] Writing 3582 bytes to fidget/fidgets.py



```python
!jupyter nbconvert --to markdown readme.ipynb
!pyreverse -o png fidget
```

    [NbConvertApp] Converting notebook readme.ipynb to markdown
    [NbConvertApp] Writing 1259 bytes to readme.md
    parsing fidget/__init__.py...
    parsing /Users/tonyfast/fidget/fidget/__init__.py...
    parsing /Users/tonyfast/fidget/fidget/attributes.py...
    parsing /Users/tonyfast/fidget/fidget/callables.py...
    parsing /Users/tonyfast/fidget/fidget/fidgets.py...



```python

```
