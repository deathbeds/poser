

```python
    from fidget import *

    %reload_ext literacy
```


    from fidget import *

    %reload_ext literacy



```python
    f = the.range() @ (lambda x: x//2)
    f (10)
```


    f = the.range() @ (lambda x: x//2)
    f (10)





    {0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6, 7], 4: [8, 9]}




```python
    from pandas import *

    df = (
        the.Path('/Users/tonyfast/gists/')
        .rglob('*.ipynb')
        .map(
            the[a.identity(), a.read_text().loads()^Exception]
        )
        .filter(the.second()**Exception)
        .dict()
        .do(a.len()["""{} notebooks""".format].print())
        .valmap(a.get('cells', default=[]) * DataFrame)
    )[concat].do(a.len()["""{} cells""".format].print())()

    (the * globals * dict.items @ the.second().type() * then.valmap(len))()
```


    from pandas import *

    df = (
        the.Path('/Users/tonyfast/gists/')
        .rglob('*.ipynb')
        .map(
            the[a.identity(), a.read_text().loads()^Exception]
        )
        .filter(the.second()**Exception)
        .dict()
        .do(a.len()["""{} notebooks""".format].print())
        .valmap(a.get('cells', default=[]) * DataFrame)
    )[concat].do(a.len()["""{} cells""".format].print())()

    (the * globals * dict.items @ the.second().type() * then.valmap(len))()


    289 notebooks
    3065 cells





    {abc.ABCMeta: 14,
     function: 9,
     toolz.functoolz.curry: 2,
     fidget.call: 5,
     builtin_function_or_method: 1,
     tuple: 1,
     str: 6,
     type: 4,
     _frozen_importlib_external.SourceFileLoader: 1,
     NoneType: 1,
     _frozen_importlib.ModuleSpec: 1,
     dict: 1}




```bash
    %%bash 
    jupyter nbconvert --to python --TemplateExporter.exclude_input_prompt=True fidget.ipynb
```


    %%bash 
    jupyter nbconvert --to python --TemplateExporter.exclude_input_prompt=True fidget.ipynb


    [NbConvertApp] Converting notebook fidget.ipynb to python
    [NbConvertApp] Writing 8476 bytes to fidget.py



```python
    !jupyter nbconvert --to markdown readme.ipynb
    !pyreverse -o png -bmy -fALL fidget
```


    !jupyter nbconvert --to markdown readme.ipynb
    !pyreverse -o png -bmy -fALL fidget


    [NbConvertApp] Converting notebook readme.ipynb to markdown
    [NbConvertApp] Writing 2565 bytes to readme.md
    parsing /Users/tonyfast/fidget/fidget.py...



```python

```
