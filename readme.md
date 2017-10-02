

```python
    from fidget import *

    %reload_ext literacy
```


```python
    f = a.range(3) @ a.floordiv(3)
    f (10)
```


    f = a.range(3) @ a.floordiv(3)
    f (10)





    {}




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
        .valmap(a.get('cells', default=[]) * DataFrame)
    )[concat].do(a.len()["""{} cells""".format].print())()

    (the * globals * dict.items @ the.second().type() * then.valmap(len))()


    3065 cells





    {fidget.call: 5,
     _frozen_importlib.ModuleSpec: 1,
     function: 9,
     abc.ABCMeta: 14,
     toolz.functoolz.curry: 2,
     tuple: 1,
     str: 6,
     type: 4,
     builtin_function_or_method: 1,
     _frozen_importlib_external.SourceFileLoader: 1,
     NoneType: 1,
     dict: 1}




```bash
    %%bash 
    jupyter nbconvert --to python --TemplateExporter.exclude_input_prompt=True fidget.ipynb
```


    %%bash 
    jupyter nbconvert --to python --TemplateExporter.exclude_input_prompt=True fidget.ipynb


    [NbConvertApp] Converting notebook fidget.ipynb to python
    [NbConvertApp] Writing 8292 bytes to fidget.py



```python
    !jupyter nbconvert --to markdown readme.ipynb
    !pyreverse -o png -bmy -fALL fidget
```


    !jupyter nbconvert --to markdown readme.ipynb
    !pyreverse -o png -bmy -fALL fidget


    [NbConvertApp] Converting notebook readme.ipynb to markdown
    [NbConvertApp] Writing 1978 bytes to readme.md
    parsing /Users/tonyfast/fidget/fidget.py...

