
# `articles` create composable functions



```python
    from articles import *    
    assert a == an == the
```

## Usage


```python
    f = the.range() @ (lambda x: x//2)
    f (10)
```




    {0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6, 7], 4: [8, 9]}




```python
    from pandas import *

    df = (
        the.Path('/Users/tonyfast/gists/').rglob('*.ipynb')
        .map(the[a.identity(), a.read_text().loads()^Exception])
        .filter(the.second()**Exception).dict()
        .do(a.len()["""{} notebooks""".format].print())
        .valmap(a.get('cells', default=[]) * DataFrame)
    )[concat].do(a.len()["""{} cells""".format].print())()

    (the * globals * dict.items @ the.second().type() * then.valmap(len))()
```

    289 notebooks
    3065 cells





    {_frozen_importlib.ModuleSpec: 1,
     function: 9,
     abc.ABCMeta: 16,
     toolz.functoolz.curry: 2,
     builtin_function_or_method: 1,
     articles.call: 5,
     str: 6,
     type: 4,
     dict: 1,
     _frozen_importlib_external.SourceFileLoader: 1,
     NoneType: 1,
     tuple: 1}



## Development

Convert the single Jupyter notebook to a python script.


```bash
    %%bash 
    jupyter nbconvert --to python --TemplateExporter.exclude_input_prompt=True articles.ipynb
```

    [NbConvertApp] Converting notebook articles.ipynb to python
    [NbConvertApp] Writing 8736 bytes to articles.py


## `articles` structure

![](classes_No_Name.png)


```python
    !jupyter nbconvert --to markdown readme.ipynb
    !pyreverse -o png -bmy -fALL articles
```

    [NbConvertApp] Converting notebook readme.ipynb to markdown
    [NbConvertApp] Writing 1532 bytes to readme.md
    parsing /Users/tonyfast/fidget/articles.py...

