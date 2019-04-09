
> The readme is a work in progress.

# Be a `poser`

`poser` is a fluent interface for lazy, (dis)functional python programming.

        pip install poser
        
> _disfunctional programming_ === Functional programming with all the side effects.

# `poser` API


```python
    from poser import λ, Composition, Λ, watch
    from toolz.curried import *; from toolz.curried.operator import *
```

# Chainable function compositions

λ composes a higher-order function that will `pipe` a set of arguments and keywords through an ordered list of functions.


```python
    f = (
        λ.range()
        .map(
            λ.mul(10))
        .list())
```

This composition is compared below with a toolz.compose, toolz.pipe, and standard lib python.


```python
   assert f(10) \
        == compose(list, map(mul(10)), range)(10) \
        == pipe(10, range, map(mul(10)), list) \
        == list(map(mul(10), range(10)))
```

## The explicit api

The api above uses shortcuts to modules that a hasty programmer may prefer.  The explicit api accesses functions by their package names first.


```python
    g = λ.builtins.range().map(
        λ.operator.mul(10)
    ).builtins.list()
```

### Imports

`poser` will import modules if they are not available.  For example, if `pandas` is not __import__ed then `poser` will __import __ it.

    λ.pandas.DataFrame()

# Symbollic composition


```python
    assert (λ[range] * λ.mul(10) + list)(10) == f(10)
```


```python
    if __name__== '__main__':
        !jupyter nbconvert --to markdown readme.ipynb
        !ipython -m poser
```


```python

```
