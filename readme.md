
> The readme is a work in progress.

# Be a `poser`

`poser` is a fluent interface for lazy, functional python programming.

        pip install poser

# `poser` API

Commonly, a `poser` expression will start with `λ`.


```python
    from poser import λ, a, an, the, x, composition
    assert λ is a is an is the
```

`poser` is inspired by Python functional programming library `toolz`.  `poser` provides dense API to integrate functional programming into python code.



```python
    from toolz.curried import *; from toolz.curried.operator import *
```

# Chainable methods

λ composes a higher-order function that will `pipe` a set of arguments and keywords through a order list of functions.


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
    assert (λ * range / λ.mul(10) * list)(10) == f(10)
```


```python
    if __name__== '__main__':
        !jupyter nbconvert --to markdown readme.ipynb
        !source activate p6 && pytest
        !source activate p6 && python -m doctest poser.py
```

    [NbConvertApp] Converting notebook readme.ipynb to markdown
    [NbConvertApp] Writing 1761 bytes to readme.md
    [1m============================= test session starts ==============================[0m
    platform darwin -- Python 3.6.3, pytest-3.5.0, py-1.5.3, pluggy-0.6.0
    benchmark: 3.1.1 (defaults: timer=time.perf_counter disable_gc=False min_rounds=5 min_time=0.000005 max_time=1.0 calibration_precision=10 warmup=False warmup_iterations=100000)
    rootdir: /Users/tonyfast/poser, inifile:
    plugins: cov-2.5.1, benchmark-3.1.1, hypothesis-3.56.5, importnb-0.3.1
    collected 0 items                                                              [0m
    
    [33m[1m========================= no tests ran in 0.78 seconds =========================[0m

