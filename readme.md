
[![Github Actions](https://github.com/deathbeds/poser/workflows/Python%20package/badge.svg)](https://github.com/deathbeds/poser/actions)
[![PyPI version](https://badge.fury.io/py/poser.svg)](https://badge.fury.io/py/poser)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/deathbeds/poser/master)
    
    import poser, toolz
    
# Be a `poser`

`poser` is a API for lazy, (dys)functional python programming. It allows complex functions to be composed using fluent or symbollic interfaces.

```bash
pip install poser
```
        
> _dysfunctional programming_ === Functional programming with all the side effects.

New to [functional programming]? Functional programming uses declarative functions to compose complex operations on arguments.
If you are familiar with python then [`toolz`][toolz] is a great starting point, [`poser`][poser] is a compact [API] for [`toolz`] and the python
[standard library].

`poser` 

# `poser` API

    from poser import λ, Λ, stars

`λ` is a composition that always returns a function. For example, below we create a list of numbers from 1 to a some value

    >>> f = λ.range(1).list(); f
    λ(<class 'list'>, functools.partial(<class 'range'>, 1))
    >>> assert callable(f)
    >>> f(5), f(10)
    ([1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8, 9])
    
`poser` can use forward references to lazily import modules.

    >>> g = λ.range(1).enumerate().dict()['pandas.Series']; g
    λ(ForwardRef('pandas.Series'), <class 'dict'>, <class 'enumerate'>, functools.partial(<class 'range'>, 1))
    >>> λ(9) + g + type + ...
    <class 'pandas.core.series.Series'>

    
`Λ` is for object function composition where the 
function represents symbollic or chained methods 
    
    >>> (Λ*10+2)(1)
    12
    >>> assert (Λ*10+2)(1) == 1*10+2
    >>> s = "A formatted string :{}: with a %s"
    ... (Λ.format('foo') % '% formatted').upper()(s)
    'A FORMATTED STRING :FOO: WITH A % FORMATTED'
    
    >>> assert (Λ.format('foo') % '% formatted').upper()(s)\
    ... == (s.format('foo') % '% formatted').upper()
    
The `poser` API expresses all of the symbols in the python data model.  Refer to the tests for examples.
    
## juxtaposition

    >>> assert isinstance(λ[range, type, str](10), tuple)
    >>> assert isinstance(λ[[range, type, str]](10), list)
    >>> assert isinstance(λ[{range, type, str}](10), set)

Value and keyword functions can be supplied to juxtapose functions across dictionaries.

    λ[{'a': range, type: str}](10)
    {'a': range(0, 10), int: '10'}
    
    
## Example recipes.

    get = λ['functools.lru_cache']()(λ['requests.get'][λ[Λ.json()] ^ BaseException | Λ.text()]); get
    
    read = λ**Λ.startswith('http') & get | λ.Path().read_text()
    
