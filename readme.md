
# `composites` compose complex functions

`composites` are untyped functional programming objects in Python _with all the side effects_.  `composites` make it easier to compose/pipeline/chain callables, classes, and other objects into higher-order functions.

                pip install git+https://github.com/tonyfast/composites

# compose functions with `a`, `an`, `the`, or `λ`


    from composites import *; assert a is an is the is λ

A basic example, __enumerate__ a __range__ and create a __dict__ionary.

    f = the[range][reversed][enumerate][dict]
    f(3), f
---





    ({0: 2, 1: 1, 2: 0},
     λ:[<class 'range'>, <class 'reversed'>, <class 'enumerate'>, <class 'dict'>])




Each <b><code>[bracket]</code></b> may accept a __callable__ or __iterable__. In either case,
a __callable__ is appended to the composition.  Compositions are immutable and may have
arbitrary complexity.

    g = f.copy()  # copy f from above so it remains unchanged.
    g[type, len]
    g[{'foo': a.do(print).len(), 'bar': the.identity()}]
    assert f < g 



Brackets juxtapose iterable objects.

    the[range, type], the[[range, type]], the[{range, type}], the[{'x': range, 'y': type}]





    (λ:[juxt(<class 'tuple'>)[<class 'range'>, <class 'type'>]],
     λ:[juxt(<class 'list'>)[<class 'range'>, <class 'type'>]],
     λ:[juxt(<class 'set'>)[<class 'range'>, <class 'type'>]],
     λ:[juxt(<class 'dict'>)[('x', <class 'range'>), ('y', <class 'type'>)]])




Each each composition is immutable.

    assert f[len] is f; f





    λ:[<class 'range'>, <class 'reversed'>, <class 'enumerate'>, <class 'dict'>, <built-in function len>]




But it is easy to copy a composition.

    g = f.copy() 
    assert g is not f and g == f and g[type] > f



# compose functions with attributes

Each composition has an extensible attribution system.  Attributes can be accessed in a shallow or verbose way.

    a.range() == a.builtins.range() == a[range]





    True




# compose functions with symbols

    assert a /  range == a.map(range)
    assert a // range == a.filter(range)
    assert a @  range == a.groupby(range)
    assert a %  range == a.reduce(range)


#### combine item getters, attributes, symbols, and other compositions to express complex ideas.

    f = a['test', 5, {42}] \
     / (a**str&[str.upper, str.capitalize]|a**int&a.range().map(
         a.range(2).len()
     ).list()|a**object&type) \
     * list
    f()


#### use compositions recursively

    f = a[:]
    f[a**a.gt(5)*range | a**a.le(5)*a.add(1)[f]](4)





    range(0, 4)




# Why functional programming with `composites`?

[Functional programming](https://en.wikipedia.org/wiki/Functional_programming) _often_ generates less code, or text, to express operations on complex data structures.  A declarative, functional style of programming approach belies Python's imperative, object-oriented (OO) 
nature. Python provides key [functional programming elements](https://docs.python.org/3/library/functional.html) that are used interchangeably with OO code.  

[`toolz`](https://toolz.readthedocs.io), the nucleus for `composites`, extends Python's functional programming with a set of 
un-typed, lazy, pure, and composable functions.  The functions in `toolz` look familiar 
to [__pandas.DataFrame__](https://tomaugspurger.github.io/method-chaining.html) methods, or [__underscorejs__](http://underscorejs.org/) and [__d3js__](https://d3js.org/) in Javascript.

An intermediate user of [`toolz`](https://toolz.readthedocs.io) will use 
[`toolz.pipe`](https://toolz.readthedocs.ioen/latest/api.html#toolz.functoolz.pipe),
[`toolz.juxt`](https://toolz.readthedocs.ioen/latest/api.html#toolz.functoolz.juxt), 
and [`toolz.compose`](https://toolz.readthedocs.ioen/latest/api.html#toolz.functoolz.compose) to create reusable, 
higher-order functions.  These patterns allow the programmer to express complex concepts 
with less typing/text over a longer time.  Repetitive patterns should occupy 
less screen space; `composites;` helps compose functions with less text. 
                      
A successful implementation of __composites__ should compose __un-typed__, __lazy__, and __serializable__ Python functions that allow
recursion.



# Syntax

A core property of `composites` is that it will not modify Python's abstract syntax tree, rather it expresses 
a large portion of Python's magic methods in the [data model](https://docs.python.org/3/reference/datamodel.html).  It considers Python's 
[order of operations](https://docs.python.org/3/reference/expressions.html#operator-precedence) in the api design.  `composites` provides symbolic expressions for common higher-order 
function operations like `map`, `filter`, `groupby`, and `reduce`. The attributes can access any of the `sys.modules;` with tab completion.

The efficiency of computing will continue to improve.  In modern collaborative development environments 
we must consider the efficiency of the programmer. Programming is a repetitive process requiring physical work from a person. 
__composites__ speed up the creation and reading repetitive and complex tasks.


## `composites` structure

![](classes_composites.png)


# Development
    if __name__== '__main__':
        !jupyter nbconvert --to markdown --TemplateExporter.exclude_input=True readme.ipynb
        !jupyter nbconvert --to python --TemplateExporter.exclude_input_prompt=True composites/*.ipynb
        !autopep8 --in-place --aggressive --aggressive composites/*.py
        !pyreverse -o png -bmy -fALL -p composites composites
        !python -m doctest composites/composites.py composites/objects.py composites/conditions.ipynb composites/operations.ipynb composites/attributes.ipynb composites/canonical.ipynb
        !echo complete


    [NbConvertApp] Converting notebook readme.ipynb to markdown
    [NbConvertApp] Writing 8378 bytes to readme.md
    [NbConvertApp] Converting notebook composites/attributes.ipynb to python
    [NbConvertApp] Writing 5755 bytes to composites/attributes.py
    [NbConvertApp] Converting notebook composites/canonical.ipynb to python
    [NbConvertApp] Writing 3332 bytes to composites/canonical.py
    [NbConvertApp] Converting notebook composites/composites.ipynb to python
    [NbConvertApp] Writing 12340 bytes to composites/composites.py
    [NbConvertApp] Converting notebook composites/conditions.ipynb to python
    [NbConvertApp] Writing 1974 bytes to composites/conditions.py
    [NbConvertApp] Converting notebook composites/objects.ipynb to python
    [NbConvertApp] Writing 5405 bytes to composites/objects.py
    [NbConvertApp] Converting notebook composites/operations.ipynb to python
    [NbConvertApp] Writing 3711 bytes to composites/operations.py
    [NbConvertApp] Converting notebook composites/partials.ipynb to python
    [NbConvertApp] Writing 1514 bytes to composites/partials.py
    parsing composites/__init__.py...
    parsing /Users/tonyfast/fidget/composites/__init__.py...
    parsing /Users/tonyfast/fidget/composites/attributes.py...
    parsing /Users/tonyfast/fidget/composites/canonical.py...
    parsing /Users/tonyfast/fidget/composites/composites.py...
    parsing /Users/tonyfast/fidget/composites/conditions.py...
    parsing /Users/tonyfast/fidget/composites/objects.py...
    parsing /Users/tonyfast/fidget/composites/operations.py...
    parsing /Users/tonyfast/fidget/composites/partials.py...
    complete

