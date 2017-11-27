
# `articles` compose complex functions

`articles` are untyped functional programming objects in Python _with all the side effects_.  They provide a typographically efficient way of expressing complex ideas in code.

                pip install git+https://github.com/tonyfast/articles
                
---
            
        

> `articles` is inspired by the [`d3js`](), [`pandas`](), [`underscorejs`](), and [`toolz`]() apis.

   
               

* untyped lambda calculus
* Higher-order functions enable partial application or currying, 
* Recursion

# compose functions with `a`, `an`, `the`, or `λ`


    from articles import *; assert a is an is the is λ



A basic example, __enumerate__ a __range__ and create a __dict__ionary.

    f = the[range][reversed][enumerate][dict]
    f(3), f





    ({0: 2, 1: 1, 2: 0},
     λ:[<class 'range'>, <class 'reversed'>, <class 'enumerate'>, <class 'dict'>])




Each <b><code>[bracket]</code></b> may accept a __callable__ or __iterable__. In either case,
a __callable__ is appended to the composition.  Compositions are immutable and may have
arbitrary complexity.

    g = f.copy()  # copy f from above so it remains unchanged.
    g[type, len]
    g[{'foo': a.do(print).len(), 'bar': the.identity()}]
    assert f < g 
    g





    λ:[<class 'range'>, <class 'reversed'>, <class 'enumerate'>, <class 'dict'>, juxt(<class 'tuple'>)[<class 'type'>, <built-in function len>], juxt(<class 'dict'>)[('foo', λ:[do:[<built-in function print>], <built-in function len>]), ('bar', λ:[<function identity at 0x1113929d8>])]]



Brackets juxtapose iterable objects.


    the[range, type], the[[range, type]], the[{range, type}], the[{'x': range, 'y': type}]





    (λ:[juxt(<class 'tuple'>)[<class 'range'>, <class 'type'>]],
     λ:[juxt(<class 'list'>)[<class 'range'>, <class 'type'>]],
     λ:[juxt(<class 'set'>)[<class 'type'>, <class 'range'>]],
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


    f = a['test', 5, {42}]      / (a**str&[str.upper, str.capitalize]|a**int&a.range().map(
         a.range(2).len()
     ).list()|a**object&type) \
     * list
    f()





    [['TEST', 'Test'], [0, 0, 0, 1, 2], set]



#### use compositions recursively


    f = a[:]
    f[a**a.gt(5)*range | a**a.le(5)*a.add(1)[f]](4)





    range(0, 6)



## `articles` structure

![](classes_No_Name.png)

# Development


    !jupyter nbconvert --to markdown --TemplateExporter.exclude_input=True readme.ipynb
    !pyreverse -o png -bmy -fALL articles
    !python -m doctest articles/composites.py articles/objects.py articles/conditions.ipynb


    [NbConvertApp] Converting notebook readme.ipynb to markdown
    [NbConvertApp] Writing 6117 bytes to readme.md
    parsing articles/__init__.py...
    parsing /Users/tonyfast/fidget/articles/__init__.py...
    parsing /Users/tonyfast/fidget/articles/attributes.py...
    parsing /Users/tonyfast/fidget/articles/composites.py...
    parsing /Users/tonyfast/fidget/articles/conditions.py...
    parsing /Users/tonyfast/fidget/articles/objects.py...
    parsing /Users/tonyfast/fidget/articles/operations.py...
    parsing /Users/tonyfast/fidget/articles/partials.py...
    **********************************************************************
    File "articles/objects.py", line 26, in objects.enumerated
    Failed example:
        dict(zip(f.data, f(10)))
    Expected:
        {do:[<built-in function len>]: range(0, 10), <class 'type'>: <class 'range'>, <class 'range'>: range(0, 10)}
    Got:
        {do:[<built-in function len>]: (range(0, 10),), <class 'type'>: <class 'tuple'>, <class 'range'>: range(0, 10)}
    **********************************************************************
    1 items had failures:
       1 of   3 in objects.enumerated
    ***Test Failed*** 1 failures.

