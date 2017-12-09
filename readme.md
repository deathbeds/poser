
# `composites` compose complex functions

`composites` are untyped functional programming objects in Python _with all the side effects_.  `composites` make it easier to compose/pipeline/chain callables, classes, and other objects into higher-order functions.

                pip install git+https://github.com/tonyfast/composites

# compose functions with `a`, `an`, `the`, or `λ`


    from composites import *; assert a is an is the

A basic example, __enumerate__ a __range__ and create a __dict__ionary.

    f = the[range][reversed][enumerate][dict]
    f(3), f
---





    ({0: 2, 1: 1, 2: 0}, <composites.Function at 0x100aeaee8>)




Each <b><code>[bracket]</code></b> may accept a __callable__ or __iterable__. In either case,
a __callable__ is appended to the composition.  Compositions are immutable and may have
arbitrary complexity.

    g = f.copy()  # copy f from above so it remains unchanged.
    g[type, len]
    g[{'foo': a.do(print).len(), 'bar': the.identity()}]





    <composites.Function at 0x100aeaee8>




Brackets juxtapose iterable objects.

    the[range, type], the[[range, type]], the[{range, type}], the[{'x': range, 'y': type}]





    (<composites.Function at 0x103d6d468>,
     <composites.Function at 0x103d6d348>,
     <composites.Function at 0x103d6d528>,
     <composites.Function at 0x103d6d6a8>)




Each each composition is immutable.

    assert f[len] is f; f





    <composites.Function at 0x100aeaee8>




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





    False




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
        !jupyter nbconvert --to markdown --execute --ExecutePreprocessor.kernel_name=p6 composites.ipynb
        !python -m doctest composites.py
        !echo complete


    [NbConvertApp] Converting notebook readme.ipynb to markdown
    [NbConvertApp] Writing 9340 bytes to readme.md
    [NbConvertApp] Converting notebook composites.ipynb to markdown
    [NbConvertApp] Executing notebook with kernel: p6
    Traceback (most recent call last):
      File "/Users/tonyfast/anaconda/lib/python3.5/site-packages/jupyter_client/kernelspec.py", line 201, in get_kernel_spec
        resource_dir = d[kernel_name.lower()]
    KeyError: 'p6'
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/Users/tonyfast/anaconda/bin/jupyter-nbconvert", line 11, in <module>
        sys.exit(main())
      File "/Users/tonyfast/anaconda/lib/python3.5/site-packages/jupyter_core/application.py", line 267, in launch_instance
        return super(JupyterApp, cls).launch_instance(argv=argv, **kwargs)
      File "/Users/tonyfast/anaconda/lib/python3.5/site-packages/traitlets/config/application.py", line 658, in launch_instance
        app.start()
      File "/Users/tonyfast/anaconda/lib/python3.5/site-packages/nbconvert/nbconvertapp.py", line 325, in start
        self.convert_notebooks()
      File "/Users/tonyfast/anaconda/lib/python3.5/site-packages/nbconvert/nbconvertapp.py", line 493, in convert_notebooks
        self.convert_single_notebook(notebook_filename)
      File "/Users/tonyfast/anaconda/lib/python3.5/site-packages/nbconvert/nbconvertapp.py", line 464, in convert_single_notebook
        output, resources = self.export_single_notebook(notebook_filename, resources, input_buffer=input_buffer)
      File "/Users/tonyfast/anaconda/lib/python3.5/site-packages/nbconvert/nbconvertapp.py", line 393, in export_single_notebook
        output, resources = self.exporter.from_filename(notebook_filename, resources=resources)
      File "/Users/tonyfast/anaconda/lib/python3.5/site-packages/nbconvert/exporters/exporter.py", line 174, in from_filename
        return self.from_file(f, resources=resources, **kw)
      File "/Users/tonyfast/anaconda/lib/python3.5/site-packages/nbconvert/exporters/exporter.py", line 192, in from_file
        return self.from_notebook_node(nbformat.read(file_stream, as_version=4), resources=resources, **kw)
      File "/Users/tonyfast/anaconda/lib/python3.5/site-packages/nbconvert/exporters/templateexporter.py", line 280, in from_notebook_node
        nb_copy, resources = super(TemplateExporter, self).from_notebook_node(nb, resources, **kw)
      File "/Users/tonyfast/anaconda/lib/python3.5/site-packages/nbconvert/exporters/exporter.py", line 134, in from_notebook_node
        nb_copy, resources = self._preprocess(nb_copy, resources)
      File "/Users/tonyfast/anaconda/lib/python3.5/site-packages/nbconvert/exporters/exporter.py", line 311, in _preprocess
        nbc, resc = preprocessor(nbc, resc)
      File "/Users/tonyfast/anaconda/lib/python3.5/site-packages/nbconvert/preprocessors/base.py", line 47, in __call__
        return self.preprocess(nb, resources)
      File "/Users/tonyfast/anaconda/lib/python3.5/site-packages/nbconvert/preprocessors/execute.py", line 257, in preprocess
        cwd=path)
      File "/Users/tonyfast/anaconda/lib/python3.5/site-packages/nbconvert/preprocessors/execute.py", line 237, in start_new_kernel
        km.start_kernel(**kwargs)
      File "/Users/tonyfast/anaconda/lib/python3.5/site-packages/jupyter_client/manager.py", line 244, in start_kernel
        kernel_cmd = self.format_kernel_cmd(extra_arguments=extra_arguments)
      File "/Users/tonyfast/anaconda/lib/python3.5/site-packages/jupyter_client/manager.py", line 175, in format_kernel_cmd
        cmd = self.kernel_spec.argv + extra_arguments
      File "/Users/tonyfast/anaconda/lib/python3.5/site-packages/jupyter_client/manager.py", line 87, in kernel_spec
        self._kernel_spec = self.kernel_spec_manager.get_kernel_spec(self.kernel_name)
      File "/Users/tonyfast/anaconda/lib/python3.5/site-packages/jupyter_client/kernelspec.py", line 203, in get_kernel_spec
        raise NoSuchKernel(kernel_name)
    jupyter_client.kernelspec.NoSuchKernel: No such kernel named p6
    complete

