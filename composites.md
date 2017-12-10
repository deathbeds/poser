

```python
__doc__ = """composites

composites work with other composites

>>> assert (a * juxt({'a': range}) * type)(10) is dict
>>> assert (a * (x + 32) )(10) == 42
>>> (parallel(4).range()/(1+x+32))(10)
[33, 34, 35, 36, 37, 38, 39, 40, 41, 42]
>>> (a / (ifthen(x>5)[range]) * list)([0, 10])
[False, range(0, 10)]
"""
```

# Complex Composite Functions

Complex composite functions have real and imaginary parts that may except specific errors.  

## Composition

Composite functions use Python syntax to append callable objects to compositions and juxtapositions.  

### Operator 


```python
    from functools import partialmethod, total_ordering, WRAPPER_ASSIGNMENTS, wraps
    
    import operator
    from collections import deque, Sized
    from itertools import zip_longest
    
    from toolz import isiterable, excepts, identity, complement, concat, reduce, groupby, merge, first
    import sys
    from inspect import unwrap
    
    import inspect

    from copy import copy

    dunder = '__{}__'.format
    
    __all__ = 'a', 'an', 'the', 'function', 'flip', 'parallel', 'star', 'do', 'preview', 'x',\
    'op', 'juxt', 'ifthen', 'cache', 'store', 'Composition', 'Operator', #'Juxtaposition'
```

Composing function strictly through the Python datamodel.


```python
    def call(object, *args, **kwargs):  
        return object(*args, **kwargs) if callable(object) else object
    def null(*args, **kwargs): return args[0] if args else None
```


```python
    class partial(__import__('functools').partial):
        def __eq__(self, other):
            if isinstance(other, partial):
                return type(self) is type(other) and self.func == other.func and all(_0==_1 for _0, _1 in zip_longest(self.args, other.args))
```


```python
    class partial_object(partial):
        def __call__(self, object): 
            return self.func(object, *self.args, **self.keywords)
```


```python
    @total_ordering
    class State(object):
        __slots__ = 'imag', 'real', 'exceptions', 'args', 'kwargs'
        def __init__(self, imag=None, real=None, exceptions=None, args=None, kwargs=None):
            
            if real is None: 
                real = list()
            if not isiterable(real): 
                real = [real]
            self.imag = imag
            self.__wrapped__ = self.real = real
            
            if exceptions and not isiterable(exceptions): exceptions = exceptions,
            self.exceptions = exceptions or tuple()
            self.args = tuple() if args is None else args
            self.kwargs = kwargs or dict()
            for attr in WRAPPER_ASSIGNMENTS:
                hasattr(type(self), attr) and setattr(self, attr, getattr(type(self), attr))
            if not hasattr(self, dunder('annotations')):
                self.__annotations__ = dict()
        
        def __len__(self): 
            if hasattr(self.real, '__len__'): return len(self.real)
            return 0
        
        def __eq__(self, other):
            return all(_0 == _1 for _0, _1 in zip_longest(self.real, other))

        def __lt__(self, other):
            return len(self.real) < len(other) and all(_0 == _1 for _0, _1 in zip(self.real, other))
        
        def __bool__(self): return bool(len(self))
        
        def __setitem__(self, item, value): return self.real.__setitem__(item, value)
        
        def __hash__(self): return hash(map(hash, self))
        
        def __getstate__(self):
            return tuple(getattr(self, slot) for slot in self.__slots__)
        
        def __setstate__(self, state):
            return State.__init__(self, *map(copy, state)) or self
        
        def __getattr__(self, attr): 
            if not(hasattr(unwrap(self.real), attr)): raise AttributeError(attr)
            def wrapped(*args, **kwargs):
                nonlocal self
                getattr(unwrap(self.real), attr)(*args, **kwargs)
                return self
            return wrapped
        
        def append(self, item):
            if isinstance(self, Factory): self = self()
            if not callable(item):
                if isinstance(item, (int, slice)) or isinstance(self.real, dict): 
                    if item == slice(None): return self
                    return self.real[item]
                if isiterable(item):
                    item = Juxtaposition(item)            
            return self.__getattr__('append')(item)
        
        __getitem__ = append
        __signature__ = inspect.signature(null)
```


```python
    class Complex(State):
        def __prepare__(self, *args, **kwargs):
            return self.args + args, merge(self.kwargs, kwargs)

        def __except__(self, object):
            return excepts(self.exceptions, object, identity) if self.exceptions and callable(object) else object
        
        def __call__(self, *args, **kwargs):
            
            # Is the imaginary part already true?
            if isinstance(self.imag, bool): 
                imag = self.imag
                
            # Has the imaginary part even been defined?
            # If None, only the real part will be evaluated.
            elif self.imag is None: 
                imag = True
                
            # Test the imaginary part of the function.
            elif callable(self.imag): 
                imag = self.__except__(self.imag)(*args, **kwargs)
                if isinstance(self.imag, Juxtaposition):
                    imag = all(imag)

            else: # Fallback
                imag = self.imag
                
            # Make the imaginary part truthy.
            imag = bool(imag)
                            
            if isinstance(imag, BaseException):
                if imag in self.exceptions:
                    return imag
                raise ConditionException(self.imag)
            return imag
```

# Composites


```python
    class Composite(Complex):
        """A complex composite function.
        
        >>> assert isinstance(Composite()(), __import__('collections').Generator)
        """
        def __init__(cls, real=None, exceptions=None, args=None, **kwargs):
            imag = kwargs.pop('imag', True)
            return super().__init__(imag, real, exceptions, args=args, kwargs=kwargs)

        def __iter__(self):
            yield from map(self.__except__, self.real or [null])

        def __call__(self, *args, **kwargs):
            args, kwargs = self.__prepare__(*args, **kwargs)
            condition = super().__call__(*args, **kwargs)
            if condition is False or isinstance(condition, ConditionException):
                yield condition
            else:
                for value in self:
                    args, kwargs = [value(*args, **kwargs) if callable(value) else value ], {}
                    yield null(*args, **kwargs)
                    if isinstance(first(args), BaseException): break
```


```python
    class Composition(Composite):
        """Callable complex composite objects.
        
        >>> f = Composition(range, exceptions=TypeError)
        >>> assert Juxtaposition(f) is f
        >>> assert f(10) == range(10)  and isinstance(f('10'), TypeError)
        """        
        def __call__(self, *args, **kwargs): 
            results = super().__call__(*args, **kwargs)
            if isinstance(self.real, dict): return type(self.real)(results)
            return deque(results, maxlen=1).pop()
```


```python
    class ComplexOperations:
        """Operations that generate complex composites.
        
        >>> f = a.bool().then(a.range()).excepts(TypeError)
        >>> assert f(10) == range(10) and f(0) is False
        >>> assert isinstance(f('10'), TypeError)
        >>> g = a.instance(int).range()
        >>> assert g(10) == range(10) and g('10') is False
        """
        def __pow__(self, object):
            """Require a complex condition be true before evaluating a composition, otherwise return False.
            
            >>> f = a**int
            >>> assert f(10) is 10 and f('10') is False
            """
            new = IfThen(object, exceptions=self.exceptions, args=self.args, **self.kwargs)
            if self: new.append(self)
            return new
        def __and__(self, object): 
            """Evaluate object if the current composition is True.
            
            >>> f = a[bool] & range
            >>> assert f(0) is False and f(10) == range(10)
            """
            return IfThen(self or bool, exceptions=self.exceptions, args=self.args, **self.kwargs).append(object)
        def __or__(self, object): 
            """Evaluate object if the current composition is False.
            
            >>> f = a[bool] | range
            >>> assert f(10) is False and f(0) == range(0)
            """

            return IfThen(complement(self or bool), exceptions=self.exceptions, args=self.args, **self.kwargs).append(object)
        def __xor__(self, object): 
            """Evaluate a composition returning exceptions in object.
            
            >>> f = a * range ^ TypeError
            >>> assert f(10) == range(10) and isinstance(f('string'), TypeError) 
            """
            self = self[:]
            self.exceptions = object
            return self
        
        then = __and__
        ifnot = __or__
        instance = __pow__
        excepts = __xor__
        
        __neg__ = partialmethod(State.append, operator.not_)
```


```python
    class OperatorOperations(ComplexOperations):        
        def _left(self, callable, arg=None, partial=partial_object):
            return self.append(partial(callable, arg))

        def _right(self, callable, left):
            return OperatorOperations._left(Operator(), callable, left, partial=partial)

        def _bool(self, callable, *args):
            return Operator(self, imag=partial_object(callable, *args))

        def __getattr__(self, attr):
            try: 
                return super().__getattr__(attr)
            except AttributeError:
                return self.append(partial_object(getattr, attr))
        
    for attr in ['add', 'sub', 'mul', 'truediv', 'floordiv', 'mod', 'getitem']:
        op=  getattr(operator, attr)
        setattr(OperatorOperations, dunder(attr), partialmethod(OperatorOperations._left, op))
        setattr(OperatorOperations, '__r' + dunder(attr).lstrip('__'), partialmethod(OperatorOperations._right, op))
    
    for attr in ['gt', 'ge', 'le', 'lt', 'eq', 'ne']:
        setattr(OperatorOperations, dunder(attr), partialmethod(OperatorOperations._bool, getattr(operator, attr)))
    
    for attr in ['abs', 'pos', 'neg', 'pow']:
        setattr(OperatorOperations, dunder(attr), partialmethod(OperatorOperations._left, getattr(operator, attr)))
    del attr
```


```python
    class Operator(OperatorOperations, Composition): 
        """Symbollic compositions that append operations functins.
        
        >>> assert (x+10)(10) is 20
        """
        __annotations__ = {}
```


```python
    class SysAttributes(ComplexOperations):
        shortcuts = 'statistics', 'toolz', 'requests', 'builtins','json', 'pickle', 'io', 'collections', \
        'itertools', 'functools', 'pathlib', 'importlib', 'inspect', 'operator'
        decorators = dict()
        
        def __getattr__(self, attr):
            """Access attributes from sys.modules or self.shortcuts"""
            return __getattr__(self).__getattr__(attr)
        
    class __getattr__(object):
        def __init__(self, object, callable=None, parent=None):
            self.object = object
            self.callable = callable
            self.parent = parent

        def __getattr__(self, attr):
            parent = self.callable
            # Convert the attribute to a callable.
            if self.callable:
                attr = getattr(self.callable, attr)
            else:
                try: return State.__getattr__(self.object, attr)
                except: pass
                
                if attr in sys.modules:
                    attr = sys.modules.get(attr)
                elif isinstance(attr, str):
                    for module in map(__import__, SysAttributes.shortcuts):
                        if hasattr(module, attr): 
                            attr = getattr(module, attr)
                            break
                    else:
                        raise AttributeError(attr)
            
            # Decorate the discovered attribute with the correct partials or call.
            for decorator, set in SysAttributes.decorators.items():
                if attr in set: 
                    attr = partial(decorator, attr)
                    break
            else:
                if isinstance(parent, type): attr = partial(partial_object, attr)
            
            wrapper = wraps(attr) if callable(attr) and not isinstance(attr, type) else False
            
            # Wrap the new object for interaction
            object = __getattr__(self.object, attr, parent) 
            
            if wrapper: object = wrapper(object)
            
            return object

        def __call__(self, *args, **kwargs):
            object = self.callable
            if isinstance(object, partial): 
                object = object(*args, **kwargs)
            return self.object.append(object)

        def __repr__(self): 
            return repr(isinstance(self.callable, partial) and self.callable.args and self.callable.args[0] or self.callable)
            

        def __dir__(self):
            if not self.callable or isinstance(self, SysAttributes):
                base = (
                    list(filter(partial_object(complement(str.__contains__), '.'), sys.modules.keys())) 
                    + list(concat(dir(__import__(module)) for module in SysAttributes.shortcuts)))
            else:
                base = dir(self.callable)
            return base
        
SysAttributes.decorators[partial_object] = [__import__('fnmatch').fnmatch]
SysAttributes.decorators[call] = operator.attrgetter('attrgetter', 'itemgetter', 'methodcaller')(operator)
SysAttributes.decorators[partial_object] += [item for item in vars(operator).values() if item not in SysAttributes.decorators[call]]
```


```python
    class CompositeOperations(SysAttributes):
        """Operations that operator on containers.
        
        >>> assert a@range == a.groupby(range)
        >>> assert a/range == a.map(range)
        >>> assert a//range == a.filter(range)
        >>> assert a%range == a.reduce(range)
        >>> assert copy(a%range) == a.reduce(range)
        """        
        def _left(self, callable, arg=None, partial=partial_object):
            return self.append(callable if arg is None else partial(callable, Juxtaposition(arg)))    
    
        def _right(right, attr, left):
            return getattr(CompositeOperations._left(Function(), left), dunder(attr))(right)

        __truediv__ = map = partialmethod(_left, map, partial=partial )
        __floordiv__ = filter = partialmethod(_left, filter, partial=partial)
        __mod__ = reduce = partialmethod(_left, reduce, partial=partial)
        __matmul__ = groupby =  partialmethod(_left, groupby, partial=partial)
        __add__ = __mul__ = __sub__ = __rshift__= State.append
        
        def __lshift__(self, object): return self.append(Do(object))
        do = __lshift__

    for attr in ['add', 'sub', 'mul', 'truediv', 'getitem', 'rshift', 'lshift']:
        setattr(CompositeOperations, '__r' + dunder(attr).lstrip('__'), partialmethod(CompositeOperations._right, attr))        
```


```python
    class ConditionException(BaseException): ...
```


```python
    class Factory(Composition):
        def __bool__(self): return False
        __dir__ = __getattr__.__dir__
```


```python
    class OperatorFactory(Operator, Factory): ...
```

# Juxtapositions


```python
    class Juxtapose(Composite):
        def __iter__(self):
            if isinstance(self.real, dict):
                yield from [Juxtaposition(tuple(map(self.__except__, item))) for item in self.real.items()]
            else:
                yield from map(self.__except__, self.real)
            
        def __call__(self, *args, **kwargs): 
            args, kwargs = self.__prepare__(*args, **kwargs)
            condition = super().__call__(*args, **kwargs)
            if condition is False or isinstance(condition, ConditionException):
                yield condition
            else:
                for value in self:
                    yield call(value, *args, **kwargs)
```


```python
    class Juxtaposition(CompositeOperations, Juxtapose):
        """
        >>> f = Juxtaposition(exceptions=TypeError)[range][type]
        >>> assert f(10) == [range(10), type(10)]
        >>> assert isinstance(f('10')[0], TypeError) 
        >>> assert isinstance(Juxtapose()[range][type](10), __import__('collections').Generator)
        
        
        >>> assert isinstance(Juxtaposition((range for i in range(10)))(10), __import__('collections').Generator) 
        >>> assert isinstance(Juxtaposition(tuple(range for i in range(10)))(10), tuple) 
        >>> assert isinstance(Juxtaposition(set(range for i in range(10)))(10), set)        
        
        
        >>> assert isinstance(juxt({}).real, dict) and isinstance(juxt([]).real, list)
        """
        
        def __new__(cls, real=None, exceptions=None, args=None, kwargs=None):
            """Juxposition is a callable dispatch operation.
            
            >>> assert Juxtaposition(range) is range
            """
            if isinstance(real, Factory): real = real()
            if isinstance(real, str): real = Composition([real])
            if callable(real): return real
            
            if real is None: real = list()
            if not isiterable(real): real = [real]
            
            # Return generators for generator inputs
            if not isinstance(real, Sized):
                return Juxtapose(real, exceptions, imag=True, args=args, kwargs=kwargs)
            
            # Return native types after the 
            self = super().__new__(cls)
            return self.__init__(real, exceptions, args=args, kwargs=kwargs) or self
        
        def __call__(self, *args, **kwargs): 
            iter = super().__call__(*args, **kwargs) 
            return type(self.real)(iter)
```


```python
    class Function(CompositeOperations, Composition):
        """Callable complex composite objects.
        
        >>> f = Composition(range, exceptions=TypeError)
        >>> assert Juxtaposition(f) is f
        >>> assert f(10) == range(10)  and isinstance(f('10'), TypeError)
        """
        __dir__ = __getattr__.__dir__
            
```


```python
    class FunctionFactory(Function, Factory):        
        def __call__(self, *args, **kwargs):
            return super().__call__(args=args, **kwargs)        
        __dir__ = __getattr__.__dir__
```


```python
    class IfThen(Function):      
        """
        >>> f = IfThen(bool)[range]
        >>> assert f(0) is False 
        >>> assert f(10) == range(10)
        """
        def __init__(self, imag=bool, real=None, exceptions=None, args=None, kwargs=None):
            # The imaginary part is a composition
            if isinstance(imag, (type, tuple)) and imag is not bool:
                imag = partial_object(isinstance, imag)
            State.__init__(self, imag, real, exceptions, args=args, kwargs=kwargs)
            if not ConditionException in self.exceptions:
                self.exceptions += ConditionException, 
```


```python
    class Flip(Function):
        def __prepare__(self, *args, **kwargs):
            args, kwargs = super().__prepare__(*args, **kwargs)
            return tuple(reversed(args)), kwargs
```


```python
    class Star(Function):
        def __call__(self, object, *args, **kwargs):
            if isinstance(object, dict):
                kwargs.update(object)
            else:
                args += tuple(object)
            return super().__call__(*args, **kwargs)
```


```python
    class Do(Function):
        def __call__(self, *args, **kwargs):
            super().__call__(*args, **kwargs)
            return null(*args, **kwargs)
```


```python
    class Preview(Function):
        def __repr__(self): return repr(self())
```


```python
    class Parallel(Function):
        """An embarassingly parallel composition.
        
        >>> import joblib
        >>> def g(x): return x+10
        >>> assert parallel(4).range().map(x+10)(100)
        """
        def __init__(self, jobs=4, *args, **kwargs):
            self.jobs = jobs
            super().__init__(*args, **kwargs)

        def map(self, object):            
            return super().map(__import__('joblib').delayed(object))

        def __call__(self, *args, **kwargs):
            return __import__('joblib').Parallel(self.jobs)(super().__call__(*args, **kwargs))

        __truediv__ = map
```


```python
    a = an = the = function = FunctionFactory(Function)
    flip = FunctionFactory(Flip)
    parallel = Factory(Parallel)
    star = FunctionFactory(Star)
    do = FunctionFactory(Do)
    preview = FunctionFactory(Preview)
    x = op = OperatorFactory(Operator)
    juxt = Factory(Juxtaposition)
    ifthen = Factory(IfThen)
```


```python
    class store(dict):
        @property
        def __self__(self): return self.__call__.__self__
        def __init__(self, real=None, *args):
            self.real = Composition() if real is None else real
            super().__init__(*args)
        def __call__(self, *args, **kwargs):
            self[args] = self.real(*args, **kwargs)
            return self[args]
        
        def __getitem__(self, item):
            if not isinstance(item, tuple): item = item,
            return super().__getitem__(item)
```


```python
    class cache(store):
        def __call__(self, *args, **kwargs):
            if args not in self:
                return super().__call__(*args, **kwargs)
            return self[args]
```


```python
    if __name__ == '__main__' and 'runtime' in sys.argv[-1]:
        print(__import__('doctest').testmod())
```

# Developer


```python
    if __name__ == '__main__':
        if 'runtime' in sys.argv[-1]:
            print('Running ', __name__)
            from IPython import get_ipython
            !jupyter nbconvert --to python --TemplateExporter.exclude_input_prompt=True composites.ipynb
            # Juxtaposition still wont work
            !pydoc -w composites
            !autopep8 --in-place --aggressive --aggressive composites.py
            !flake8 composites.py            
            !pyreverse -o png -pcomposites -fALL composites
            !pyreverse -o png -pcomposites.min composites

            !python -m composites 
        else:
            print('run from cli')
```

    run from cli

