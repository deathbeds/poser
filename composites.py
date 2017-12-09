# coding: utf-8

# # Complex Composite Functions
#
# Complex composite functions have real and imaginary parts that may except specific errors.
#
# ## Composition
#
# Composite functions use Python syntax to append callable objects to compositions and juxtapositions.
#
# ### Canonical

# In[1]:

from functools import partialmethod, total_ordering, WRAPPER_ASSIGNMENTS, wraps

import operator
from collections import deque, Sized, Generator
from itertools import zip_longest

from toolz import isiterable, excepts, identity, complement, concat, reduce, groupby, merge, first
import sys
from inspect import unwrap

from copy import copy

dunder = '__{}__'.format

# Composing function strictly through the Python datamodel.

# In[2]:


def call(object, *args, **kwargs):
    return (object if callable(object) else null)(*args, **kwargs)


def null(*args, **kwargs):
    return args[0] if args else None

# In[3]:


class partial(__import__('functools').partial):
    def __eq__(self, other):
        if isinstance(other, partial):
            return type(self) is type(
                other) and self.func == other.func and all(
                    _0 == _1 for _0, _1 in zip_longest(self.args, other.args))


# In[4]:


class partial_object(partial):
    def __call__(self, object):
        return self.func(object, *self.args, **self.keywords)


# In[55]:


@total_ordering
class State(object):
    __slots__ = 'imag', 'real', 'excepts', 'args', 'kwargs'

    def __init__(self,
                 imag=None,
                 real=None,
                 excepts=None,
                 args=None,
                 kwargs=None):
        self.imag = imag
        if real is None:
            real = list()
        if not isiterable(real):
            real = [real]
        self.__wrapped__ = self.real = real
        if excepts and not isiterable(excepts): excepts = excepts,
        self.excepts = excepts or tuple()
        self.args = tuple() if args is None else args
        self.kwargs = kwargs or dict()
        for attr in WRAPPER_ASSIGNMENTS:
            hasattr(type(self), attr) and setattr(self, attr,
                                                  getattr(type(self), attr))
        if not hasattr(self, dunder('annotations')):
            self.__annotations__ = dict()

    def __len__(self):
        if hasattr(self.real, '__len__'): return len(self.real)
        return 0

    def __eq__(self, other):
        return all(_0 == _1 for _0, _1 in zip_longest(self.real, other))

    def __lt__(self, other):
        return len(self.real) < len(other) and all(
            _0 == _1 for _0, _1 in zip(self.real, other))

    def __bool__(self):
        return bool(len(self))

    def __setitem__(self, item, value):
        return self.real.__setitem__(item, value)

    def __hash__(self):
        return hash(map(hash, self))

    def __getstate__(self):
        return tuple(getattr(self, slot) for slot in self.__slots__)

    def __setstate__(self, state):
        return State.__init__(self, *map(copy, state)) or self

    def __getattr__(self, attr):
        if not (hasattr(unwrap(self.real), attr)): raise AttributeError(attr)

        def wrapped(*args, **kwargs):
            nonlocal self
            getattr(unwrap(self.real), attr)(*args, **kwargs)
            return self

        return wrapped

    def append(self, item):
        if isinstance(self, Factory): self = self()
        if not callable(item) and isinstance(item, (dict, int, slice)):
            if item == slice(None): return self
            return self.real[item]

        if not isinstance(self, Juxtaposition):
            self, item = map(Juxtaposition, (self, item))
        return self.__getattr__('append')(Juxtaposition(item))

    __getitem__ = append


class Complex(State):
    def __prepare__(self, *args, **kwargs):
        return self.args + args, merge(self.kwargs, kwargs)

    def __except__(self, object):
        return excepts(
            self.excepts, object,
            identity) if self.excepts and callable(object) else object

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

        else:  # Fallback
            imag = self.imag

        # Make the imaginary part truthy.
        imag = bool(imag)

        if isinstance(imag, BaseException):
            if imag in self.excepts:
                return imag
            raise ConditionException(self.imag)
        return imag


# In[56]:


class SysAttributes:
    shortcuts = 'statistics', 'toolz', 'requests', 'builtins', 'json', 'pickle', 'io', 'collections', 'itertools', 'functools', 'pathlib', 'importlib', 'inspect', 'operator'
    decorators = dict()

    def __getattr__(self, attr):
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
            try:
                return State.__getattr__(self.object, attr)
            except:
                pass

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

        wrapper = wraps(attr) if callable(attr) and not isinstance(
            attr, type) else False

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
        return repr(
            isinstance(self.callable, partial) and self.callable.args and
            self.callable.args[0] or self.callable)

    def __dir__(self):
        base = dir(self.object)
        if isinstance(self, SysAttributes) or not self.callable:
            return base + (list(
                filter(
                    partial_object(complement(str.__contains__), '.'),
                    sys.modules.keys())) + list(
                        concat(
                            dir(__import__(module))
                            for module in SysAttributes.shortcuts)))
        return base + dir(self.callable)


import fnmatch
SysAttributes.decorators[partial_object] = [fnmatch.fnmatch]
SysAttributes.decorators[call] = operator.attrgetter(
    'attrgetter', 'itemgetter', 'methodcaller')(operator)
SysAttributes.decorators[partial_object] += [
    item for item in vars(operator).values()
    if item not in SysAttributes.decorators[call]
]

# In[57]:


class ComplexOperations:
    """Operations that generate complex composites.
    
    >>> assert a@range == a.groupby(range)
    >>> assert a/range == a.map(range)
    >>> assert a//range == a.filter(range)
    >>> assert a%range == a.reduce(range)
    >>> assert copy(a%range) == a.reduce(range)
    """

    def __pow__(self, object):
        new = IfThen(
            object, excepts=self.excepts, args=self.args, **self.kwargs)
        if self: new.append(self)
        return new

    def __and__(self, object):
        return IfThen(
            self or bool, excepts=self.excepts, args=self.args,
            **self.kwargs).append(object)

    def __or__(self, object):
        return IfThen(
            complement(self or bool),
            excepts=self.excepts,
            args=self.args,
            **self.kwargs).append(object)

    def __xor__(self, object):
        self = self[:]
        self.excepts = object
        return self


# In[58]:


def complex_operation(self, callable, *args, partial=partial_object, **kwargs):
    return self.append(
        partial(callable, *args, **kwargs) if args or kwargs else callable)


def right_operation(right, attr, left):
    return getattr(complex_operation(Function(), left),
                   dunder(attr))(Juxtaposition(right))


class HigherOrderOperations(ComplexOperations):
    __truediv__ = map = partialmethod(complex_operation, map, partial=partial)
    __floordiv__ = filter = partialmethod(
        complex_operation, filter, partial=partial)
    __mod__ = reduce = partialmethod(
        complex_operation, reduce, partial=partial)
    __matmul__ = groupby = partialmethod(
        complex_operation, groupby, partial=partial)
    __add__ = __mul__ = __sub__ = __rshift__ = State.append

    def __lshift__(self, object):
        return self.append(Do(object))


for attr in ['add', 'sub', 'mul', 'truediv', 'getitem', 'rshift', 'lshift']:
    setattr(HigherOrderOperations, '__r' + dunder(attr).lstrip('__'),
            partialmethod(right_operation, attr))

# In[59]:


def right_canonical_operation(self, callable, left):
    return complex_operation(Canonical(), callable, left, partial=partial)


def logical_operation(self, callable, *args):
    return Canonical(self, imag=partial(callable, *args))


class CanonicalOperations(ComplexOperations):
    def __getattr__(self, attr):
        try:
            return super().__getattr__(attr)
        except AttributeError:
            return self.append(partial_object(getattr, attr))


for attr in ['add', 'sub', 'mul', 'truediv', 'floordiv', 'mod', 'getitem']:
    op = getattr(operator, attr)
    setattr(CanonicalOperations,
            dunder(attr), partialmethod(complex_operation, op))
    setattr(CanonicalOperations, '__r' + dunder(attr).lstrip('__'),
            partialmethod(right_canonical_operation, op))

for attr in ['gt', 'ge', 'le', 'lt', 'eq', 'ne']:
    setattr(CanonicalOperations,
            dunder(attr),
            partialmethod(logical_operation, getattr(operator, attr)))

for attr in ['abs', 'pos', 'neg', 'pow']:
    setattr(CanonicalOperations,
            dunder(attr),
            partialmethod(complex_operation, getattr(operator, attr)))
del attr

# In[60]:


class ConditionException(BaseException):
    ...

# In[61]:


class Juxtapose(Complex):
    def __iter__(self):
        yield from map(self.__except__, self.real)

    def __call__(self, *args, **kwargs):
        args, kwargs = self.__prepare__(*args, **kwargs)
        condition = super().__call__(*args, **kwargs)
        if condition is False or isinstance(condition, ConditionException):
            yield condition
        else:
            for value in self:
                yield value(*args, **kwargs) if callable(value) else value


# In[62]:


class Juxtaposition(SysAttributes, HigherOrderOperations, Juxtapose):
    """
    >>> f = Juxtaposition(excepts=TypeError)[range][type]
    >>> assert f(10) == [range(10), type(10)]
    >>> assert isinstance(f('10')[0], TypeError) 
    >>> assert isinstance(Juxtapose()[range][type](10), Generator)
    
    
    >>> assert isinstance(Juxtaposition((range for i in range(10)))(10), Generator) 
    >>> assert isinstance(Juxtaposition(tuple(range for i in range(10)))(10), tuple) 
    >>> assert isinstance(Juxtaposition(set(range for i in range(10)))(10), set)        
    
    """

    def __new__(cls, real=None, excepts=None, args=None, kwargs=None):
        """Juxposition is a callable dispatch operation.
        
        >>> assert Juxtaposition(range) is range
        """
        if isinstance(real, Factory): real = real()
        if callable(real): return real

        if real is None: real = list()
        if isinstance(real, str): pass  # ignore strings
        elif not isiterable(real): real = [real]

        # Return generators for generator inputs
        if not isinstance(real, Sized):
            return Juxtapose(True, real, excepts, args=args, kwargs=kwargs)
        # Return native types after the
        self = super().__new__(cls)
        return self.__init__(real, excepts, args=args, kwargs=kwargs) or self

    def __init__(cls, real=None, excepts=None, args=None, **kwargs):
        super().__init__(
            kwargs.pop('imag', True), real, excepts, args=args, kwargs=kwargs)

    def __call__(self, *args, **kwargs):
        iter = super().__call__(*args, **kwargs)
        return type(self.real)(iter)


# In[63]:


class Composite(Complex):
    """A complex composite function.
    
    >>> assert isinstance(Composite()(), Generator)
    """

    def __iter__(self):
        yield from map(self.__except__, self.real or [null])

    def __call__(self, *args, **kwargs):
        args, kwargs = self.__prepare__(*args, **kwargs)
        condition = super().__call__(*args, **kwargs)
        if condition is False or isinstance(condition, ConditionException):
            yield condition
        else:
            for value in self:
                args, kwargs = [
                    value(*args, **kwargs) if callable(value) else value
                ], {}
                yield null(*args, **kwargs)
                if isinstance(first(args), BaseException): break


# In[64]:


class Composition(Composite):
    """Callable complex composite objects.
    
    >>> f = Composition(range, excepts=TypeError)
    >>> assert Juxtaposition(f) is f
    >>> assert f(10) == range(10)  and isinstance(f('10'), TypeError)
    """

    def __init__(cls, real=None, excepts=None, args=None, **kwargs):
        imag = kwargs.pop('imag', True)
        imag, real = map(Juxtaposition, (imag, real))
        return super().__init__(imag, real, excepts, args=args, kwargs=kwargs)

    def __call__(self, *args, **kwargs):
        results = super().__call__(*args, **kwargs)
        if isinstance(self.real, dict): return type(self.real)(results)
        return deque(results, maxlen=1).pop()


# In[65]:


class Function(SysAttributes, HigherOrderOperations, Composition):
    __dir__ = __getattr__.__dir__


# In[66]:


class Factory:
    ...

# In[67]:


class FunctionFactory(Function, Factory):
    def __call__(self, *args, **kwargs):
        return super().__call__(args=args, **kwargs)

    def __bool__(self):
        return False


# In[68]:


class IfThen(Function):
    """
    >>> f = IfThen(bool)[range]
    >>> assert f(0) is False and f(10) == range(10)
    """

    def __init__(self,
                 imag=bool,
                 real=None,
                 excepts=None,
                 args=None,
                 kwargs=None):
        # The imaginary part is a composition
        if isinstance(imag, (type, tuple)) and imag is not bool:
            imag = partial_object(isinstance, imag)
        State.__init__(self, imag, real, excepts, args=args, kwargs=kwargs)
        if not ConditionException in self.excepts:
            self.excepts += ConditionException,


# In[69]:


class Canonical(CanonicalOperations, HigherOrderOperations, Composition):
    __annotations__ = {}


class CanonicalFactory(Canonical, Factory):
    ...

# In[70]:


class Flip(Function):
    def __prepare__(self, *args, **kwargs):
        args, kwargs = super().__prepare__(*args, **kwargs)
        return tuple(reversed(args)), kwargs


# In[71]:


class Star(Function):
    def __call__(self, object, *args, **kwargs):
        if isinstance(object, dict):
            kwargs.update(object)
        else:
            args += tuple(object)
        return super().__call__(*args, **kwargs)


# In[72]:


class Do(Function):
    def __call__(self, *args, **kwargs):
        super().__call__(*args, **kwargs)
        return null(*args, **kwargs)


# In[73]:


class Preview(Function):
    def __repr__(self):
        return repr(self())


# In[74]:


class Parallel(Function):
    """An embarassingly parallel composition.
    
    >>> import joblib
    >>> def g(x): return x+10
    >>> assert parallel(4).range().map(x+10)(100)
    """

    def __init__(self, *args, jobs=4, **kwargs):
        self.jobs = jobs
        super().__init__(*args, **kwargs)

    def map(self, object):
        return super().map(__import__('joblib').delayed(object))

    def __call__(self, *args, **kwargs):
        return __import__('joblib').Parallel(self.jobs)(
            super().__call__(*args, **kwargs))

    __truediv__ = map


# In[75]:

a = an = FunctionFactory(Function)
flip = FunctionFactory(Flip)
parallel = FunctionFactory(Parallel)
star = FunctionFactory(Star)
do = FunctionFactory(Do)
preview = FunctionFactory(Preview)
x = CanonicalFactory(Canonical)
juxt = FunctionFactory(Juxtaposition)
ifthen = FunctionFactory(IfThen)

# In[82]:


class store(dict):
    @property
    def __self__(self):
        return self.__call__.__self__

    def __init__(self, real=None, *args):
        self.real = Composition() if real is None else real
        super().__init__(*args)

    def __call__(self, *args, **kwargs):
        self[args] = self.real(*args, **kwargs)
        return self[args]

    def __getitem__(self, item):
        if not isinstance(item, tuple): item = item,
        return super().__getitem__(item)


# In[83]:


class cache(store):
    def __call__(self, *args, **kwargs):
        if args not in self:
            return super().__call__(*args, **kwargs)
        return self[args]


# In[84]:

if __name__ == '__main__':
    print(__import__('doctest').testmod())

# In[85]:

if __name__ == '__main__':
    from IPython import get_ipython
    get_ipython().system('jupyter nbconvert --to python composites.ipynb')
    get_ipython().system('yapf -i composites.py')
    get_ipython().system('flake8 composites.py')
    get_ipython().system('pyreverse -o png -pcomposites -fALL composites')
    get_ipython().system('pyreverse -o png -pcomposites.min composites')
    get_ipython().system('pydoc -w composites')

# In[86]:

f = cache(range)

# In[87]:

f(10)
