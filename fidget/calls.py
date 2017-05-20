# coding: utf-8

# > `fidget` uses the python data model to compose higher-order functions.
# 
# ---

try:
    from .classes import Functions, Composer, Attributes, Compose
    from .callables import flipped, do, step, starred, excepts, ifnot, ifthen
except Exception as e:
    from classes import Functions, Composer, Attributes, Compose
    from callables import flipped, do, step, starred, excepts, ifnot, ifthen

from functools import wraps
from toolz.curried import (isiterable, flip, complement, interpose, groupby,
                           partial, reduce, filter, map)

__all__ = ['Flips', 'Stars', 'Does', 'Maps', 'Filters', 'Groups', 'Reduces']
_calls_ = (flipped, starred, do, map, filter, groupby, reduce)


class Calls(Composer, Attributes):
    def __xor__(self, object):
        self, _isinstance = self[:], flip(isinstance)  # noqa: F823
        if not isiterable(object) and isinstance(object, type):
            object = (object, )
        if isiterable(object):
            if all(map(_isinstance(type), object)) and all(
                    map(flip(issubclass)(BaseException), object)):
                self.function = Compose(excepts(object, self.function))
                return self

            if all(map(_isinstance(BaseException), object)):
                object = tuple(map(type, object))

            if all(map(_isinstance(type), object)):
                object = _isinstance(object)

        self.function = Compose([ifthen(Compose([object]), self.function)])
        return self

    def __or__(self, object):
        self = self[:]
        self.function = Compose([ifnot(self.function, Compose([object]))])
        return self

    def __and__(self, object):
        self = self[:]
        self.function = Compose([step(self.function, Compose([object]))])
        return self

    def __pos__(self):
        return self[bool]

    def __neg__(self):
        return self[complement(bool)]

    def __lshift__(self, object):
        return Does()[object] if self._factory_ else self[do(object)]

    def __round__(self, n):
        self.function.function = list(interpose(n, self.function.function))
        return self

    __invert__, __pow__ = Functions.__reversed__, __xor__
    __mul__ = __add__ = __rshift__ = __sub__ = Composer.__getitem__


_attribute_ = "__{}{}__".format

for attr, method in [['call'] * 2, ['do', 'lshift'], ['pipe', 'getitem'],
                     ['ifthen', 'xor'], ['step', 'and'], ['ifnot', 'or']]:
    setattr(Calls, attr, getattr(Calls, _attribute_('', method)))


def operator(attr,
             method,
             partialize=False,
             juxtapose=False,
             force=False,
             cls=Calls):
    if force or not hasattr(cls, attr):

        def operator(self, *args, **kwargs):
            if len(args) is 1 and juxtapose and not partialize:
                args = (Compose([args[0]]), )

            return self[method(*args, **kwargs) if partialize else partial(
                method, *args, **kwargs) if args or kwargs else method]

        setattr(cls, attr, getattr(cls, attr, wraps(method)(operator)))


for attr, method in [('__matmul__', groupby), ('__div__', map), (
        '__truediv__', map), ('__floordiv__', filter), ('__mod__', reduce)]:
    operator(attr, method, True) or setattr(Calls, method.__name__,
                                            getattr(Calls, attr))


def fallback(attr):
    def fallback(right, left):
        right = right[:]
        return getattr(type(right)()[left], attr)(right)

    return wraps(getattr(Calls, attr))(fallback)


for attr in [
        'add', 'sub', 'mul', 'matmul', 'div', 'truediv', 'floordiv', 'mod',
        'lshift', 'rshift', 'and', 'xor', 'or', 'pow'
]:
    setattr(Calls,
            _attribute_('i', attr), getattr(Calls, _attribute_('', attr)))
    setattr(Calls, _attribute_('r', attr), fallback(_attribute_('', attr)))

for name, func in (('Flips', flipped), ('Stars', starred), ('Does', do), (
        'Maps', map), ('Filters', filter), ('Groups', groupby), ('Reduces',
                                                                 reduce)):
    locals().update({
        name:
        type(name, (Calls, ), {'_decorate_': staticmethod(func)})
    })

__all__ += ['Calls']
