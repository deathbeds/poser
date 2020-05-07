import pluggy, multipledispatch, dataclasses, decorator, contextlib, abc, collections, functools, inspect
_object_dir = dir(object)    

_impl, method_spec, _manager = pluggy.HookimplMarker('methodz'), pluggy.HookspecMarker('methodz'), pluggy.PluginManager('methodz')

__all__ = "method", "method_spec"

class Spec:
    @method_spec
    @_impl
    def register(input):  _impl(input, tryfirst=True)
    @method_spec
    @_impl
    def unregister(input): 
        plugin = _manager.get_plugin(input)
        plugin and _manager.unregister(plugin)
    @method_spec(firstresult=True)
    @_impl(trylast=True)
    def __bop__(attr, a, b):
        return getattr(super(type(a), a).__thisclass__, attr)(a, b)
    @method_spec(firstresult=True)
    @_impl(trylast=True)
    def __uop__(attr, a): 
        return getattr(super(type(a), a).__thisclass__, attr)(a)

_manager.add_hookspecs(Spec)
_manager.register(Spec)

def __op__(attr, a, *b):
    if b:
        b, *_ = b
        return _manager.hook.__bop__(attr=attr, a=a,b=b)
    return _manager.hook.__uop__(attr=attr, a=a)



def get_annotations(input): 
    return [x.annotation for x in inspect.signature(input).parameters.values() if x.annotation is not inspect._empty]



_methods = "add sub pow mul matmul mod lshift rshift abs round iter".split()
_methods = list(map("__{}__".format, _methods))

class HookBase(abc.ABCMeta):
    def __getattr__(cls, attr): return functools.partial(__op__, attr)
    def __dir__(self): return _methods 
class Hook(metaclass=HookBase):
    def register(input):
        annotations = get_annotations(input)
        if annotations:
            try:
                print(set)
                setattr(annotations[0], input.__name__, selfish(getattr(Hook, input.__name__)))
            except TypeError: ...
_manager.register(Hook)



namespace = {}
class MultipleDispatch:
    @_impl
    def register(input): multipledispatch.dispatch(*get_annotations(input),namespace=namespace)(input)

    namespace = {}
    @_impl
    def __bop__(attr, a, b):            
        return MultipleDispatch.__op__(attr, a, b)
    @_impl
    def __uop__(attr, a):
        return MultipleDispatch.__op__(attr, a)
        
    def __op__(attr, *a):            
        if attr in namespace:
            try:
                return namespace.get(attr)(*a)
            except NotImplementedError: ...

        
_manager.register(MultipleDispatch)



def selfish(callable):
    @functools.wraps(callable)
    def call(self, *a):
        return callable(self, *a)
    return call



def method(input):
    if isinstance(input, type):
        for key in _methods:
            if hasattr(input, key): method(getattr(input, key))
        try: _manager.register(input)
        except ValueError: ...
    elif inspect.isfunction(input):
        _manager.hook.register(input=input)
        annotations = get_annotations(input)
        if annotations:
            try:
                setattr(annotations[0], input.__name__, selfish(getattr(Hook, input.__name__)))
            except TypeError: ...

    return input