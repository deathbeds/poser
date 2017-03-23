
# coding: utf-8

# In[1]:

from fidget import _xx, _x, x_, call
from toolz.curried import identity
compositions = [_xx, _x, x_]


# In[2]:

def test_syntax(_y, functions):
    assert (
        tuple(_y[func] for func in functions)
        == tuple(_y.pipe(func) for func in functions) 
        == tuple(_y>>func for func in functions) 
    )

def test_exception(_y):
    assert type(_y.excepts(TypeError)) is not type(_y)
    assert _y.excepts(TypeError)._functions[0].exc        == (_y | TypeError)._functions[0].exc

def test_types(_y):
    assert bool(_y(10)**int)
    assert not bool(_y(10)**str)
    assert bool(_y(10)**(lambda *x: len(x)==1))
    if _y is not _xx:
        assert bool(_y(10, 10)**(int, int))
        assert not bool(_y(10, 10)**(int, str))
        assert not bool(_y(10, 10)**(lambda *x: len(x)==1))

def test_do(_y, function=identity):
    assert (_y << identity >> range)._do
    assert not (_y >> identity << range)._do
    
    assert (_y << identity >> range) == (_y.do(identity)[range])
    assert (_y >> identity << range) == (_y[identity].do(range))


# In[13]:

if __name__ == '__main__':
    for _y in compositions:
        assert bool(_y)
        assert type(_y()) is not type(_y)
        assert type(_y[:]) is not type(_y)
        test_syntax(_y, [range, list])
        test_do(_y)
        test_exception(_y)
        test_types(_y)

        assert _x(1,2,3) == _xx([1,2,3])
        assert _x(tuple([1,2,3])) == _xx(1,2,3)

        assert len(_xx(1,2,4)._args)==1
        assert len(_x(1,2,4)._args)==len(_xx(1,2,4)._args[0])==3
        
        assert isinstance(_x[[1, 2, 3]](), list)
        assert isinstance(_x[1, 2, 3](), tuple)
        assert isinstance(_x[{1, 2, 3}](), set)
        assert isinstance(_x[{'a': 1, 'b': 2, 'c': 3}](), dict)
        
        print("""Completed tests for {}""".format(type(_y).__name__))

