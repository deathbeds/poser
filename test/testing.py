# coding: utf-8

from fidget import *
from toolz.curried import *
from collections import OrderedDict
from six import PY34
# from IPython import get_ipython
# get_ipython().ast_node_interactivity = 'all'

_x(10, 20) != _x(10)
_x(10, 20) == _x(10, 20)
x_ & _xx[:] & call(20, 10) == _xx(10, 20) >> call

_x(10, 20)
_x(10, 20)()
_x[:](10, 20)
_x[_x[:]](10, 20)
_x[_x[_x[:]]](10, 20)

_x[_x[_x[_x[_x[_x[_x[_x[_x[_x[range]]]]]]]]]](10, 30, 2)
_x[_x[_x[_x[_x[_x[_x[_x[_x[x_[range]]]]]]]]]](10, 30, 2)
x_[_x[_x[_x[_x[_x[_x[_x[_x[x_[range]]]]]]]]]](10, 30, 2)

_x[_x[_x[_x[_x[_x[_x[_x[_x[_x(10, 30, 2)[range]]]]]]]]]]()
_x[_x[_x(10, 30, 2)[_x[_x[_x[_x[_x[_x[_x[range]]]]]]]]]]()

_x()(10)
_x()(10, 20)
_x()(10, 20, 30)

_xx()(10)
_xx()(10, 20)
_xx()(10, 20, 30)

x_()(10)
x_()(10, 20)
x_()(10, 20, 30)

_x[_xx()] >> call(10, 20, 30)
x_[_xx()] >> call(10, 20, 30)
_xx[x_()] >> call(10, 20, 30)
_xx[_x()] >> call(10, 20, 30)

_x >> range >> [list, len, type] >> list >> call(10)

_x >> range >> [list, len, type] >> call(10)
_x + range + [list, len, type] + call(10)
_x & range & [list, len, type] & call(10)
_x + range & [list, len, type] & call(10)
_x.pipe(range).pipe([list, len, type])(10)

_x & [type, isiterable, callable] & [type, identity] & call(10)
_x & {type, isiterable, callable} & [type, identity] & call(10)
_x & {
    'foo': type,
    isiterable: isiterable,
    'bar': callable
} & [type, identity] & call(10)
_x & OrderedDict((('foo', type),
                  ('bar', callable))) & [type, identity] & call(10)

_x[(range(i) for i in range(10))]
_x >> (range(i) for i in range(10))
_x[(range(i) for i in reversed(range(10)))].map(len)[list]()

import pandas as pd
import random

if PY34:
    _x >> {
        'data': _x[range].map(_x[[]][_xx >> random.random]) >> enumerate
        >> list,
        'index': _x[range][reversed][list],
        'columns': ['i', 'value'],
    } >> (_xx[pd.DataFrame]) >> call(3)

if PY34:
    _x >> [
        _x[range].map(_x[[]][_xx >> random.random]) >> enumerate >> list,
        _x[range][reversed][list],
        ['i', 'value'],
    ] >> (_xx[pd.DataFrame]) >> call(3)

from jinja2 import Environment
from IPython.display import Markdown

import builtins

env = _x[Environment()]()
env.globals.update(
    _x=_x,
    **merge(
        keyfilter(_x.first().eq('_').not_(), globals()),
        vars(builtins), vars(operator)))
tpl = env.from_string("""
# This example introduces `{{_x[str.upper][call(title)]}}`

`{{_x[str.upper][call(title)]}}` is as easy as {{_x[range].map(add(1)).map(str)[', '.join][call(i)]}}.

Maybe this could be literate programming?
""")

_x(i=3, title='fidget') >> tpl.render >> Markdown >> call

# ### polymorphisms

(_x + range + (_x + [list, len, type]) + list)(10)

(_x - range - (_x - [list, len, type]) - list)(10)

(_x >> range >> (_x >> [list, len, type]) >> list)(10)

(_x[range][_x[list, len, type]][list])(10)

(_x.pipe(range).pipe(_x[list, len, type]).pipe(list))(10)

_y[range][type](10)
_y >> range >> type >> call(10)
_x[_y >> range >> type] >> list >> call(10)
_x[_y >> range >> type] >> tuple >> call(10)

(_x ^ x_(int)[isinstance]) >> call(10)
(_x**x_(int)[isinstance]) >> call(10)
(_x ^ x_(int)[isinstance]) >> call('10')
(_x**x_(int)[isinstance]) >> call('10')

_xx[type, len](10, 30, 2)
_xx[_xx[range]](10, 30, 2)
_xx[:](10, 30, 2)
_xx[_xx[:]](10, 30, 2)
_xx[_xx[_xx[:]]](10, 30, 2)

f = _x & range ^ TypeError
f >> call(10)
f << (_x & type & print) >> call('10')
f | _x & str.upper ^ str
f << (_x & type & print) >> call(10)
f >> call('asdf')

f = _x(10, 20) >> range >> [len, list, type]
_x[range].map(f)[list].call(1, 4)

type({i: [str, int, float][i] for i in range(3)})

f = _y[1, str][type, type][str, int]['float', float]
_x[f][list](10)
_x[f][dict](10)
