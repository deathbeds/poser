from subprocess import call
from .composition import poser, λ
import toolz, operator, builtins
from operator import attrgetter, itemgetter, methodcaller
from .functools import istype

__all__ = "poser", "λ"

F = dict(kind="flipped")


def is_public(str):
    return x[0] == "_"


import pathlib
import builtins, copy, io, typing, types, dataclasses, abc, statistics, itertools, json, math, string, random, re, glob, ast, dis, tokenize
import fnmatch
poser.add_method(istype, "issubclass", **F)
poser.add_method(isinstance, "isinstance", **F)
poser.add_method(pathlib.Path)
normal_modules = (toolz, builtins, builtins, copy, io, typing, types, dataclasses, abc)
normal_modules += (statistics, itertools, json, math, string, random)
normal_modules += (re, glob, ast, dis, tokenize)

for m in normal_modules:
    for k, v in vars(m).items():
        if k.startswith("_"):
            continue
        if k in {"isinstance", "issubclass"}:
            continue
        if callable(v):
            poser.add_method(v, k)

flipped_modules = operator, str, dict, list, pathlib.Path, fnmatch

for m in flipped_modules:
    for k, v in vars(m).items():
        if k.startswith("_"):
            continue
        if m is operator:
            if v in {attrgetter, itemgetter, methodcaller}:
                poser.add_method(v, k, kind="instance")
                continue
        elif callable(v):
            poser.add_method(v, k, kind="flipped")
