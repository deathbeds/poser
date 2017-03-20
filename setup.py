
# coding: utf-8

# In[85]:

from os.path import join, dirname
import setuptools


def read(fname):
    with open(join(dirname(__file__), fname)) as f:
        return f.read()

from distutils.core import setup, Command
# you can also import from setuptools
print(setuptools.find_packages())

setuptools.setup(
    name="fidget",
    version="0.2.0",
    author="Tony Fast",
    author_email="tony.fast@gmail.com",
    description="A Test Harness in a DataFrame",
    license="BSD-3-Clause",
    keywords="IPython Magic Jupyter",
    url="http://github.com/tonyfast/fidget",
    py_modules=['fidget'],
#     long_description=read("readme.rst"),
    classifiers=[
        "Topic :: Utilities",
        "Framework :: IPython",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Intended Audience :: Developers",
        "Development Status :: 3 - Alpha",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Topic :: Software Development :: Testing",
    ],
    install_requires=[
#         "sklearn", "pandas", "jinja2", "bokeh"
    ],
    tests_require=[
#         'pytest', 'pytest-ipynb'
    ],
)

