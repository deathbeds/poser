from pathlib import Path
import setuptools

name = "poser"

__version__ = "0.2.1"

here = Path(__file__).parent

setup_args = dict(
    name=name,
    version=__version__,
    author="deathbeds",
    author_email="tony.fast@gmail.com",
    description="Dysfunctional programming in Python with all the side effects.",
    long_description=(here / "readme.md").read_text(),
    long_description_content_type='text/markdown',
    url="https://github.com/deathbeds/poser",
    python_requires=">=3.6",
    license="BSD-3-Clause",
    install_requires=[
        "dataclasses",
        "toolz",
        "requests",
        "joblib"
    ],
    include_package_data=True,
    py_modules=["poser"],
    classifiers=(
        "Development Status :: 4 - Beta",
        "Framework :: IPython",
        "Framework :: Jupyter",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
    ),
    zip_safe=False,
)

if __name__ == "__main__":
    setuptools.setup(**setup_args)
