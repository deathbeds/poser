import toml, setuptools, pathlib

flit = toml.load('pyproject.toml')['tool']['flit']
metadata = flit['metadata']

module = __import__(metadata['module'])

setuptools.setup(
    name=metadata['module'],
    version=module.__version__,
    packages=[metadata['module']],
    classifiers=metadata['classifiers'],
    url=metadata['home-page'],
    author=metadata['author'],
    author_email=metadata['author-email'],
    description=module.__doc__,
    long_description=pathlib.Path(metadata['description-file']).read_text(),
    long_description_content_type="text/markdown",
    install_requires=metadata['requires']
)