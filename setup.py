import toml, setuptools, pathlib, flit

flit_data = toml.load('pyproject.toml')['tool']['flit']
metadata = flit_data['metadata']

description, version = flit.common.get_docstring_and_version_via_import(flit.common.Module(metadata['module']))

setuptools.setup(
    name=metadata['module'],
    version=version,
    packages=setuptools.find_packages(),
    classifiers=metadata.get('classifiers') or [],
    url=metadata['home-page'],
    author=metadata['author'],
    install_requires=metadata.get('requires', []),
    author_email=metadata['author-email'],
    description=description,
    long_description=pathlib.Path(metadata['description-file']).read_text(),
    long_description_content_type="text/markdown",
    extras_require=metadata.get('requires-extra', {}),
    #keywords=metadata.get('keywords', ),
    #console_scripts=metadata.get('scripts', []),
)