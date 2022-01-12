from setuptools import setup, find_packages
from pkg_resources import resource_filename
from pathlib import Path


with (Path(__file__).parent / 'README.md').open() as readme_file:
    readme = readme_file.read()

setup(
    name='FASER-alignment',
    packages=find_packages(),
    url="",
    author='Markus Tobias Prim',
    author_email='markus.prim@cern.ch',
    description='''
Prototype of the FASER alignment code.
''',
    install_requires=[
        'uproot3',
        'numpy',
        'pandas',
        'scipy',
        'matplotlib',
        'uncertainties',
        'tabulate',
    ],
    extras_require={
        "examples":  ['jupyterlab'],
    },
    long_description=readme,
    long_description_content_type='text/markdown',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License "
    ],
    license='MIT',
)
