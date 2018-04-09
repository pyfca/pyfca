#!/usr/bin/env python

#py.test --cov pyfca --cov-report term-missing

#sudo python setup.py bdist_wheel
#twine upload ./dist/*.whl

from setuptools import setup
import platform
import os, os.path

__version__ = '1.0'

def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname),encoding='utf-8') as f:
        return f.read().split('\n"""')[1]

long_description = '\n'.join(["pyfca\n=====\n\n"
,open('readme.rst').read()
])

setup(name = 'pyfca',
    version = __version__,
    description = 'pyfca - python formal concept analysis',
    license = 'MIT',
    author = 'Roland Puntaier',
    keywords=['Documentation'],
    author_email = 'roland.puntaier@gmail.com',
    url = 'https://github.com/pyfca/pyfca',
    classifiers = [
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6',
        'Intended Audience :: Information Technology'
        ],

    install_requires = [],
    extras_require = {'develop': ['pytest-coverage']},
    long_description = long_description,
    packages=['pyfca'],
    include_package_data=True,
    package_data={'pyfca':[]},
    zip_safe=False,
    tests_require=['pytest','pytest-coverage'],
    entry_points={
      },

    )

