#!/usr/bin/env python3

import setuptools

import progresstools
long_description = progresstools.__doc__

setuptools.setup(
    name='progresstools',
    version='0.1.0',
    author='Mihail Georgiev',
    author_email='misho88@gmail.com',
    description='progress indicators',
    long_description=long_description,
    long_description_content_type='text/plain',
    url='https://github.com/misho88/progresstools',
    packages=setuptools.find_packages(),
    python_requires='>=3.8',
    py_modules=['progresstools']
)
