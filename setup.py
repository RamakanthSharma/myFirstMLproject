'''This file helps us build our project as a package. 
Which can be easily installed and used by other users.'''

from setuptools import find_packages, setup

setup(
    name = 'firstMLproject',
    version = '0.0.1',
    author = 'Ramakanth Sharma',
    author_email = 'ramakanthsharma214@gmail.com',
    packages = find_packages(),
    install_requires = ['pandas', 'numpy', 'seaborn']
)