'''This file helps us build our project as a package. 
Which can be easily installed and used by other users.'''

from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:
    '''
    This function will return list of requirements from requirement.txt file
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    
    return requirements



setup(
    name = 'firstMLproject',
    version = '0.0.1',
    author = 'Ramakanth Sharma',
    author_email = 'ramakanthsharma214@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)