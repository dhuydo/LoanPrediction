from setuptools import find_packages, setup


def get_requirements(path):
    '''
    This function will return the list of requirements
    '''
    requirements = []
    
    with open(path) as file:
        requirements = file.readlines()
        requirements = [req.replace('\n', ' ') for req in requirements]
        if '-e .' in requirements:
            requirements.remove('-e .')
    
    return requirements

setup(
    name='loan-project',
    version='0.0.1',
    author='Duc Huy',
    author_email='ncdhuy.jul04@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt') 
)