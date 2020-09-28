from distutils.core import setup

setup(
    name='pyChemometrics',
    version='0.13.7',
    packages=['pyChemometrics'],
    url='https://github.com/Gscorreia89/pyChemometrics/',
    documentation='http://pychemometrics.readthedocs.io/en/stable/',
    license='BSD 3-Clause License',
    author='Gonçalo Correia',
    author_email='gscorreia89@gmail.com',
    setup_requires=['wheel'],
    description='The pyChemometrics provides objects which wrap pre-existing '
                'scikit-learn PCA and PLS algorithms and adds model assessment metrics and functions '
                'common in the Chemometrics literature.'
)
