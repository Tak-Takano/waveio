from setuptools import setup, find_packages

setup(
    name='waveio',
    version='1.0.0',
    author='Takano, Takeshi',
    author_email='takano.tak@gmail.com',
    description='interface of wave files',
    packages=['waveio'],
    package_dir={'waveio': 'src'},
    install_requires=["pandas", "numpy", "pathlib", "python-box"]
)