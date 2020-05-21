from setuptools import setup
# from distutils.core import setup
from anko import __version__

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name='anko',
    packages=['anko'],
    version=__version__,
    license='MIT',
    description='Toolkit for performing anomaly detection algorithm on time series.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='tao-lin',
    author_email='tanlin2013@gmail.com',
    url='https://github.com/tanlin2013/anko',
    download_url='https://github.com/tanlin2013/anko/archive/v%s.tar.gz' % __version__,
    keywords=['statistics', 'time series', 'anomaly detection'],
    install_requires=[
        'numpy>=1.16.4',
        'scipy>=1.2.1',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',  # Define that your audience are developers
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: MIT License',  # Pick a license
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
