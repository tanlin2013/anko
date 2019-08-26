from distutils.core import setup

setup(
  name = 'tamago',
  packages = ['tamago'],
  version = '0.0.1',
  license='MIT',
  description = 'Toolkit for performing anomaly detection algorithm on time series.',
  author = 'tao-lin',
  author_email = 'tanlin2013@gmail.com', 
  url = 'https://github.com/tanlin2013/tamago',
  download_url = 'https://github.com/tanlin2013/tamago/archive/v0.0.1.tar.gz',
  keywords = ['statistics', 'time series', 'anomaly detection'],
  install_requires=[
          'numpy==1.16.4',
          'scipy==1.2.1',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Scientific/Engineering :: Mathematics',
    'License :: OSI Approved :: MIT License',   # Pick a license
    'Programming Language :: Python :: 3',      # Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)

