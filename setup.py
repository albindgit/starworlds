from setuptools import setup, find_packages

setup(name='starworlds',
      version='1.0',
      packages=find_packages(),
      install_requires=[
          'pyyaml',
          'numpy',
          'scipy',
          'matplotlib',
          'shapely'
      ]

)
