from setuptools import setup

setup(name='SPROLIM',
      version='1.1dev0',
      description='Semantic Proto-Role Linking Model',
      url='http://github.com/aaronstevenwhite/SPROLIM',
      author='Aaron Steven White',
      author_email='aswhite@jhu.edu',
      license='MIT',
      packages=['sprolim'],
      install_requires=['numpy',
                        'scipy',
                        'pandas',
                        'pymc',
                        'theano'],
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)
