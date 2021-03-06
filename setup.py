from setuptools import setup

setup(name='SemanticProtoRoleLinkingModel',
      version='0.1dev0',
      description='Semantic Proto-Role Linking Model',
      url='https://github.com/aaronstevenwhite/SemanticProtoRoleLinkingModel',
      author='Aaron Steven White',
      author_email='aswhite@jhu.edu',
      license='MIT',
      packages=['sprolim'],
      install_requires=['numpy',
                        'scipy',
                        'pandas',
                        'theano'],
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)
