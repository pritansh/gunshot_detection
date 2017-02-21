from setuptools import setup

setup(name='Gunshot Detection using NN',
      version='0.1',
      description='Gunshot Detection in Python',
      url='https://github.com/pritansh/gunshot_detection',
      author='Pritansh',
      author_email='pritansh.chaudhary@gmail.com',
      license='MIT',
      packages=['gunshot_detection'],
      install_requires=[
          'setuptools',
          'tensorflow',
          'librosa',
          'numpy',
          'matplotlib',
          'setuptools'
      ])
