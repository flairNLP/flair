from setuptools import setup, find_packages

setup(
   name='flair',
   version='0.1',
   description='A useful module',
   author='Alan Akbik',
   author_email='alan.akbik@zalando.de',
   url='https://github.com/zalandoresearch/flair',
   packages=find_packages(exclude='test'),  #same as name
   install_requires=['torch==0.4.0', 'awscli==1.14.32', 'gensim==3.4.0', 'typing==3.6.4', 'tqdm']
)
