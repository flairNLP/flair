from setuptools import setup, find_packages

with open('requirements.txt') as f:
   requirements = f.read().split('\n')

setup(
   name='flair',
   version='0.1',
   description='A very simple framework for state-of-the-art NLP',
   author='Alan Akbik',
   author_email='alan.akbik@zalando.de',
   url='https://github.com/zalandoresearch/flair',
   packages=find_packages(exclude='test'),  #same as name
   install_requires=requirements
)
