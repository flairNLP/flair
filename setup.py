from setuptools import setup, find_packages

setup(
    name='flair',
    version='0.1.1',
    description='A very simple framework for state-of-the-art NLP',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author='Alan Akbik',
    author_email='alan.akbik@zalando.de',
    url='https://github.com/zalandoresearch/flair',
    packages=find_packages(exclude='test'),  # same as name
    license='MIT',
    install_requires=[
        'torch==0.4.0',
        'awscli==1.14.32',
        'gensim==3.4.0',
        'typing==3.6.4',
        'tqdm==4.23.4',
        'segtok==1.5.6'
    ],
    include_package_data=True,
    python_requires='>=3.6',
)
