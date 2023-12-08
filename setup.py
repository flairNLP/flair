from pathlib import Path

from setuptools import find_packages, setup

required = Path("requirements.txt").read_text(encoding="utf-8").split("\n")

setup(
    name="flair",
    version="0.13.1",
    description="A very simple framework for state-of-the-art NLP",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Alan Akbik",
    author_email="alan.akbik@gmail.com",
    url="https://github.com/flairNLP/flair",
    packages=find_packages(exclude="tests"),  # same as name
    license="MIT",
    install_requires=required,
    include_package_data=True,
    python_requires=">=3.8",
)
