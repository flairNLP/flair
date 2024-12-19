from pathlib import Path

from setuptools import find_packages, setup

required = Path("requirements.txt").read_text(encoding="utf-8").split("\n")

setup(
    name="flair",
    version="0.14.0",
    description="A very simple framework for state-of-the-art NLP",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Alan Akbik",
    author_email="alan.akbik@gmail.com",
    url="https://github.com/flairNLP/flair",
    packages=find_packages(exclude=["tests", "tests.*"]),  # same as name
    license="MIT",
    install_requires=required,
    extras_require={
        "word-embeddings": ["gensim>=4.2.0", "bpemb>=0.3.5"],
    },
    include_package_data=True,
    python_requires=">=3.9",
)
