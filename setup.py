import io
import os
import re

from setuptools import find_packages
from setuptools import setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding="utf-8") as fd:
        return re.sub(text_type(r":[a-z]+:`~?(.*?)`"), text_type(r"``\1``"), fd.read())


requirements = [
    # use environment.yml
]


setup(
    name="eml20_session02_training",
    version="0.0.1",
    url="https://github.com/akshatthakar/EML20_session02_training",
    author="Akshat Thakar",
    author_email="akshat.thakar@gmail.com",
    description="assignment for training based on pytorch lightening, hydra, cog, cookiecutter",
    long_description=read("README.rst"),
    packages=find_packages(exclude=("tests",)),
    entry_points={
        "console_scripts": [
            "eml20_session02_training=eml20_session02_training.cli:cli"
        ]
    },
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
    ],
)
