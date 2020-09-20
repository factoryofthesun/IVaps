from setuptools import setup, find_packages
from os import path, getcwd
import sys

here = path.abspath(path.dirname(__file__))
sys.path.insert(0, path.join(here, "mlisne"))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

requirements_path = path.join(getcwd(), "requirements.txt")
if not path.exists(requirements_path):
    this = path.dirname(__file__)
    requirements_path = path.join(this, "requirements.txt")
if not path.exists(requirements_path):
    raise FileNotFoundError("Unable to find 'requirements.txt'")
with open(requirements_path) as f:
    install_requires = f.read().splitlines()

setup(
    name="mlisne",
    version=mlisne.__version__,
    description="Machine Learning is Natural Experiment",
    url="https://github.com/factoryofthesun/mlisne",
    author="Richard Liu",
    author_email="guanzhi97@gmail.com",
    keywords=["treatment effect estimation", "causal machine learning"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=install_requires,
    license="Apache License",
    packages=find_packages(exclude=["examples", "tests", "docs"]),
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        "License :: OSI Approved :: Apache Software License",
    ],
    python_requires='>=3.5, <3.8',
)
