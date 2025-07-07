from setuptools import setup, find_packages

setup(
    name="ml",
    version="0.1.0",
    packages=find_packages(),  
    install_requires=[
        "numpy",
        "matplotlib"
    ],
    author="Jonatan Prepuk",
    url="https://github.com/jonatanoprepuk/ml",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ],
)