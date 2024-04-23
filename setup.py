#!/usr/bin/env python3
import os
import subprocess
from setuptools import setup, find_packages

module_name = "chipon"

__dir__ = os.path.abspath(os.path.dirname(__file__))
version = subprocess.check_output(
    [
        "python3",
        os.path.join(
            __dir__,
            module_name,
            "__version__.py",
        ),
    ],
    encoding="utf8",
)

requirements = open("requirements.txt").read().strip().split("\n")
setup(
    name=module_name,
    packages=find_packages(),
    version=version,
    description="An infrastructure for implementing chip design flows",
    long_description=open(os.path.join(__dir__, "README.md")).read(),
    long_description_content_type="text/markdown",
    author="",
    author_email="",
    install_requires=requirements,
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Developers",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
    ],
    python_requires=">3.8",
)
