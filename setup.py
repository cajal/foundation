#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="foundation",
    version="0",
    description="Foundation Pipeline",
    author="Eric Y. Wang",
    author_email="eric.wang2@bcm.edu",
    packages=find_packages(),
    install_requires=[
        "datajoint>=0.12.9,<0.13.0",
    ],
)
