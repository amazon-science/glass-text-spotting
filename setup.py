# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="glass-text-spotting",
    version="0.1",
    author="Shahar Tsiper",
    author_email="tsiper@amazon.com",
    license='Apache License 2.0',
    description="GLASS: Global to Local Scene-Text Spotting (ECCV'22)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/amazon-research/glass-text-spotting",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)