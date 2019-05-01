import setuptools
with open("README.md","r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="poissonblend",
    version="0.0.1",
    author="James Noeckel",
    author_email="jamesn8@cs.washington.edu",
    description="tools for poisson image editing",
    long_description_content_type="text/markdown",
    url="https://github.com/ShnitzelKiller/PoissonBlending",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: MIT License",
        "Operating System :: OS Independent"
    ]
)
