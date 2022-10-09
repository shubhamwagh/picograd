"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

from setuptools import setup, find_packages

with open("README.md") as f:
    README = f.read()

setup(
    name='picograd',
    version='1.0.0',
    description='A lightweight machine learning framework',
    url='https://github.com/shubhamwagh/picograd',
    long_description=README,
    long_description_content_type='text/markdown',
    license="MIT",
    platforms=['Ubuntu 20.04', 'Windows'],
    author='Shubham Wagh',
    author_email='shubhamwagh48@gmail.com',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        'Programming Language :: Python :: 3 :: Only',
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Intended Audience :: Developers",
    ],
    keywords='picograd, autograd, backprop, nn, graph',
    packages=find_packages(),
    python_requires='>=3.6, <4',
    install_requires=[],
)
