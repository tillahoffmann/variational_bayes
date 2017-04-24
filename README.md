# variational_bayes [![Build Status](https://travis-ci.org/tillahoffmann/variational_bayes.svg?branch=master)](https://travis-ci.org/tillahoffmann/variational_bayes)

Skeleton for python projects using `conda`, py.test and TravisCI.

## Setup

This package can be used to set up a skeleton for a python package that uses a `conda` environment and supports
continuous integration testing using `py.test` running on TravisCI.

To set up your own project, you can download or clone this repository and execute the `setup_skeleton.py` script which
will ensure that all folders, files and contents are renamed.

```
usage: setup_skeleton [-h] [--reset] [--author AUTHOR] [--version VERSION]
                      [name]

positional arguments:
  name                  name of the python package

optional arguments:
  -h, --help            show this help message and exit
  --reset, -r           reset the repository to HEAD
  --author AUTHOR, -a AUTHOR
                        author of the package
  --version VERSION, -v VERSION
                        version of the package
```

Once you have executed `setup_skeleton.py` you can commit your code and push it to GitHub. To run continuous integration
testing, make sure that you have added the project to https://travis-ci.org for public projects or https://travis-ci.com
for private projects.
