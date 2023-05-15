# Age Prediction on the UTK Face Dataset

![Tests](https://img.shields.io/github/actions/workflow/status/MoritzM00/AgePrediction-UTKFace/test_deploy.yaml?style=for-the-badge&label=Test%20and%20Deploy)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white&style=for-the-badge)][pre-commit]
[![Black](https://img.shields.io/static/v1?label=code%20style&message=black&color=black&style=for-the-badge)][black]
![License](https://img.shields.io/github/license/MoritzM00/AgePrediction-UTKFace?style=for-the-badge)

[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black

## Introduction

With the provided CLI, you can train several CNNs, from custom to pretrained, on the UTKFace dataset. The dataset is available [here](https://susanqq.github.io/UTKFace/). Besides the relatively small UTK Face Dataset, it is also possible to use the much larger B3F-Dataset (B3FD), which is around 6GB in size.

## CLI Usage

Before using the CLI, you need to install the package. Refer to the [Installation Instructions](#installation-instructions) for more information.

Then, you can use the CLI to train a model:

```bash
train <option>=<value> <option>=<value> ...
```

e.g. to train the EfficientNetV2B0-Model on the UTKFace dataset with default parameters, run:

```bash
train model=efficientnetv2 model.version=B0
```

By default it uses the UTKFace dataset. You can also train on B3FD, using `dataset=b3fd` as additional configuration.
Also note, that by default it logs to Weights & Biases. You can disable this by setting `wandb.mode=offline`.

## CLI Options

TODO

## Installation Instructions

1. Create a virtual environment with Python 3.9 or higher. It is highly recommended to use Conda because of the Tensorflow dependency.
2. Install the package with `pip install .` or `poetry install`

This is already provided via the `Makefile`. You can just run

```bash
make setup
```

to create a conda environment and install the package into it.

## Developer Guide

### Set up the environment

1. Install [Poetry](https://python-poetry.org/docs/#installation)
2. Set up the environment:

```bash
make setup
```

### Install new packages

To install new PyPI packages, run:

```bash
poetry add <package-name>
```

To add dev-dependencies, run:

```bash
poetry add <package-name> --group dev
```

### Documentation

The Documentation is automatically deployed to GitHub Pages.

To view the documentation locally, run:

```bash
make docs_view
```

## Credits

This project was generated with the [Light-weight Python Template](https://github.com/MoritzM00/python-template) by Moritz Mistol.

```

```
