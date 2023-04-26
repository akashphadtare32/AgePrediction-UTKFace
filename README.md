# Age Prediction on the UTK Face Dataset

![Tests](https://img.shields.io/github/actions/workflow/status/MoritzM00/AgePrediction-UTKFace/test_deploy.yaml?style=for-the-badge&label=Test%20and%20Deploy)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white&style=for-the-badge)][pre-commit]
[![Black](https://img.shields.io/static/v1?label=code%20style&message=black&color=black&style=for-the-badge)][black]
![License](https://img.shields.io/github/license/MoritzM00/AgePrediction-UTKFace?style=for-the-badge)

[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black

---

## Quick Start

Below you can find the quick start guide for development.

### Set up the environment

1. Install [Poetry](https://python-poetry.org/docs/#installation)
2. Set up the environment:

```bash
make setup
make activate
```

### Additional first-time setup

1. After setting up the environment, commit the `poetry.lock` file to your repository, so that the workflow on github can use it.
2. Enable [Pre-Commit CI](https://pre-commit.ci/) for your repository.
3. Enable **Github Pages** for your documentation.
   To do that, go to the _Settings_ tab of your repository and scroll down to the _GitHub Pages_ section.
   For the _Source_ option, select _GitHub Action_. Done!

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
