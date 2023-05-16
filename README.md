# Age Prediction from Facial Images using Deep CNNs

![Tests](https://img.shields.io/github/actions/workflow/status/MoritzM00/AgePrediction-UTKFace/test_deploy.yaml?style=for-the-badge&label=Test%20and%20Deploy)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white&style=for-the-badge)][pre-commit]
[![Black](https://img.shields.io/static/v1?label=code%20style&message=black&color=black&style=for-the-badge)][black]
![License](https://img.shields.io/github/license/MoritzM00/AgePrediction-UTKFace?style=for-the-badge)

[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black

## Introduction

With the provided CLI, you can train several CNNs, from custom to pretrained, on the UTKFace dataset. The dataset is available [here](https://susanqq.github.io/UTKFace/). Besides the relatively small UTK Face Dataset, it is also possible to use the much larger B3F-Dataset (B3FD), which is around 6GB in size.

## Quickstart

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

## CLI Usage

Run `train --help` to see a quick overview for the most important options. This will also list all default values.

If you run `train` without any options, it will use the default values for all options which are defined in `src/conf/`. You can override these values by passing them as arguments to the CLI.

- dataset: either `utkface` or `b3fd`
- lr_schedule: either `cosine_decay` or `exponential_decay`
- model: many options available, e.g. `efficientnetv2` or `baselinecnn`. See `train --help` for all options
- optimizer: either `adam` or `adamw`

### Running K-Fold Cross Validation

Specify the number of folds, e.g. `train.cv_folds=5` to train with 5-Fold Cross Validation. This will train `K` models and evaluate them on the test set. The final score test score is calculated on a hold-out test set (different from the test set within the Cross Validation) using the ensemble prediction (average the output of all `K` models).

If `train.cv_folds=null` (default) it will use a ordinary train-test-split, as specified with `train.test_size` (default: `0.2`).
This test size determines the size of the hold-out test set, which is used to calculate the final test score in cross validation.

### Options regarding the Model

For most pretrained models, you can specify the version/size of the model seperately.
E.g. for `efficientnetv2` you can specify `model.version=B0` to use the B0 version of the model. See [here](https://keras.io/api/applications/) for a detailed specification of the pretrained models on Keras.

#### Finetuning a pretrained model

The pretrained models are trained in two-stage fashion as it is described on Keras [here](https://keras.io/guides/transfer_learning/).

1. Train only the top layers of the model with a larger learning rate.
2. Fine-tune the whole (or a part of the) model with a smaller learning rate.

This behavior is determined based on the settings of
`model.freeze_base`, `model.finetune_base` and `model.num_finetune_layers`.

- `freeze_base`: If `true`, it will freeze the base model (default: `true`). This means that only the top layers are trained.
- `finetune_base`: If `true`, it will fine-tune the base model. This means that the whole model is trained in the second stage. This setting depends on the model in use.
- `num_finetune_layers`: The number of layers to fine-tune. If `all`, it will fine-tune the whole (base) model. If not `null` it must be an integer greater 0. E.g. use `model.num_finetune_layers=20` to finetune the top 20 layers of the **base** model (ignores the top layers outside of the base model)

.. warning:: Using an integer to specify the number of layers to fine-tune does not work currently. Use `all` instead.

#### Switching out the top-layer-architecture

You can specify the top layer architecture with
`model.top_layer_architecture._target_`. This must point to valid reference in `src.top_layer_architectures`

For example: The resnet top is specified via `model.top_layer_architecture._target_='src.top_layer_architectures.resnet_top'`

Available top layer architectures are

- `src.top_layer_architectures.resnet_top` (usually performs well)
- `src.top_layer_architectures.pass_through`
- `src.top_layer_architectures.vgg_top`
- `src.top_layer_architectures.fully_connected`
- `src.top_layer_architectures.fully_connected_with_dropout`
- `src.top_layer_architectures.conv_with_fc_top` (not recommended)

### Other important Options

This is an overview of other important options to configure the training process.

#### Training-related Options

- `train.target_size`: Specify the size of the images (default: `[150, 150]`). Note that some models require a minimum size. For `VGGFace` this must be at least `[224, 224]`. For `SENet50` it must be at least `[197, 197]`.
- `train.cache_dataset`: If `True`, it will cache the dataset in memory. This is recommended for the UTKFace dataset, but not for B3FD, because it is too large.
- `train.batch_size`: The batch size for training. Default: `32`
- `train.epochs`: The number of epochs to train. Default: `150`
- `lr_schedule.stage_1.initial_learning_rate`: The initial learning rate for the first stage of training (and `stage_2` analogously)

#### Data Augmentation

`augment.active`: If `True`, it will use data augmentation. Default: `True`. You can specify the augmentation options in `augment.factors`. E.g. `augment.factors.random_rotation=0.1` to specify the rotation range or `augment.factors.random_rotation=null` to disable only the rotation.

**Available augmentations**:

- `random_rotation`: Randomly rotate the image by a given angle
- `random_brightness`: Randomly change the brightness of the image
- `random_translation`: Randomly translate th image
- `random_flip`: Randomly flip the image horizontally or vertically (or both)

#### Callbacks

- `callbacks.early_stopping_patience`: Set the early stopping patience (default: `5`)
- `callbacks.model_ckpt`: If `true`, it will save the best model checkpoint after each epoch. Default: `false`
- `callbacks.with_wandb_ckpt`: If `true`, it will save the best model checkpoint to Weights & Biases. Default: `false`.
- `callbacks.visualize_predictions`: If `true`, it will visualize the predictions of the model on the validation set after each epoch. Default: `false`. This requires wandb to be active.
-

#### Miscellanous

- `wandb.mode`: either `online` or `offline`. If `online`, it will log to Weights & Biases. If `offline`, it will still use Weights & Biases, but it does not require an account.
- `wandb.project`: The name of the Weights & Biases project. Default: `UTKFace-v2`

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
