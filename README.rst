========================
EML20_session02_training
========================
PyTorch deep learning project made easy.

.. contents:: Table of Contents
   :depth: 2

Folder Structure
================

::

  EML20_session02_training/
  │
  ├── eml20_session02_training/
  │    │
  │    ├── cli.py - command line interface
  │    ├── main.py - main script to start train/test
  │    │
  │    ├── data_loader/ - anything about data loading goes here
  │    │   └── cfar_datamodule.py
  │    │
  │    ├── model/ - models, losses, and metrics
  │    │   ├── cfar_model.py
  │    │
  │    │
  │    └── utils/
  │        ├── rich_utils.py - class for train logging
  │        ├── utils.py - class for Tensorboard visualization support
  │
  ├── logging.yml - logging configuration
  │
  ├── data/ - directory for storing input data
  │
  ├── Makefile - make file for build docker image and run
  │
  ├── Dockerfile - docker file
  │
  └── tests/ - tests folder


Usage
=====

Build docker image for training timm model with Pytorch Lightening using -

.. code-block::

make build

Run docker image for training timm model with Pytorch Lightening -
.. code-block::

make run

