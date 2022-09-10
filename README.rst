========================
EML20_session02_training
========================
PyTorch lightening with hydra, cookies cutter template

.. contents:: Table of Contents
   :depth: 2

Folder Structure
================

::

  EML20_session02_training/
  │
  ├── eml20_session02_training/
  │    │
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
  ├── entrypoint.sh - entry point script with train and eval script execution for docker container
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

