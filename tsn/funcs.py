#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# .../neuralnetwork/utils/funcs.py

Functions for neural network classes.
"""

import pathlib

import numpy as np


def create_experiment_log_dir(root: str, parents: bool = True) -> str:
    root_path = pathlib.Path(root).resolve()
    child = (
        create_from_missing(root_path)
        if not root_path.exists()
        else create_from_existing(root_path)
    )
    child.mkdir(parents=parents)
    return child.as_posix()


def create_from_missing(root: pathlib.Path) -> pathlib.Path:
    return root / "0"


def create_from_existing(root: pathlib.Path) -> pathlib.Path:
    children = [
        int(c.name) for c in root.glob("*") if (c.is_dir() and c.name.isnumeric())
    ]
    if is_first_experiment(children):
        child = create_from_missing(root)
    else:
        child = root / increment_experiment_number(children)
    return child


def is_first_experiment(children: list[int]) -> bool:
    return len(children) == 0


def increment_experiment_number(children: list[int]) -> str:
    return str(max(children) + 1)


def shuffle_data(X, y, seed=None):
    """ Random shuffle of the samples in X and y """
    if seed:
        np.random.seed(seed)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]
