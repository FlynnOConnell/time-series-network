#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# neuralnetwork.py

Module (neuralnetwork): Classes/functions for NeuralNetwork module training.
"""

from __future__ import division

import logging
from enum import Enum, auto

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import (
    StratifiedShuffleSplit,
    RepeatedStratifiedKFold,
    GridSearchCV,
)
from sklearn.svm import SVC

from graphs.graph_utils.helpers import plot_learning_curve
from neuralnetwork.nn_utils._properties import _validate, _props
from neuralnetwork.nn_utils.datahandler import DataHandler
from neuralnetwork.nn_utils.scores import Scoring

logger = logging.getLogger(__name__)


# %%


class Stage(Enum):
    BASE = auto(int)
    SPLIT = auto(int)
    SCALE = auto(int)


class SupportVectorMachine(_validate, _props):
    def __init__(self, data, target, stage, cv=None):
        """
        Base class for svc.SVM neural network model to handle:
             a binary classification to [-1, 1] values, or 
             a multivariate classification to [-1, n-1]. 
        -Instances of StandardScaler() needed to scale data to a mean = 0 and stdev = 1. 
        -These attributes will be shared for each session.
        
        ..:Steps: 
            1) Split dataset into Train/Train-labels (X_train, Y_train) and Test/Test-labels
            2) Scale data to mean = 0 Stdev = 1,
    and encode labels to integers -1 to 1-n (n = len(labels))
            3) Optimize classifier (optional),
    tune hyperparameters with sklearn.GridSearchCV using default or 
                   custom parameters for C, Gamma, and anything else. 
            4) Fit the model to training data. 
            5) Predict test labels from fitten model.
            6) If scores are sufficient, use fitted model to predict labels from different eval dataset.
                
            Note: Each step can be performed independantly with custom data. 
        
        Parameters
        ----------
        data : pd.DataFrame | np.ndarray
            Features x Samples.
        target : pd.Series | np.ndarray
            Labels for data input.
        stage: str
            Stage of network: train, test, or eval.

        Returns
        -------
        None.

        """
        self.stage = stage
        self.traindata = DataHandler(data=data, target=target)
        self.scaler = preprocessing.StandardScaler()
        self.model = None
        self.grid = None
        # default CV if no other is given
        if not cv:
            self._cv = RepeatedStratifiedKFold()
        else:
            self._cv = cv
        self.trainset = {}
        self.scores_train = None
        self.scores_eval = None

    @staticmethod
    def to_numpy(arg):
        if isinstance(arg, np.ndarray):
            return arg
        elif isinstance(arg, pd.Series):
            return arg.to_numpy()
        else:
            return np.array(arg)

    def split(self, train_size: float = 0.9, n_splits=1, **params) -> None:
        """
        Split training dataset into train/test data.

        Args:
            train_size (float, optional): Proportion used for training. Defaults to 0.9.
            n_splits (int, optional): Number of iterations to split. Defaults to 1.
            **params (dict): Optional ShuffleSplit estimator params or custom labels/data.

        Returns:
            X_train (Iterable[Any]): Training features.
            x_test (Iterable[Any]): Testing features.
            Y_train (Iterable[Any]): Training labels.
            y_test (Iterable[Any]): Testing labels.
        """
        data = params.pop("data", self.traindata.data)
        target = params.pop("target", self.traindata.target)
        if self.cv is None:
            cv = StratifiedShuffleSplit(
                n_splits=n_splits, train_size=train_size, **params
            )
        else:
            cv = self.cv

        train_index, test_index = next(cv.split(data, target))
        X_train, x_test = data[train_index], data[test_index]
        Y_train, y_test = target[train_index], target[test_index]

        self.trainset["X_train"] = X_train
        self.trainset["x_test"] = x_test
        self.trainset["Y_train"] = Y_train
        self.trainset["y_test"] = y_test
        self.scale()

        return None

    def scale(self, **kwargs):
        """
        Scale to mean = 0 and st.dev = 1 a train/split dataset.

        Parameters
        ----------
        **kwargs : dict
            Custom train/test data. Must be supplied given names: 
                -> X_train, x_test, Y_train, y_test
                
        Returns
        -------
        None.
        """
        assert hasattr(self.trainset, "x_test")
        if not kwargs:
            X_train = self.trainset["X_train"]
            x_test = self.trainset["x_test"]
        else:
            X_train = kwargs["X_train"]
            x_test = kwargs["x_test"]
        # Get scaler for only training data, apply to training and test data
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        x_test_scaled = self.scaler.transform(x_test)
        self.trainset["X_train"] = self.to_numpy(X_train_scaled)
        self.trainset["x_test"] = self.to_numpy(x_test_scaled)
        return None

    def optimize_clf(
        self,
        X_train=None,
        Y_train=None,
        param_grid=None,
        verbose=True,
        refit=True,
        **svcparams,
    ):
        assert hasattr(self.trainset, "x_test")
        if param_grid:
            param_grid = param_grid
        else:
            param_grid = {
                "C": [0.1, 1, 10, 100, 150, 200, 500, 1000],
                "gamma": ["scale", "auto"],
                "kernel": ["linear", "rbf", "poly"],
            }
        if not X_train:
            X_train = self.trainset["X_train"]
            Y_train = self.trainset["Y_train"]
        else:
            X_train = X_train
            Y_train = Y_train

        svc = SVC(class_weight="balanced", **svcparams)
        self.grid = GridSearchCV(
            svc, param_grid=param_grid, cv=self.cv, refit=refit, verbose=verbose
        )
        self.grid.fit(X_train, Y_train.ravel())
        print("**", "-" * 20, "*", "-" * 20, "**")
        print(f"Best params: {self.grid.best_params_}")
        print("**", "-" * 5)
        print(f"Score: {self.grid.best_score_}")
        # Use optimized parameters for SVC model
        kernel_best = self.grid.best_params_["kernel"]
        c_best = self.grid.best_params_["C"]
        gamma_best = self.grid.best_params_["gamma"]
        self.model = SVC(C=c_best, kernel=kernel_best, gamma=gamma_best, verbose=False)
        print("Model optimized")
        return None

    def fit_clf(self, model=None) -> object:
        """
        Fit classifier to training data.

        Args:
            model (model): optional, input custom model.
        """
        assert hasattr(self.trainset, "x_test")
        X_train = self.trainset["X_train"]
        Y_train = self.trainset["Y_train"]
        if model:
            self.model = model  # If model proveed, override current model
            logging.info(f"Model provided has been fit: {model}")
        self.model.fit(X_train, np.ravel(Y_train))
        return None

    def predict_clf(self, x_test=None):
        assert hasattr(self.trainset, "x_test")
        if not x_test:
            x_test = self.trainset["x_test"]
        y_pred = self.model.predict(x_test)
        self.trainset["y_pred"] = y_pred
        y_true = self.trainset["y_test"]
        _validate._full_predict(y_pred, y_true)
        self.scores_train = Scoring(y_pred, y_true, desc="train", mat=True)
        return None

    def evaluate_clf(self):
        y_pred = self.model.predict(self.evaldata.data)
        y_true = self.evaldata.target
        self.trainset["eval_y_pred"] = y_pred
        self.trainset["eval_y_true"] = y_true
        _validate._full_predict(y_pred, y_true)
        self.scores_eval = Scoring(y_pred, y_true, desc="eval", mat=True)
        return None

    def get_learning_curves(self, estimator, title: str = "Learning Curve"):

        import matplotlib.pyplot as plt

        _, axes = plt.subplots(3, 2, figsize=(10, 15))

        X = self.trainset["X_test"]
        y = self.trainset["y_test"]

        # Cross validation with 50 iterations to get smoother mean test and train
        # score curves, each time with 20% data randomly selected as a validation set.
        title = title
        plot_learning_curve(
            estimator, title, X, y, axes=axes[:, 0], ylim=(0, 1.01), cv=self.cv
        )
        kern = self.grid.best_params_["kernel"]
        gam = self.grid.best_params_["gamma"]
        C = self.grid.best_params_["C"]
        title = f"SVC - kernel = {kern}, gamma = {gam}, C = {C}"
        plot_learning_curve(
            estimator, title, X, y, axes=axes[:, 1], ylim=(0, 1.01), cv=self.cv
        )
        plt.show()


if __name__ == "__main__":
    pass
