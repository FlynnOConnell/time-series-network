#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# _validate.py

Module (nn_utils): 
    Validation functions for SVM. Determine which steps have occured, 
    Prevent steps from occuring before previous attributes have been set
"""

import numpy as np


class _validate:
    @staticmethod
    def _full_predict(pred, true):
        assert (np.unique(pred)) == (np.unique(true))

    @staticmethod
    def _validate_shape(x, y):
        assert x.shape[0] == y.shape[0]

    @staticmethod
    def _check_fitted(model):
        if not hasattr(model, "fit"):
            raise TypeError(f"{model} is not an estimator instance.")

    @staticmethod
    def _num_classes(classes):
        if len(np.unique(classes)) == 2:
            print("Estimator class type: Multivariate")
            return "binary"
        elif len(np.unique(classes)) > 2:
            print("Estimator class type: Multivariate")
            return "multivariate"
        else:
            raise AttributeError(
                "Number of classes do not match binary or multivariate"
            )
