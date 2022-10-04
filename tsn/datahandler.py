#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# datahandler.py

Module (nn_utils): Class to handle data for import into SVM classifier.
"""
import numpy as np


class DataHandler:
    def __init__(self, data, target):
        """
        Check and index specific data to feed into SVM. Accepted as input to sklearn.GridSearchCV().
        Features are the data used for regression and margin vectorizations.
        Labels (or targets, synonymous) are what the classifier is being trained on.

        Parameters
        ----------
        data : pd.DataFrame | np.ndarray
            Features. 
        target : pd.Series | np.ndarray
            Labels/targets.
        """
        assert data.shape[0] == target.shape[0]
        self.data = np.array(data)
        self.target = np.array(target)

    def __getitem__(self, idx: int):
        """
        Indexer. Returns both feature and target values.

        Parameters
        ----------
        idx : int | Iterable[Any]
            Indexors of any type matching the index of the given dataset.

        Returns
        -------
        slice
            Indexed features.
        slice
            Indexed targets.
        """
        return self.data[idx], self.target[idx]
