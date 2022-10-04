#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# scores.py

Module (neuralnetwork): Structures to score and keep scores for evaluated data.

"""

from __future__ import division

import logging
import pickle
from dataclasses import dataclass
from typing import Optional, Iterable, Any

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from graphs.plot import Plot

logger = logging.getLogger(__name__)


def save(save_file_path, team):
    with open(save_file_path, "wb") as f:
        pickle.dump(team, f)


def load(save_file_path):
    with open(save_file_path, "rb") as f:
        return pickle.load(f)


@dataclass
class Scoring(object):
    def __init__(
        self,
        pred: np.ndarray,
        true: np.ndarray,
        desc: Optional[str] = "",
        mat: bool = False,
    ) -> None:
        """
        Class to manage scoring variables from fitted classifiers. 

        Parameters
        ----------
        pred : ndarray
            Fitted model's "predicted" output.
        true : ndarray
            Descriptors of each predictable value.
        mat : bool, optional
            Whether to output a Confusion Matrix. The default is False.

        Returns
        -------
        None.
        """
        # Input variables
        self.predicted: Iterable[Any] = pred
        self.true: Iterable[Any] = true
        self.classes: list = list(np.unique(self.predicted))
        self.descriptor: str = desc
        self.report: pd.DataFrame = self.get_report()

        if mat:
            self.mat: Any = self.get_confusion_matrix()
        if desc is None:
            logging.info("No descriptor")
            pass

    def get_report(self) -> pd.DataFrame:
        """ Get classification report"""
        if self.descriptor:
            assert self.descriptor in [
                "train",
                "training",
                "test",
                "testing",
                "eval",
                "val",
                "evaluate",
            ]

        self.report = classification_report(
            self.true,
            self.predicted,
            target_names=self.classes,
            labels=self.classes,
            output_dict=True,
        )
        report_df = pd.DataFrame(data=self.report).transpose()
        return report_df

    def get_confusion_matrix(self, caption: Optional[str] = "") -> object:
        """ Get confusion matrix"""
        mat = Plot.confusion_matrix(
            y_true=self.true,
            y_pred=self.predicted,
            labels=self.classes,
            caption=caption,
        )

        return mat
