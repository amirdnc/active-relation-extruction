from collections import Counter
from typing import Optional
import numpy as np
from overrides import overrides
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric

from my_lib import scorer


NO_RELATION = 0
@Metric.register("special_loss_term")
class SpecialLoss(Metric):
    def __init__(self, no_relation_label : int) -> None:
        self.no_relation_label = no_relation_label
        self.correct_by_relation = Counter()
        self.guessed_by_relation = Counter()
        self.gold_by_relation = Counter()
        self.count = 0

    def __call__(self, pred, labels):
        preds = pred.topk(1)[1]
        for row in range(len(labels)):
            gold = labels[row]
            guess = preds[row]

            if gold == NO_RELATION and guess == NO_RELATION:
                pass
            elif gold == NO_RELATION and guess != NO_RELATION:
                self.guessed_by_relation[guess] += 1
            elif gold != NO_RELATION and guess == NO_RELATION:
                self.gold_by_relation[gold] += 1
            elif gold != NO_RELATION and guess != NO_RELATION:
                self.guessed_by_relation[guess] += 1
                self.gold_by_relation[gold] += 1
                if gold == guess:
                    self.correct_by_relation[guess] += 1
        self.count += 1

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated accuracy.
        """

        prec_micro = 1.0
        if sum(self.guessed_by_relation.values()) > 0:
            prec_micro = float(sum(self.correct_by_relation.values())) / float(sum(self.guessed_by_relation.values()))
        recall_micro = 0.0
        if sum(self.gold_by_relation.values()) > 0:
            recall_micro = float(sum(self.correct_by_relation.values())) / float(sum(self.gold_by_relation.values()))
        f1_micro = 0.0
        if prec_micro + recall_micro > 0.0:
            f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)

        if reset:
            self.reset()
        return prec_micro, recall_micro, f1_micro

    @overrides
    def reset(self):
        self.correct_by_relation = Counter()
        self.guessed_by_relation = Counter()
        self.gold_by_relation = Counter()
        self.count = 0