from dataclasses import dataclass, astuple
from itertools import combinations
from typing import List

import numpy as np

@dataclass
class Metrics:
    true_positive: int
    false_positive: int
    false_negative: int

    def prec(self) -> float:
        n_predicted = self.true_positive + self.false_positive
        if not n_predicted:
            return 1.0
        return float(self.true_positive) / n_predicted

    def recall(self) -> float:
        n_true = self.true_positive + self.false_negative
        if not n_true:
            return 1.0
        return float(self.true_positive) / n_true

    def f1(self) -> float:
        p, r = self.prec(), self.recall()
        if p + r == 0:
            return 0
        return 2 * p * r / (p + r)

    def f_beta(self, beta: float):
        p, r = self.prec(), self.recall()
        if p + r == 0:
            return 0
        b2 = beta*beta
        return (1 + b2) * (p * r) / ((b2 * p) + r)

    def positives(self) -> int:
        return self.true_positive + self.false_positive

    def errors(self) -> int:
        return self.false_negative + self.false_positive

    def accuracy(self, n_decisions: int) -> float:
        return 1 - (self.errors()/float(n_decisions))

    def as_np(self):
        """ Return the tuple representation of this Metrics as numpy array. """
        return np.array(astuple(self))

    """ Operators """
    def __add__(self, other):
        if other == 0:
            return self
        return Metrics( *(self.as_np() + np.array(astuple(other))))

    def __radd__(self, other):
        if other == 0:
            return self
        return Metrics( *(self.as_np() + np.array(astuple(other))))

    def __str__(self):
        return f"P: {self.prec():.3f}   R: {self.recall():.3f}   F1: {self.f1():.3f}"

    def __repr__(self):
        return str(self)

    @classmethod
    def empty(cls):
        return Metrics(0,0,0)


@dataclass
class ManyToOneMetrics():
    predicted_true_positive: int
    reference_true_positive: int
    false_positive: int
    false_negative: int    

    def prec(self) -> float:
        n_predicted = self.predicted_true_positive + self.false_positive
        if not n_predicted:
            return 1.0
        return float(self.predicted_true_positive) / n_predicted

    def recall(self) -> float:
        n_true = self.reference_true_positive + self.false_negative
        if not n_true:
            return 1.0
        return float(self.reference_true_positive) / n_true

    def f1(self) -> float:
        p, r = self.prec(), self.recall()
        if p + r == 0:
            return 0
        return 2 * p * r / (p + r)

    def f_beta(self, beta: float):
        p, r = self.prec(), self.recall()
        if p + r == 0:
            return 0
        b2 = beta*beta
        return (1 + b2) * (p * r) / ((b2 * p) + r)

    def positives(self) -> int:
        return self.predicted_true_positive + self.false_positive

    def errors(self) -> int:
        return self.false_negative + self.false_positive

    def accuracy(self, n_decisions: int) -> float:
        return 1 - (self.errors()/float(n_decisions))

    def as_np(self):
        """ Return the tuple representation of this Metrics as numpy array. """
        return np.array(astuple(self))

    """ Operators """
    def __add__(self, other):
        if other == 0:
            return self
        return ManyToOneMetrics( *(self.as_np() + np.array(astuple(other))))

    def __radd__(self, other):
        if other == 0:
            return self
        return ManyToOneMetrics( *(self.as_np() + np.array(astuple(other))))

    def __str__(self):
        return f"P: {self.prec():.3f}   R: {self.recall():.3f}   F1: {self.f1():.3f}"

    def __repr__(self):
        return str(self)

    @classmethod
    def empty(cls):
        return ManyToOneMetrics(0,0,0,0)    


@dataclass
class BinaryClassificationMetrics:
    """ Helper dataclass to compute accuracy on a binary classification task. """
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int

    def instances(self) -> int:
        """ Number of decisions counted in this Metric """
        return sum([self.true_positive, self.true_negative, self.false_positive, self.false_negative])

    def positives(self) -> int:
        return self.true_positive + self.false_positive

    def errors(self) -> int:
        return self.false_negative + self.false_positive

    def correct_predictions(self) -> int:
        return self.true_positive + self.true_negative

    def accuracy(self) -> float:
        if not self.instances():
            return 1.0
        return self.correct_predictions() / float(self.instances())

    """ When desired, we can also treat it as a retrieval problem with p, r & f1. """
    def prec(self):
        n_predicted = self.true_positive + self.false_positive
        if not n_predicted:
            return 1.0
        return float(self.true_positive) / n_predicted

    def recall(self):
        n_true = self.true_positive + self.false_negative
        if not n_true:
            return 1.0
        return float(self.true_positive) / n_true

    def f1(self):
        p, r = self.prec(), self.recall()
        if p+r == 0:
            return 0
        return 2 * p * r / (p + r)

    def as_np(self):
        """ Return the tuple representation of this Metrics as numpy array. """
        return np.array(astuple(self))

    """ Operators """
    def __add__(self, other):
        if other == 0:
            return self
        return BinaryClassificationMetrics( *(self.as_np() + np.array(astuple(other))))

    def __radd__(self, other):
        if other == 0:
            return self
        return BinaryClassificationMetrics(*(self.as_np() + np.array(astuple(other))))

    def __mul__(self, other: int):
        return BinaryClassificationMetrics( *(self.as_np()*other) )

    def __sub__(self, other):
        return BinaryClassificationMetrics( *(self.as_np() - np.array(astuple(other))))

    @classmethod
    def empty(cls):
        return BinaryClassificationMetrics(0,0,0,0)

    @classmethod
    def simple_boolean_decision(cls, sys_decision: bool, grt_decision: bool):
        tp = int(sys_decision and grt_decision)
        tn = int(not sys_decision and not grt_decision)
        fp = int(sys_decision and not grt_decision)
        fn = int(not sys_decision and grt_decision)
        return BinaryClassificationMetrics(tp,tn,fp,fn)


