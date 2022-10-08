from picograd.engine import Var

from typing import List


def mean_squared_error(y_true: List[Var], y_pred: List[Var]) -> float:
    """MSE loss"""
    assert len(y_true) == len(y_pred)
    total_squared_error = sum([(y_true_i - y_pred_i) ** 2 for y_true_i, y_pred_i in zip(y_true, y_pred)], 0.0).data
    n_total = max(len(y_true), 1)
    return total_squared_error / n_total


def binary_accuracy(y_true: List[Var], y_pred: List[Var]) -> float:
    """Binary accuracy"""
    assert len(y_true) == len(y_pred)
    n_exact = sum([y_true_i.data == round(y_pred_i.data) for y_true_i, y_pred_i in zip(y_true, y_pred)], 0)
    n_total = max(len(y_true), 1)
    return n_exact / n_total
