

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
import random
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MetricsTop:
    def __init__(self, mode):
        """Initialize with available metric evaluation functions."""

        # Dictionary mapping dataset names to their respective evaluation functions
        self.metrics_dict = {
            'Empathy': self.__eval_regression
        }

    def __eval_regression(self, y_pred, y_true):
        """Compute Acc2, F1, MAE, Corr. Return dict of rounded metrics."""
        # Handle both numpy arrays and PyTorch tensors
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()

        # Flatten predictions and true values for correlation calculation
        y_pred = np.array(y_pred).flatten()
        y_true = np.array(y_true).flatten()
        corr = np.corrcoef(y_pred, y_true)[0][1]

        # Mean Absolute Error (MAE)
        mae = np.mean(np.absolute(y_pred - y_true)).astype(np.float64)

        # Remove the value of 0 (i.e., midpoint), and calculate Acc2 and F1 score
        non_zeros = np.array([i for i, e in enumerate(y_true) if e != 0])
        #For binary evaluation, scores above 0 were coded as positive (high empathy),
        # whereas scores below 0 were coded as negative (low empathy)
        binary_truth = (y_true[non_zeros] > 0)
        binary_preds = (y_pred[non_zeros] > 0)

        # Calculate Acc2 and F1 score
        acc2 = accuracy_score(binary_preds, binary_truth)
        f1_value = f1_score(binary_truth, binary_preds, average='weighted')
        # Compile results into a dictionary with rounded values
        eval_results_reg = {
            "Acc_2": round(acc2, 4),
            "F1_score": round(f1_value, 4),
            "MAE": round(mae, 4),
            "Corr": round(corr, 4),
        }
        return eval_results_reg

    def getMetrics(self, datasetName):
        """Return the evaluation function for the given dataset."""
        return self.metrics_dict[datasetName]
