"""
Description: Code adapted from aicc-ognet-global/eval/metrics.py 
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score
)

"""  
TODO:
- Try weighted average 
- Report micro 
- Oversample weighted 
- sklearn.metrics.precision_recall_fscore_support
- Look into implementing oversampling 
"""
def get_metrics_multiclass(labels, probs, metrics_strategy="weighted", threshold=None, num_classes=3):
    # Convert probabilities to binary predictions based on the threshold
    auprc_valid = torch.unique(labels).shape[0] == num_classes
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()

    if threshold is not None:
        predictions = (probs >= threshold).astype(int)
    else:
        predictions = np.argmax(probs, axis=1)

    # Calculate metrics for each class
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average=metrics_strategy)
    recall = recall_score(labels, predictions, average=metrics_strategy)
    f1 = f1_score(labels, predictions, average=metrics_strategy)

    # Calculate AUROC and AUPRC
    if auprc_valid:
        auroc = roc_auc_score(labels, probs, average=metrics_strategy, multi_class='ovr')
        auprc = average_precision_score(labels, probs, average=metrics_strategy)

        # Calculate AUROC and AUPRC for each class
        auroc_per_class = roc_auc_score(labels, probs, average=None, multi_class='ovr')
        auprc_per_class = average_precision_score(labels, probs, average=None)
    else:
        auroc = 0
        auprc = 1 

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auroc': auroc,
        'auprc': auprc
    }

def get_optimal_f1(groundtruth, probabilities,
                   return_threshold=False):
    """Get threshold maximizing f1 score."""
    prec, rec, threshold = precision_recall_curve(
        groundtruth, probabilities
    )

    f1_values = 2 * (prec * rec) / (prec + rec)

    argmax_f1 = np.nanargmax(f1_values)
    max_f1 = np.nanmax(f1_values)

    if return_threshold:
        threshold_result = threshold[argmax_f1] if not (argmax_f1 + 1 > len(threshold)) else threshold[0]
        return max_f1, threshold_result
    else:
        return max_f1
    

def get_max_precision_above_recall(groundtruth, probabilities, value,
                                   return_threshold=False):
    """Get maximum precision such that recall >= value."""
    if value > 1:
        raise ValueError(f"Cannot attain a recall of {value}")
    prec, rec, threshold = precision_recall_curve(
        groundtruth, probabilities
    )
    
    max_prec_above_rec = max(p for p, r in zip(prec, rec) if r >= value)

    if return_threshold:
        index = list(prec).index(max_prec_above_rec)
        return max_prec_above_rec, threshold[index - 1]
    else:
        return max_prec_above_rec


def get_metrics(labels, probs, threshold=None):
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()[:, 1]

    if threshold is None:
        _, threshold = get_optimal_f1(labels, probs, return_threshold=True)
    prec_95_recall, threshold_95 = get_max_precision_above_recall(labels, probs, 0.95, return_threshold=True)
    prec_90_recall, threshold_90 = get_max_precision_above_recall(labels, probs, 0.90, return_threshold=True)
    preds = (probs > threshold).astype(int)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    try:
        auroc = roc_auc_score(labels, probs)
    except ValueError:
        # Catch when labels are all one class
        auroc = 0
    auprc = average_precision_score(labels, probs)
    prevalence = np.mean(labels)

    return {
        'threshold': threshold,
        'prevalence': prevalence,
        'f1': f1,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'auroc': auroc,
        'auprc': auprc,
        'prec_at_0.95_rec': prec_95_recall, 
        'prec_at_0.9_rec' : prec_90_recall,
        'threshold_0.95' : threshold_95,
        'threshold_0.9' : threshold_90,
    }

