import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

def compute_metrics(y_true, y_prob, thresh=0.5):
    y_pred = (np.array(y_prob) >= thresh).astype(int)
    y_true = np.array(y_true)
    res = {}
    try:
        res['auc'] = roc_auc_score(y_true, y_prob)
    except:
        res['auc'] = float('nan')
    res['acc'] = float(accuracy_score(y_true, y_pred))
    res['prec'] = float(precision_score(y_true, y_pred, zero_division=0))
    res['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
    res['f1'] = float(f1_score(y_true, y_pred, zero_division=0))
    return res
