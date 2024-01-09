from sklearn import metrics as skmetrics
from matplotlib import pyplot as plt

def get_roc_figure(roc_curve, metric_title):
    fpr, tpr, thresholds = roc_curve
    fig, axis = plt.subplots()
    axis.plot(fpr, tpr)
    axis.scatter(fpr, tpr)
    axis.set_title(metric_title)
    axis.set_xlabel('False Positive Rate')
    axis.set_ylabel('True Positive Rate')
    return fig

def get_pr_figure(pr_curve, metric_title):
    precision, recall, thresholds = pr_curve
    fig, axis = plt.subplots()
    axis.plot(recall, precision)
    axis.scatter(recall, precision)
    axis.set_title(metric_title)
    axis.set_xlabel('Recall')
    axis.set_ylabel('Precision')
    return fig

# Input data X_test (NxD) and labels y_test (N). Output metrics.
# 1 is positive (fine transaction). 0 is negative (bad transaction).
def evaluate(predicted, labels, plot=False):
    metrics = {}

    metrics['auc_roc'] = skmetrics.roc_auc_score(labels, predicted)
    metrics['roc_curve'] = skmetrics.roc_curve(labels, predicted)
    if plot:
        metrics['roc_figure'] = get_roc_figure(metrics['roc_curve'], 'ROC Curve Observer')

    metrics['ap'] = skmetrics.average_precision_score(1 - labels, 1 - predicted)
    metrics['pr_curve'] = skmetrics.precision_recall_curve(1 - labels, 1 - predicted)
    if plot:
        metrics['pr_figure'] = get_pr_figure(metrics['pr_curve'], 'PR Curve Observer (on detecting Negative / Bad / Bugged transaction)')

    return metrics