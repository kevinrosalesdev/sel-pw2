from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


def get_global_accuracy(ground_truth, prediction):
    return sum([1 if prediction[i] == ground_truth[i] else 0 for i in range(len(prediction))])/len(prediction)


def get_classified_accuracy(ground_truth, prediction):
    gt = np.array(ground_truth)
    pr = np.array(prediction)
    non_classified_instances_idx = np.where(pr == 'False')
    gt = np.delete(gt, non_classified_instances_idx)
    pr = np.delete(pr, non_classified_instances_idx)
    return sum([1 if pr[i] == gt[i] else 0 for i in range(len(pr))])/len(pr)


def get_classification_report(ground_truth, prediction):
    return classification_report(ground_truth, prediction, zero_division=1)


def get_confusion_matrix(ground_truth, prediction):
    return confusion_matrix(ground_truth, prediction)
