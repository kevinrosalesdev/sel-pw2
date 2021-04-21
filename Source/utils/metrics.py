from sklearn.metrics import classification_report, confusion_matrix


def get_global_accuracy(ground_truth, prediction):
    return sum([1 if prediction[i] == ground_truth[i] else 0 for i in range(len(prediction))])/len(prediction)


def get_classification_report(ground_truth, prediction):
    return classification_report(ground_truth, prediction, zero_division=1)


def get_confusion_matrix(ground_truth, prediction):
    return confusion_matrix(ground_truth, prediction)
