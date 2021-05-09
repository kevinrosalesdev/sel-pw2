import numpy as np
import pandas as pd

from collections import Counter
from forests.CART import CART


def vote(predictions):
    votes = [Counter(np.array(predictions)[:, i]) for i in range(len(predictions[0]))]
    res = [None]*len(votes)
    for idx, vote in enumerate(votes):
        for prediction, counter in vote.most_common(2):
            if prediction is not None:
                res[idx] = prediction
                break

    return res


class RandomForest:

    def __init__(self, F: int, NT: int, threshold_ratio=0.05):
        self.F = F
        self.carts = []
        self.feat_freq = {}
        for i in range(NT):
            self.carts.append(CART(threshold_ratio=threshold_ratio))

    def fit(self, training_set: pd.DataFrame):
        for cart in self.carts:
            cart.fit(training_set=training_set.sample(frac=1, replace=True), f=self.F)

        self.get_feature_freq(training_set)

    def predict(self, test_set: pd.DataFrame):
        predictions = [cart.predict(test_set) for cart in self.carts]
        return vote(predictions)

    def get_feature_freq(self, training_set):
        # Absolute Frequency
        for column in training_set.columns[:-1]:
            self.feat_freq[column] = np.sum([cart.feat_freq[column] for cart in self.carts])

        counter = np.sum(list(self.feat_freq.values()))

        # Relative Frequency
        for column in training_set.columns[:-1]:
            self.feat_freq[column] = round(self.feat_freq[column]/counter, 3)
