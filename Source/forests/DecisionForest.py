import numpy as np
import pandas as pd

from forests.CART import CART
from forests.RandomForest import vote


class DecisionForest:

    def __init__(self, F: int, NT: int, threshold_ratio=0.05):
        self.F = F
        self.carts = []
        self.feat_freq = {}
        for i in range(NT):
            self.carts.append(CART(threshold_ratio=threshold_ratio))

    def fit(self, training_set: pd.DataFrame):
        run_if = self.F is None
        for cart in self.carts:
            if run_if:
                self.F = np.random.random_integers(low=1, high=training_set.shape[1]-1)

            random_features = []
            while len(random_features) < self.F:
                random_feature = np.random.random_integers(low=0, high=training_set.shape[1]-2)
                if random_feature not in random_features:
                    random_features.append(random_feature)
            cart.fit(training_set=training_set, f=random_features)

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

