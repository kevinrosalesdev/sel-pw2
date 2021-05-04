import numpy as np
import pandas as pd

from scipy.stats import mode
from forests.CART import CART


class DecisionForest:

    def __init__(self, F: int, NT: int):
        self.F = F
        self.carts = [CART()]*NT

    def fit(self, training_set: pd.DataFrame):
        for idx, cart in enumerate(self.carts, 1):
            print(f"[Training tree {idx}/10]")
            random_features = []
            while len(random_features) < self.F:
                random_feature = np.random.random_integers(low=0, high=training_set.shape[1]-2)
                if random_feature not in random_features:
                    random_features.append(random_feature)
            cart.fit(training_set=training_set, f=random_features)

    def predict(self, test_set: pd.DataFrame):
        predictions = [cart.predict(test_set) for cart in self.carts]
        return list(mode(predictions)[0])[0]
