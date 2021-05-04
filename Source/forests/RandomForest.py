import pandas as pd

from scipy.stats import mode
from forests.CART import CART


class RandomForest:

    def __init__(self, F: int, NT: int):
        self.F = F
        self.carts = [CART()]*NT

    def fit(self, training_set: pd.DataFrame):
        for idx, cart in enumerate(self.carts, 1):
            print(f"[Training tree {idx}/10]")
            cart.fit(training_set=training_set.sample(frac=1, replace=True), f=self.F)

    def predict(self, test_set: pd.DataFrame):
        predictions = [cart.predict(test_set) for cart in self.carts]
        return list(mode(predictions)[0])[0]
