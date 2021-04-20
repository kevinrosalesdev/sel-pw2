import pandas as pd


class DecisionForest:

    def __init__(self, F: int, NT: int):
        self.F = F
        self.NT = NT

    def fit(self, training_set: pd.DataFrame) -> list():
        pass

    def predict(self, test_set: pd.DataFrame) -> list():
        pass
