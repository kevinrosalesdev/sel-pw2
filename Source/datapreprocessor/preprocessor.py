import pandas as pd


def load_test_dataset() -> pd.DataFrame:
    return pd.read_csv('Data/example.csv')


def load_lenses_dataset() -> pd.DataFrame:
    return pd.read_csv('Data/lenses.csv')


def load_ecoli_dataset() -> pd.DataFrame:
    ecoli = pd.read_csv('Data/ecoli.csv')
    ecoli.drop(['Sequence Name', 'lip', 'chg'], axis=1, inplace=True)
    return ecoli


def load_car_dataset() -> pd.DataFrame:
    return pd.read_csv('Data/car.csv')


def load_rice_dataset() -> pd.DataFrame:
    return pd.read_csv('Data/rice.csv')


def load_stroke_dataset() -> pd.DataFrame:
    stroke = pd.read_csv('Data/stroke.csv')
    stroke.drop(['id'], axis=1, inplace=True)
    stroke['bmi'].fillna(stroke['bmi'].median(), inplace=True)
    return stroke

