import pandas as pd


def load_test_dataset() -> pd.DataFrame:
    return pd.read_csv('Data/example.csv')


def load_ecoli_dataset() -> pd.DataFrame:
    ecoli = pd.read_csv('Data/ecoli.csv')
    ecoli.drop(['Sequence Name', 'lip', 'chg'], axis=1, inplace=True)
    for idx in range(ecoli.shape[1]-1):
        ecoli.iloc[:, idx] = pd.qcut(ecoli.iloc[:, idx], 4)
    return ecoli


def load_car_dataset() -> pd.DataFrame:
    return pd.read_csv('Data/car.csv')


def load_rice_dataset() -> pd.DataFrame:
    rice = pd.read_csv('Data/rice.csv')
    for idx in range(rice.shape[1]-1):
        rice.iloc[:, idx] = pd.qcut(rice.iloc[:, idx], 4)
    return rice


def load_stroke_dataset() -> pd.DataFrame:
    stroke = pd.read_csv('Data/stroke.csv')
    stroke.drop(['id'], axis=1, inplace=True)
    stroke['bmi'].fillna(stroke['bmi'].median(), inplace=True)
    for column in ['age', 'avg_glucose_level', 'bmi']:
        stroke[column] = pd.qcut(stroke[column], 4)
    return stroke

