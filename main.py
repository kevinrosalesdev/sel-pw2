import numpy as np
import pandas as pd
import random
import warnings

from datapreprocessor import preprocessor
from forests.RandomForest import RandomForest
from forests.DecisionForest import DecisionForest
from utils import metrics
from datetime import datetime

warnings.filterwarnings("ignore")


def generate_tables(dataset, dataset_name, train_test_ratio, threshold_ratio=0.05):
    train_set = dataset.iloc[:int(dataset.shape[0] * train_test_ratio)].reset_index(drop=True)
    test_set = dataset.iloc[int(dataset.shape[0] * train_test_ratio):].reset_index(drop=True)
    M = train_set.shape[1] - 1

    pd_rf = pd.DataFrame(columns=['F', 'NT', 'Feat_Freq', 'Accuracy', 'Train_Time', 'Classif_Time'])
    for NT in [1, 10, 25, 50, 75, 100]:
        for F in [1, 3, int(np.log2(M) + 1), int(np.sqrt(M))]:
            print(f"[Testing RF with NT={NT} and F={F}]")
            rf = RandomForest(F=F, NT=NT, threshold_ratio=threshold_ratio)
            start_time = datetime.now()
            rf.fit(train_set)
            train_time = datetime.now() - start_time
            start_time = datetime.now()
            prediction = rf.predict(test_set)
            classif_time = datetime.now() - start_time
            ground_truth = list(test_set.iloc[:, -1])
            accuracy = round(metrics.get_global_accuracy(ground_truth, prediction) * 100, 3)
            pd_rf = pd_rf.append({'F': F,
                                  'NT': NT,
                                  'Feat_Freq': rf.feat_freq,
                                  'Accuracy': accuracy,
                                  'Train_Time': str(train_time),
                                  'Classif_Time': str(classif_time)},
                                 ignore_index=True)

    pd_rf.to_csv(f'Out/RF-{dataset_name}.csv', index=False)

    pd_df = pd.DataFrame(columns=['F', 'NT', 'Feat_Freq', 'Accuracy', 'Train_Time', 'Classif_Time'])
    for NT in [1, 10, 25, 50, 75, 100]:
        for F in [int(M/4), int(M/2), int(3*M/4), None]:
            if F == 0:
                F = 1

            print(f"[Testing DF with NT={NT} and F={F}]")
            df = DecisionForest(F=F, NT=NT, threshold_ratio=threshold_ratio)
            start_time = datetime.now()
            df.fit(train_set)
            train_time = datetime.now() - start_time
            start_time = datetime.now()
            prediction = df.predict(test_set)
            classif_time = datetime.now() - start_time
            ground_truth = list(test_set.iloc[:, -1])
            accuracy = round(metrics.get_global_accuracy(ground_truth, prediction) * 100, 3)
            pd_df = pd_df.append({'F': F,
                                  'NT': NT,
                                  'Feat_Freq': df.feat_freq,
                                  'Accuracy': accuracy,
                                  'Train_Time': str(train_time),
                                  'Classif_Time': str(classif_time)},
                                 ignore_index=True)

    print(pd_df)
    pd_df.to_csv(f'Out/DF-{dataset_name}.csv', index=False)


def evaluate(dataset, train_test_ratio=0.8, F=None, NT=10, threshold_ratio=0.05):
    train_set = dataset.iloc[:int(dataset.shape[0] * train_test_ratio)].reset_index(drop=True)
    test_set = dataset.iloc[int(dataset.shape[0] * train_test_ratio):].reset_index(drop=True)
    start_time = datetime.now()

    df = DecisionForest(F=F, NT=NT, threshold_ratio=threshold_ratio)
    df.fit(train_set)
    print(f"[DF] Relative Frequency of Features: {df.feat_freq}")
    end_time = datetime.now()
    print(f'[DF] Training Time: {end_time - start_time}')
    start_time = datetime.now()
    prediction = df.predict(test_set)
    end_time = datetime.now()
    print(f'[DF] Classification Time: {end_time - start_time}')
    print("________________________________________________")
    ground_truth = list(test_set.iloc[:, -1])
    print(f"[DF] Predicted:\n{prediction}")
    print(f"[DF] GT:\n{ground_truth}\n")
    print(f"[DF] Accuracy                  : {round(metrics.get_global_accuracy(ground_truth, prediction) * 100, 3)}%")
    print(f"[DF] Classification Report [Global]:\n{metrics.classification_report(ground_truth, prediction)}")
    print(f"[DF] Confusion Matrix [Global]:\n{metrics.get_confusion_matrix(ground_truth, prediction)}")

    if F is None:
        F = int(np.sqrt(train_set.shape[1] - 1))

    rf = RandomForest(F=F, NT=NT, threshold_ratio=threshold_ratio)
    rf.fit(train_set)
    print(f"[RF] Relative Frequency of Features: {rf.feat_freq}")
    end_time = datetime.now()
    print(f'[RF] Training Time: {end_time - start_time}')
    start_time = datetime.now()
    prediction = rf.predict(test_set)
    end_time = datetime.now()
    print(f'[RF] Classification Time: {end_time - start_time}')
    print("________________________________________________")
    ground_truth = list(test_set.iloc[:, -1])
    print(f"[RF] Predicted:\n{prediction}")
    print(f"[RF] GT:\n{ground_truth}\n")
    print(f"[RF] Accuracy                  : {round(metrics.get_global_accuracy(ground_truth, prediction) * 100, 3)}%")
    print(f"[RF] Classification Report [Global]:\n{metrics.classification_report(ground_truth, prediction)}")
    print(f"[RF] Confusion Matrix [Global]:\n{metrics.get_confusion_matrix(ground_truth, prediction)}")


if __name__ == '__main__':
    random.seed(0)

    # print("===============================================")
    # print("Test Dataset Evaluation [Slides Dataset]")
    # print("===============================================")
    # slides = preprocessor.load_test_dataset().sample(frac=1, random_state=0)
    # generate_tables(dataset=slides, dataset_name='slides', train_test_ratio=0.8, threshold_ratio=0.05)
    # evaluate(slides, train_test_ratio=0.8)

    # print("===============================================")
    # print("Lenses Dataset Evaluation [Very Small Dataset]")
    # print("===============================================")
    # lenses = preprocessor.load_lenses_dataset().sample(frac=1, random_state=0)
    # generate_tables(dataset=lenses, dataset_name='lenses', train_test_ratio=0.8, threshold_ratio=0.05)
    # evaluate(lenses, train_test_ratio=0.8)

    print("\n===============================================")
    print("Ecoli Dataset Evaluation [Small Dataset]")
    print("===============================================")
    ecoli = preprocessor.load_ecoli_dataset().sample(frac=1, random_state=0)
    generate_tables(dataset=ecoli, dataset_name='ecoli', train_test_ratio=0.8, threshold_ratio=0.05)
    # evaluate(ecoli, train_test_ratio=0.8)

    # print("\n===============================================")
    # print("Car Dataset Evaluation [Medium Dataset]")
    # print("===============================================")
    # car = preprocessor.load_car_dataset().sample(frac=1, random_state=0)
    # generate_tables(dataset=car, dataset_name='car', train_test_ratio=0.8, threshold_ratio=0.05)
    # evaluate(car, train_test_ratio=0.8)

    # print("\n===============================================")
    # print("Rice Dataset Evaluation [Large Dataset]")
    # print("===============================================")
    # rice = preprocessor.load_rice_dataset().sample(frac=1, random_state=0)
    # generate_tables(dataset=rice, dataset_name='rice', train_test_ratio=0.8, threshold_ratio=0.05)
    # evaluate(rice, train_test_ratio=0.8)

    # print("\n===============================================")
    # print("Stroke Dataset Evaluation [Large Dataset]")
    # print("===============================================")
    # stroke = preprocessor.load_stroke_dataset().sample(frac=1, random_state=0)
    # generate_tables(dataset=stroke, dataset_name='stroke', train_test_ratio=0.8, threshold_ratio=0.05)
    # evaluate(stroke, train_test_ratio=0.8)
