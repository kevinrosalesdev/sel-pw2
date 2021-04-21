from datapreprocessor import preprocessor
from forests.CART import CART
from utils import metrics
from datetime import datetime
import random


def evaluate(dataset, train_test_ratio=0.8):
    train_set = dataset.iloc[:int(dataset.shape[0]*train_test_ratio)].reset_index(drop=True)
    test_set = dataset.iloc[int(dataset.shape[0]*train_test_ratio):].reset_index(drop=True)
    start_time = datetime.now()
    cart = CART()
    cart.fit(train_set)
    end_time = datetime.now()
    print(f'Tree Generation Time: {end_time - start_time}')
    start_time = datetime.now()
    prediction = cart.predict(test_set)
    end_time = datetime.now()
    print(f'Tree Classification Time: {end_time - start_time}')
    print("________________________________________________")
    ground_truth = list(test_set.iloc[:, -1])
    print(f"Predicted:\n{prediction}")
    print(f"GT:\n{ground_truth}\n")
    print(f"Accuracy                  : {round(metrics.get_global_accuracy(ground_truth, prediction)*100, 3)}%")
    print(f"Classification Report [Global]:\n{metrics.classification_report(ground_truth, prediction)}")
    print(f"Confusion Matrix [Global]:\n{metrics.get_confusion_matrix(ground_truth, prediction)}")


if __name__ == '__main__':
    random.seed(0)

    print("===============================================")
    print("Test Dataset Evaluation [Slides Dataset]")
    print("===============================================")
    slides = preprocessor.load_test_dataset().sample(frac=1, random_state=0)
    evaluate(slides, train_test_ratio=0.8)
    #
    print("===============================================")
    print("Lenses Dataset Evaluation [Very Small Dataset]")
    print("===============================================")
    lenses = preprocessor.load_lenses_dataset().sample(frac=1, random_state=0)
    evaluate(lenses, train_test_ratio=0.8)

    print("\n===============================================")
    print("Ecoli Dataset Evaluation [Small Dataset]")
    print("===============================================")
    ecoli = preprocessor.load_ecoli_dataset().sample(frac=1, random_state=0)
    evaluate(ecoli, train_test_ratio=0.8)

    print("\n===============================================")
    print("Car Dataset Evaluation [Medium Dataset]")
    print("===============================================")
    car = preprocessor.load_car_dataset().sample(frac=1, random_state=0)
    evaluate(car, train_test_ratio=0.8)

    # print("\n===============================================")
    # print("Rice Dataset Evaluation [Large Dataset]")
    # print("===============================================")
    # rice = preprocessor.load_rice_dataset().sample(frac=1, random_state=0)
    # evaluate(rice, train_test_ratio=0.8)
    #
    # print("\n===============================================")
    # print("Stroke Dataset Evaluation [Large Dataset]")
    # print("===============================================")
    # stroke = preprocessor.load_stroke_dataset().sample(frac=1, random_state=0)
    # evaluate(stroke, train_test_ratio=0.8)
