# test_model.py - тестирование модели на тестовых данных
import pandas as pd
import os, sys
import pprint
import pickle
from sklearn.metrics import classification_report


if __name__ == "__main__":
    # пытаемся прочитать файл с тестовыми данными
    try:
        df_test = pd.read_csv("../data/data_test.csv", delimiter=",")
    except:
        print("ERROR: тестовый датасет не найден!")
        sys.exit(1)

    print("Тестовый датасет загружен...")

    # открываем файл с ранее сохраненной моделью
    with open("../model/model.pkl", "rb") as file:
        model = pickle.load(file)

    print("Модель загружена...")
    # достаем из тестового датафрейма тарегеты и помещаем их в y
    y_test = df_test["target"]
    # удалем тарегет и передаем фичи в X
    X_test = df_test.drop("target", axis=1)

    y_pred = model.predict(X_test)
    print("Результаты тестирования модели на тестовом датасете:")
    print(classification_report(y_test, y_pred))
