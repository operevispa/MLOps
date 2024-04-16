import numpy as np
import pandas as pd
import sys, os
import joblib
from pathlib import Path
from sklearn.linear_model import LinearRegression
from common_functions import get_num_dataset
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# базовая директория
BASE_DIR = Path(__file__).resolve().parent


# получение коэффициентов модели
def get_coefs(model):
    B0 = model.intercept_[0]
    B = model.coef_
    return B0, B


# написание в наглядном виде формулы регрессии
def print_model(B0, B, features_names):
    # определяем смещение в качестве первого элемента формулы
    line = str(np.round(B0, 3))
    # проходимся в цикле по всем фичам и их значениям и добавляем в формулу
    for f, b in zip(features_names, *np.round(B, 3)):
        line += f"{b if b <0 else '+'+str(b)}*{f}"

    print("F =", line)


# функция расчета метрик линейной регрессии
def calculate_metric(model_pipe, X, y, metric=r2_score, **kwargs):
    y_model = model_pipe.predict(X)
    return metric(y, y_model, **kwargs)


# тренировка модели
def train_model(ds):
    # отправляем в X - данные с фичами, а в y - таргеты
    X = ds[ds.columns[:-1]]
    y = ds[ds.columns[-1:]]

    # создаем модель линейной регрессии
    model = LinearRegression()

    # Тренируем модель
    model.fit(X, y)

    # предсказываем данные
    y_pred = model.predict(X)

    # расчет метрик
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    return model, {"MSE": mse, "R2": r2}


# основной блок
if __name__ == "__main__":
    print("Start file:", Path(__file__).resolve())

    # создаем директорию для хранения моделей
    os.makedirs(BASE_DIR / "models", exist_ok=True)

    # определяем количество датасетов, которые нужно использовать для обучения
    num_dataset = get_num_dataset(sys.argv)

    for i in range(num_dataset):
        # читаем датасеты из файла (ов)
        try:
            ds_train = pd.read_csv(
                f"{BASE_DIR}/train/trainds_preprocessed-{i+1}.csv", delimiter=","
            )
        except:
            num_dataset = i
            print(
                f"ERROR: the dataset file '/train/trainds_preprocessed-{i+1}.csv' could not be found"
            )
            break

        # отправялем тренировочный датасет в функцию создания и обучения модели
        model, results = train_model(ds_train)
        # получаем коэффициенты модели
        B0, B = get_coefs(model)
        # формируем список фич
        features_names = list(ds_train.columns[:-1])
        # выводим формулу уравнения линейной регрессии
        print_model(B0, B, features_names)
        # и выводим метрики полученной модели
        print("MSE=", results["MSE"], "R2=", results["R2"])

        # сохраняем модель
        joblib.dump(model, f"models/model-{i+1}.pkl")
        print("The model is trained and saved in |model| folder\n")

    print(f"{num_dataset} dataset`s has been preprocessed!")
    print("-" * 100)
