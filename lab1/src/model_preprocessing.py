import numpy as np
import pandas as pd
import sys, os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from common_functions import get_num_dataset

# базовая директория
BASE_DIR = Path(__file__).resolve().parent


def quantile_info(df, threshold=0.05):
    for col in df.select_dtypes(include=np.number):
        print("колонка:", col)
        print("квантиль", threshold, "=", df[col].quantile(threshold).round(2))
        print("квантиль", 1 - threshold, "=", df[col].quantile(1 - threshold).round(2))
        print()


def quantile_replacer(df, threshold=0.05):
    # избавляемся от аномалий. Размер чувствительности к аномалии определяется аргументом trashhold
    coln = "credit_limit"
    df.loc[df[coln] > df[coln].quantile(1 - threshold), [coln]] = df[coln].quantile(
        1 - threshold
    )
    df.loc[df[coln] < df[coln].quantile(threshold), [coln]] = df[coln].quantile(
        threshold
    )

    """ 
    # по всем данным - и в фичах и в таргете
    for col in df.select_dtypes(include=np.number):
        df.loc[df[col] > df[col].quantile(1 - threshold), [col]] = df[col].quantile(
            1 - threshold
        )
        df.loc[df[col] < df[col].quantile(threshold), [col]] = df[col].quantile(
            threshold
        )
    """
    return df


# функция стандартизации параметров (фич)
# на вход подается pandas dataframe, считанный из файла
def preprocess(ds):
    # формируем список фич
    # в датасете таргетная переменная в последней колонке, отсекаем ее (она не стандартизуется)
    features = ds.columns[:-1]

    # создаем объект класса стандартизации
    scaler = StandardScaler()
    # проводим трансформацию данных в стандартные
    scaled_data = scaler.fit_transform(ds[features])
    ds[features] = scaled_data

    return ds


# функция стандартизации всех данных, в т.ч. таргета
# на вход подается pandas dataframe, считанный из файла
def preprocess2(ds):
    # создаем объект класса стандартизации
    scaler = StandardScaler()
    # проводим трансформацию данных в стандартные
    scaled_data = scaler.fit_transform(ds)
    ds[ds.columns] = scaled_data
    return ds


# основной блок
if __name__ == "__main__":
    print("Start file:", Path(__file__).resolve())
    # определяем количество датасетов, которые нужно предобработать
    num_dataset = get_num_dataset(sys.argv)

    data_type = {
        "age": np.float64,
        "income": np.float64,
        "credit_rating": np.float64,
        "employment": np.float64,
        "credit_limit": np.float64,
    }

    for i in range(num_dataset):
        # читаем датасеты из файла (ов)
        try:
            ds_train = pd.read_csv(
                f"{BASE_DIR}/train/trainds-{i+1}.csv", delimiter=",", dtype=data_type
            )
        except:
            num_dataset = i
            print(
                f"ERROR: the dataset file '/train/trainds-{i+1}.csv' could not be found"
            )
            break

        # quantile_info(ds_train, threshold=0.06)
        # избавляемся от аномалий
        ds_train = quantile_replacer(ds_train, threshold=0.06).round(2)
        # отправялем датасет в предобработку (стандаризуем все кроме таргета)
        # ds_train = preprocess2(ds_train)
        # ds_train = preprocess2(ds_train)
        # записываем стандартизованные данные в новый файл
        ds_train.to_csv(f"{BASE_DIR}/train/trainds_preprocessed-{i+1}.csv", index=False)

        # аналогичное проделываем с файлами с тестовыми данными
        ds_test = pd.read_csv(
            f"{BASE_DIR}/test/testds-{i+1}.csv", delimiter=",", dtype=data_type
        )
        ds_test = quantile_replacer(ds_test, threshold=0.06).round(2)
        ds_test.to_csv(f"{BASE_DIR}/test/testds_preprocessed-{i+1}.csv", index=False)

    print(f"{num_dataset} dataset`s has been preprocessed!")
    print("-" * 100)
