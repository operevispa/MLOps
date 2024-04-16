import numpy as np
import pandas as pd
import sys, os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path
from common_functions import get_num_dataset


# базовая директория
BASE_DIR = Path(__file__).resolve().parent


def generate_loan_dataset(sample_size=50, anomaly_ratio=0.05):
    # инициализируем random
    rng = np.random.default_rng(seed=42)
    # генерируем значения фичас age, incom, cred_rating, employment
    age = rng.integers(low=18, high=65, size=sample_size)
    income = np.round(80000 * rng.lognormal(mean=0, sigma=0.3, size=sample_size), 2)
    credit_rating = rng.integers(low=100, high=999, size=sample_size)
    employment = np.round(40 * rng.random(size=sample_size), 1)
    # генерируем шум
    # noises = 1000 * rng.random(size=sample_size)
    noises = 0

    # рассчитываем целевую переменную - кредитный лимит
    credit_limit = np.round(
        (age * 1000 + income * 10 + credit_rating * 100 + employment * 2000 + noises), 0
    )
    # print(target)

    # считаем количесвто аномалий, которые будем генерировать
    anomaly_half_size = int(anomaly_ratio / 2 * sample_size)
    # генерируем отдельно высокие и низкие аномалии (поровну)
    anomaly = np.concatenate(
        (
            np.round(5000000 * rng.random(size=anomaly_half_size), 2) + 10000000,
            np.round(20000 * rng.random(size=anomaly_half_size), 2),
        ),
        axis=0,
    )
    # перемешиваем аномалии внутри (высокие с низкими)
    rng.shuffle(anomaly)
    # корректируем вывод числе в принте (кол-во знаков после запятой 2, и без степени 10)
    # np.set_printoptions(precision=2, suppress=True)
    # print(anomaly)

    # создаем массив индексов, в значения которых будем подставлять аномалии
    ind = rng.integers(low=0, high=sample_size, size=anomaly_half_size * 2)
    # print('ind=',ind)
    # подставляем аномалии в таргеты
    credit_limit[ind] = np.round((anomaly), 0)
    # print('target=',target)

    # делаем датафрейм из массивов фич и таргетов
    df = pd.DataFrame(
        {
            "age": age,
            "income": income,
            "credit_rating": credit_rating,
            "employment": employment,
            "credit_limit": credit_limit,
        }
    )
    # смотрим первые значения датафрейма
    # print(df.head(5).to_markdown())
    # смотрим гистограмму фич и таргета
    # df.hist(bins=100, figsize=(18, 9))
    return df


# основной блок
if __name__ == "__main__":
    print("Start file:", Path(__file__).resolve())

    # определяем количество датасетов, которые нужно создать
    num_dataset = get_num_dataset(sys.argv)

    # создаем две директории для хранения датасетов
    os.makedirs(BASE_DIR / "train", exist_ok=True)
    os.makedirs(BASE_DIR / "test", exist_ok=True)

    for i in range(num_dataset):
        # генерируем датасет
        ds = generate_loan_dataset(sample_size=1000, anomaly_ratio=0.00)
        # делим датасет на тренировочный и тестовый
        ds_train, ds_test = train_test_split(ds, test_size=0.2, random_state=42)
        ds_train.to_csv(f"{BASE_DIR}/train/trainds-{i+1}.csv", index=False)
        ds_test.to_csv(f"{BASE_DIR}/test/testds-{i+1}.csv", index=False)

    print(f"{num_dataset} datasets have been created!")
    print(f"Datasets with training data in the |train| folder")
    print(f"Datasets with testing data in the |test| folder")
    print("-" * 100)
