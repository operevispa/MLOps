# data_creation.py - загрузка датасета и подготовка данных
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle
import pandas as pd
import os


def create_ds():
    # используем стандартный датасет из библиотеки sklearn (рак груди)
    # данный датасет используется для задач классификации
    ds = load_breast_cancer()
    X = ds.data
    y = ds.target
    print("Датасет загружен...")
    print("Количество параметров=", ds.feature_names.shape[0])

    # для оптимизации модели выбраем 5 лучших фич из 30
    selector = SelectKBest(chi2, k=5)
    X_top = selector.fit_transform(X, y)
    mask = selector.get_support()

    # определяем и выводим лучше 5 параметров
    columns = ds.feature_names
    new_features = columns[mask]
    print("Отобраны ТОП-5 параметров:", list(new_features))

    # разделяем датасет на тренировочную и тестовую выборки в соотношениии 70 на 30
    X_train, X_test, y_train, y_test = train_test_split(
        X_top, y, test_size=0.3, random_state=42, shuffle=True
    )

    # проводим нормализацию данных (для исключения искажения из-за масштаба данных)
    scaler = MinMaxScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Данные датасета обработаны и разделены на тренировочную и тестовую выборки")

    # создаем папку для хранения файлов
    os.makedirs("data", exist_ok=True)

    # формируем датафрейми с тренировочными данными и тестовыми, и выгружаем их на диск
    df_train = pd.DataFrame(X_train_scaled, columns=new_features)
    df_train["target"] = y_train
    df_train.to_csv("data/data_train.csv", index=False)
    print("Тренировочный датасет сохранен: data/data_train.csv")

    df_test = pd.DataFrame(X_test_scaled, columns=new_features)
    df_test["target"] = y_test
    df_test.to_csv("data/data_test.csv", index=False)
    print("Тестовый датасет сохранен: data/data_train.csv")

    # сохраняем скалер для использования на новых данных
    with open("data/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("Скалер сохранен: data/scaler.pkl")

    # сохраняем именая таргетов для использования на новых данных
    with open("data/target_names.pkl", "wb") as f:
        pickle.dump(ds.target_names, f)
    print("Имена целевых значений сохранены: data/target_names.pkl")

    # сохраняем описание датафрейма в файл, из которого мы сможем взять необходимые данные по ТОП-5 фич
    # а именно, имена параметром, их минимальные и максимальные значения и тп
    with open("data/features_info.pkl", "wb") as f:
        pickle.dump(pd.DataFrame(X_top, columns=new_features).describe(), f)
    print("Данные по ТОП-5 лучших параметров сохранены: data/features_info.pkl")

    print("-" * 100)


if __name__ == "__main__":
    create_ds()
