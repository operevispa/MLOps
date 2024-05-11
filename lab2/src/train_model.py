# train_model.py - выбор лучшей модели классификации, обучение и сохранение модели
import pandas as pd
import sys, os
import pprint
import pickle
from sklearn.model_selection import train_test_split

# будем использовать несколько моделей классификации, выберем лучшуюю
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report


if __name__ == "__main__":
    # пытаемся прочитать файл с тренировочными данными
    try:
        df_train = pd.read_csv("data/data_train.csv", delimiter=",")
    except:
        print("ERROR: тренировочный датасет не найден!")
        sys.exit(1)

    print("Тренировочный датасет загружен...")
    # достаем из датафрейма тарегеты и помещаем их в y
    y_train = df_train["target"]
    # удалем тарегет и передаем фичи в X
    X_train = df_train.drop("target", axis=1)

    # формируем список моделей классификации
    models = [
        LogisticRegression(random_state=42),
        SGDClassifier(random_state=42),
        KNeighborsClassifier(n_neighbors=3, weights="distance"),
        DecisionTreeClassifier(random_state=42),
        RandomForestClassifier(n_estimators=10, random_state=42),
        SVC(random_state=42),
    ]

    # проводим оценку качества моделей с помощью кросс-валидации
    scores = {}
    for model in models:
        scores[model] = cross_val_score(model, X_train, y_train, cv=5).mean()

    # выводим список моделей с их скор-баллами
    print("Точность предсказания каждой из моеделей:")
    pprint.pprint(scores)

    # определеяем лучшую модель по самому высокому скор.баллу
    best_model = max(scores, key=scores.get)
    print(f"Лучшая модель: {best_model} со средней точностью: {scores[best_model]}")

    # для сохранения лучшей модели, проводим ее обучение
    best_model.fit(X_train, y_train)

    # создаем директорию для хранения модели
    os.makedirs("model", exist_ok=True)
    with open("model/model.pkl", "wb") as file:
        pickle.dump(best_model, file)
        print("Модель сохранена: model/model.pkl")

    print("-" * 100)
