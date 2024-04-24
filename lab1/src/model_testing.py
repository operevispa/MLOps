import pandas as pd
import sys
import joblib
from pathlib import Path
from common_functions import get_num_dataset
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# базовая директория
BASE_DIR = Path(__file__).resolve().parent


# тренировка модели
def test_model(ds, model):
    # отправляем в X - данные с фичами, а в y - таргеты
    X = ds[ds.columns[:-1]]
    y = ds[ds.columns[-1:]]

    # предсказываем данные
    y_pred = model.predict(X)

    # расчет метрик
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    return {"MSE": mse, "R2": r2}


# основной блок
if __name__ == "__main__":

    print("Start file:", Path(__file__).resolve())

    # определяем количество датасетов, которые нужно использовать для обучения
    num_dataset = get_num_dataset(sys.argv)

    for i in range(num_dataset):
        # читаем тестовый датасеты из файла
        try:
            ds_test = pd.read_csv(f"{BASE_DIR}/test/testds_preprocessed-{i+1}.csv")
        except:
            num_dataset = i
            print(
                f"ERROR: the dataset file '/test/testds_preprocessed-{i+1}.csv' could not be found"
            )
            break

        try:
            model = joblib.load(f"{BASE_DIR}/models/model-{i+1}.pkl")
        except:
            num_dataset = i
            print(
                f"ERROR: the file with model data '/models/model-{i+1}.pkl' could not be found"
            )
            break

        # отправялем тестовый датасет и модель в функцию тестирования модели, обученной на тренировочном датасете
        results = test_model(ds_test, model)
        # выводим метрики проведенного тестирования
        print("MSE=", results["MSE"], "R2=", results["R2"])

    print("-" * 100)
