import pandas as pd
import joblib
from sklearn.metrics import r2_score


def test_datasets():
    num_ds = 4

    model = joblib.load("data/model.joblib")
    ds = {}
    for i in range(1, num_ds + 1):
        ds[i] = pd.read_csv(f"data/ds-{i}.csv")
        X = ds[i][ds[i].columns[:-1]]
        y = ds[i][ds[i].columns[-1:]]

        # предсказываем данные
        y_pred = model.predict(X)

        # расчет метрики r2
        r2 = r2_score(y, y_pred)
        assert r2 > 0.95, f"The ds-{i}.csv dataset is bad!! R2 = {r2}."
