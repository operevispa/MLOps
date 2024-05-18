# стадия 1 - загружаем базовый датасет и сохраняем его в папку datasets без какой-либо обработки
from sklearn.datasets import load_breast_cancer
import pandas as pd

# загружаем датасет "рак груди" и сохраняем его в папку datasets
ds = load_breast_cancer()
X = ds.data
y = ds.target
df = pd.DataFrame(ds.data, columns=ds.feature_names)
df["target"] = y
df.to_csv("./datasets/dataset1.csv", index=False)
print("Базовый датасет сохранен: datasets/dataset1.csv")
