# стадия 3 - загружаем базовый датасет, корректируем его и снова сохраняем
from sklearn.datasets import load_breast_cancer
import pandas as pd

# загружаем базовый датасет, ранее сохраненный в папке datasets
df = pd.read_csv("./datasets/dataset1.csv", delimiter=",")

# удаляем два параметра из датасета mean radius и mean texture
df = df.drop(["mean radius", "mean texture"], axis=1)
print(df.describe())

# сохраняем измененный датасет
df.to_csv("./datasets/dataset1.csv", index=False)
print("Датасет с удаленными mean radius и mean texture сохранен: datasets/dataset1.csv")
