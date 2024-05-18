# стадия 2 - загружаем базовый датасет, корректируем его и снова сохраняем
from sklearn.datasets import load_breast_cancer
import pandas as pd

# загружаем базовый датасет, ранее сохраненный в папке datasets
df = pd.read_csv("./datasets/dataset1.csv", delimiter=",")

# меняем значения во всех строках параметра mean radius на среднее значение
df["mean radius"] = df["mean radius"].mean()

# меняем значения во всех строках параметра mean area на максимальное значение этого параметра
df["mean area"] = df["mean area"].max()

# сохраняем измененный датасет
df.to_csv("./datasets/dataset1.csv", index=False)
print("Датасет с изменениями mean radius и mean area сохранен: datasets/dataset1.csv")
