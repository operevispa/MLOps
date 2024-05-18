# стадия 4 - загружаем датасет, корректируем его и снова сохраняем
from sklearn.datasets import load_breast_cancer
import pandas as pd

# загружаем базовый датасет, ранее сохраненный в папке datasets
df = pd.read_csv("./datasets/dataset1.csv", delimiter=",")
print(df.describe())

# удаляем еще 4 параметра из датасета и меняем
df = df.drop(
    ["mean perimeter", "mean area", "mean smoothness", "mean compactness"], axis=1
)
# меняем mean concavity на минимальные значения
df["mean concavity"] = df["mean concavity"].min()


print(df.describe())

# сохраняем измененный датасет
df.to_csv("./datasets/dataset1.csv", index=False)
print(
    "Датасет с 4 удаленными и 1 измененным параметром сохранен: datasets/dataset1.csv"
)
