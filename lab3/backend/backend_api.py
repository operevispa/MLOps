# backend_api.py - файл с API интерфейсом модели предсказания
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
import os
import uvicorn

from train_model import train_model

app = FastAPI()  # Создаем приложение в переменной app

# Пути к файлам
MODEL_PATH = "model/model.pkl"
SCALER_PATH = "data/scaler.pkl"
TARGET_PATH = "data/target_names.pkl"
FEATURES_PATH = "data/features_info.pkl"


class Item(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float
    feature5: float


# загружаем ранее сохраненную модель
def load_model():
    # проверяем наличии файла model.pkl
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            # формируем ошибку с кодом 500 (внутренняя ошибка сервера)
            raise HTTPException(
                status_code=500, detail=f"Ошибка загрузки модели: {str(e)}"
            )
    else:
        # файла с моделью не оказалось,
        # поэтому запускаем функцию тренировки модели (для учебной задачи допустимо)
        model = (
            train_model.train_model()
        )  # Предполагается, что train_model.train_model() возвращает обученную модель
        pickle.dump(model, MODEL_PATH)
        return model


# загружаем скалер, имена целевых значений и информацию о фичах
def load_data():
    # проверяем наличие файлов
    try:
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)

        with open(TARGET_PATH, "rb") as f:
            target_names = pickle.load(f)

        with open(FEATURES_PATH, "rb") as f:
            features = pickle.load(f)
    except Exception as e:
        # формируем ошибку с кодом 500 (внутренняя ошибка сервера)
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки данных!")

    return scaler, target_names, features


# загрузка или обучение модели при запуске backend
model = load_model()
# загрузка скалера, имен целевой переменной, информации по фичам
scaler, target_names, features = load_data()


@app.post("/predict")
def predict(item: Item):
    """
    Принимает POST-запрос с данными для предсказания и возвращает результат.
    """
    # Преобразование объекта Item в словарь
    item_dict = dict(item)
    # Преобразование словаря в список значений
    item_list = list(item_dict.values())

    # проводим преобразование входных данных с использование сохраненного скалера
    ft_scaled = scaler.transform([item_list])
    # формируем датафрейм с названием параметров, иначе kNearestNeigbours может не сработать
    df_predict = pd.DataFrame(ft_scaled, columns=features.columns)
    # собственно предсказывам значение

    try:

        prediction = target_names[model.predict(df_predict)][0]
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get_features_info")
async def get_features_info():
    """
    Возвращает информацию о параметрах, которые используются моделью для предсказания.
    Возвращается pandas dataframe, которые представляет собой describe() от датафрейма с переменными.
    Т.е. в нем содержатся имена переменных, минимальные, средние и максимальные значения переменных.
    """
    if not features.empty:
        res = []
        for f in features.columns:
            res.append(
                {"name": f, "min": features[f]["min"], "max": features[f]["max"]}
            )
        return {"features": res}
    else:
        raise HTTPException(
            status_code=500, detail="There is no data about the features"
        )


if __name__ == "__main__":
    uvicorn.run(app, port=8000)
