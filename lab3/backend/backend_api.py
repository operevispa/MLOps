# backend_api.py - файл с API интерфейсом модели предсказания
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pickle
import os
import train_model 

app = FastAPI()  # Создаем приложение в переменной app

# Путь к файлу модели
MODEL_PATH = '../model/model.pkl'


class Item(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float
    feature5: float


def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        try:
            return pickle.load(MODEL_PATH)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ошибка при загрузке модели: {str(e)}")
    else:
        # Обучение новой модели
        model = train_model.train_model()  # Предполагается, что train_model.train_model() возвращает обученную модель
        pickle.dump(model, MODEL_PATH)
        return model

# Загрузка или обучение модели при запуске скрипта
model = load_or_train_model()


@app.post("/predict")
async def predict(item: Item):
    """
    Принимает POST-запрос с данными для предсказания и возвращает результат.
    """	
  	try:
        #prediction = model.predict([np.array([item.feature1, item.feature2])])
        prediction = model.predict(item)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)