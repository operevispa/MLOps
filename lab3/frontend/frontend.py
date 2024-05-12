# импортируем необходимые бибилиотеки
import streamlit as st
import requests
import json
import time

# выводим приверственный тайтл и кратко обозначаем, что делает помощник
st.title("Рак молочной железы")
st.write(
    """Предсказание строится моделью, обученной на датасете 'Рак молочной железы в штате Висконсин', 
    входящем в библиотеку sklearn. Необходимо указать значения 5 параметров 
    (ТОП-5 параметров с лучшей предсказательной силой) и нажать кнопку Предсказать."""
)

ftinfo = {}
with st.spinner("Ожидаем запуск backend API..."):
    # ожидаем запуск backend
    while True:
        # обращаемся по API и получаем данные о "размерностях" параметров
        # это необходимо нам для построения слайдера с нормальными границами (минимальные и максимальные значения)
        # а также получения названий параметров
        try:
            response = requests.get("http://localhost:8000/get_features_info")
            if response.status_code == 200:
                ftinfo = response.json()["features"]
                break
            else:
                time.sleep(10)
        except requests.exceptions.RequestException as e:
            time.sleep(10)  # Засыпаем на 5 секунд перед следующей проверкой


features = []
with st.container():
    for i in range(len(ftinfo)):
        features.append(
            st.slider(
                ftinfo[i]["name"],
                min_value=ftinfo[i]["min"],
                max_value=ftinfo[i]["max"],
            )
        )

# выводим кнопку Предсказать
if st.button("Предсказать"):
    # формируем словарь из фич и их значений, заданных пользователем
    fdict = {f"feature{i+1}": ft for i, ft in enumerate(features)}
    # словарь переводим в json-объект и отправляем post запрос по API на предсказание
    res = requests.post("http://localhost:8000/predict", json.dumps(fdict))

    # проверяем статус ответа сервера
    if res.status_code == 200:
        st.markdown(f"Опухоль является: **{res.json()['prediction']}**")
    else:
        st.markdown("Ошибка в работе API")
