# импортируем необходимые бибилиотеки
import streamlit as st
import requests
import json

# выводим приверственный тайтл и кратко обозначаем, что делает помощник
st.title("Рак молочной железы")
st.write(
    """Предсказание строится моделью, обученной на датасете 'Рак молочной железы в штате Висконсин', 
    входящем в библиотеку sklearn. Необходимо указать значения 5 параметров 
    (ТОП-5 параметров с лучшей предсказательной силой) и нажать кнопку Предсказать."""
)


# обращаемся по API и получаем данные о "размерностях" параметров
# это необходимо нам для построения слайдера с нормальными границами (минимальные и максимальные значения)
# а также получения названий параметров
ftinfo = requests.get("http://127.0.0.1:8000/get_features_info").json()["features"]

features = []
with st.container():
    for i in range(len(ftinfo)):
        features.append(
            st.slider(
                ftinfo[i]["name"],
                min_value=ftinfo[i]["min"],
                max_value=ftinfo[i]["max"],
                # value=ftinfo[i]["min"],
                # format="%.1f",
            )
        )

# выводим кнопку Предсказать
if st.button("Предсказать"):
    # формируем словарь из фич и их значений, заданных пользователем
    fdict = {f"feature{i+1}": ft for i, ft in enumerate(features)}
    # словарь переводим в json-объект и отправляем post запрос по API на предсказание
    res = requests.post("http://127.0.0.1:8000/predict", json.dumps(fdict))

    # проверяем статус ответа сервера
    if res.status_code == 200:
        st.markdown(f"Опухоль является: **{res.json()['prediction']}**")
    else:
        st.markdown("Ошибка в работе API")
