# импортируем необходимые бибилиотеки
import streamlit as st

# from transformers import pipeline


@st.cache_resource
def load_model():
    model = "IlyaGusev/mbart_ru_sum_gazeta"
    return model
    # return pipeline('summarization', model=model)


# Делаем условие, чтобы код не отрабатывался во время импорта в другие модули, в частности для тестирования.
if __name__ == "__main__":
    # выводим приверственный тайтл и кратко обозначаем, что делает помощник
    st.title("Давай определим к какому типу ириса относится твой вариант :blossom:")