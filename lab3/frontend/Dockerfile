FROM python:3.10-slim

COPY . /frontend
WORKDIR /frontend

#RUN python3 -m venv venv
#ENV PATH="/frontend/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

# Обновление списка пакетов
RUN apt-get update 

# Установка curl
RUN apt-get install -y curl

CMD ["streamlit", "run", "frontend.py"]