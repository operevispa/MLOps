FROM python:3.11

WORKDIR /front

COPY . /front

RUN pip install --no-cache-dir -r requirements.txt


CMD [ "streamlit","run", "./front.py" ]