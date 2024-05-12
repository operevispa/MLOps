FROM python:3.10-slim

COPY . /frontend
WORKDIR /frontend

#RUN python3 -m venv venv
#ENV PATH="/frontend/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

CMD streamlit run frontend.py --server.port=8501 --server.address=0.0.0.0