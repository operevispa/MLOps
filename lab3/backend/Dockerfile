FROM python:3.10-slim

WORKDIR /backend
COPY data_creation.py train_model.py backend_api.py requirements.txt /backend/
#COPY train_model_creation.py /backend
#COPY backend_api.py /backend
#COPY requirements.txt /backend



#RUN python3 -m venv venv
#ENV PATH="/backend/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "backend_api:app", "--host", "0.0.0.0", "--port", "8000"]
#CMD python3 backend_api.py