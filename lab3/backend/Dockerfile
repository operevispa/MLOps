FROM python:3.10-slim

COPY . /backend
WORKDIR /backend
#RUN python3 -m venv venv
#ENV PATH="/backend/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
CMD uvicorn backend_api:app --host 0.0.0.0 --port 8000
#CMD python3 backend_api.py