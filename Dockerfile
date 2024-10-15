FROM python:3.12
WORKDIR /client_default_pred
COPY /requirements.txt /client_default_pred/requirements.txt
RUN pip install -r /client_default_pred/requirements.txt
COPY /app /client_default_pred/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]