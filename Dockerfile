FROM python:3.12-slim-buster
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python3,"application.py"]