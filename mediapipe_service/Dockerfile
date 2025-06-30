FROM python:3.12-slim

# WORKDIR /mediapipe_services

COPY /app /app

RUN pip install mediapipe flask

EXPOSE 5000

CMD ["python", "/app/mediapipe_api.py"]