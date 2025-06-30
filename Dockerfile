FROM python:3.12-slim as builder

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.12-slim

WORKDIR /app

COPY --from=builder . .

COPY ./app ./app

EXPOSE 8501

CMD ["streamlit", "run", "app/app.py", "--server.port=8501"]
