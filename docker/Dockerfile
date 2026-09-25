FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY tests/ tests/
COPY config/ config/
COPY pytest.ini .

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
