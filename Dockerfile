# Backend for Hugging Face Spaces (Docker SDK) or any container host.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN useradd -m appuser && chown -R appuser /app
USER appuser

ENV HOST=0.0.0.0 PORT=7860
EXPOSE 7860
CMD ["python", "app.py"]
