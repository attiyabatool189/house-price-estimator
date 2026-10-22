# Works on Hugging Face Spaces (Docker), Railway, Fly.io, Google Cloud Run, ...
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face Spaces expects port 7860; other platforms set PORT themselves.
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 120"]
