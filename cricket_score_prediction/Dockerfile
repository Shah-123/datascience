# Build from the repository root:
#   docker build -f cricket_score_prediction/Dockerfile -t odi-score-api .
#   docker run -p 8000:8000 odi-score-api
FROM python:3.11-slim

WORKDIR /app
COPY cricket_score_prediction/requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY cricket_score_prediction/cricket_model.py cricket_score_prediction/api.py cricket_score_prediction/train.py ./
COPY cricket_score_prediction/data ./data
# Train at build time so the container starts instantly.
RUN python train.py

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
