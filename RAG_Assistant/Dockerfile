FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Optional: bake the GloVe vectors in so the semantic retriever works offline (adds ~134 MB)
ARG FETCH_EMBEDDINGS=0
RUN if [ "$FETCH_EMBEDDINGS" = "1" ]; then python -m rag_assistant fetch-embeddings; fi
EXPOSE 8501
# Provide an LLM key at runtime:  docker run -p 8501:8501 -e MODELSCOPE_API_KEY=... <image>
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
