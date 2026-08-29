FROM python:3.12-slim

WORKDIR /app

# Hugging Face writes its cache here. TRANSFORMERS_CACHE is deprecated in
# favour of HF_HOME, so only the latter is set.
ENV HF_HOME=/app/.cache
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# web/ is copied too: docker-compose runs the Streamlit client from this same
# image, and without it that service has no entry point to start.
COPY ./api ./api
COPY ./core ./core
COPY ./web ./web

# Bake both heads into the image so a cold container does not spend its first
# request downloading them from the Hub.
RUN python -c "from core.model_loader import get_irony_model, get_model; get_model(); get_irony_model()"

RUN mkdir -p /app/.cache && chmod -R 777 /app/.cache

EXPOSE 7860

# Hugging Face Spaces and most container hosts inject $PORT. Exec form would
# not expand it, so this goes through sh; "exec" keeps uvicorn as PID 1 so it
# still receives SIGTERM.
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
