FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables to fix permission issues on Hugging Face
ENV HF_HOME=/app/.cache
ENV TRANSFORMERS_CACHE=/app/.cache
ENV PORT=7860

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API code folder
COPY ./api ./api

# Create the cache directory and fix permissions for the HF user
RUN mkdir -p /app/.cache && chmod -R 777 /app/.cache

# Expose the Hugging Face port
EXPOSE 7860

# Command to start the API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]