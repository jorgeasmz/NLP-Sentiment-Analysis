---
title: NLP Sentiment Analysis
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

# NLP Sentiment Analysis Service

A modular sentiment analysis system utilizing a fine-tuned DistilBERT model, served via a FastAPI backend and a Streamlit frontend.

## Technical Architecture

The project implements a layered architecture to ensure separation of concerns between model inference, API handling, and the user interface.

*   **Core (`/core`)**: Handles model loading and inference logic. Implements a Singleton pattern for the Hugging Face pipeline to ensure the model is loaded into memory only once.
*   **API (`/api`)**: A RESTful service built with FastAPI. Manages request validation (Pydantic), endpoint routing, and integration with the Core service.
*   **Frontend (`/web`)**: A Streamlit interface for interactive testing and visualization of model outputs.

## Technology Stack

*   **Model**: `distilbert-base-uncased-finetuned-sst-2-english` (Hugging Face Transformers)
*   **Backend**: FastAPI, Uvicorn
*   **Frontend**: Streamlit
*   **Containerization**: Docker, Docker Compose

## Installation

Ensure Python 3.8+ is installed.

```bash
pip install -r requirements.txt
```

## Running the Application

### 1. Start the Backend API

The API server initializes the model upon startup.

```bash
uvicorn api.main:app --reload
```

*   **URL:** `http://localhost:8000`
*   **Swagger Docs:** `http://localhost:8000/docs`

### 2. Start the Frontend UI

Open a new terminal session to launch the Streamlit dashboard.

```bash
streamlit run web/app.py
```

*   **URL:** `http://localhost:8501`

## API Reference

### Analyze Text

**Endpoint**: `POST /predict`

Analyzes the sentiment of the provided text string.

**Request Body** (`application/json`):

```json
{
  "text": "I love this product!"
}
```

**Response** (`application/json`):

```json
{
  "label": "POSITIVE",
  "score": 0.9998
}
```

## Configuration

Model parameters can be modified in `core/config.py`.

```python
MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
```

## Docker Execution

To run the entire stack using Docker Compose:

```bash
docker compose up --build
```
