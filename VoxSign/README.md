# VoxSign Translator

VoxSign is an intelligent sign language translation project that converts hand gestures into text in real time using computer vision and deep learning. It is built as an end-to-end pipeline with a React frontend and FastAPI backend, designed for accessibility-focused communication workflows.

## Project Vision

- Build an intelligent sign language translation system for accessibility.
- Apply NLP-style post-processing (sentence cleanup/grammar stage) with vision-based sign recognition.
- Implement an end-to-end pipeline combining language processing and computer vision models for real-time translation.
- Keep the system deployable and usable on different computers with a modern web stack.

## Core Features

- User-friendly data collection for custom sign datasets.
- LSTM + Dense neural architecture for gesture sequence classification.
- Real-time webcam inference with live prediction smoothing.
- MediaPipe Holistic-based landmark extraction for robust hand tracking.
- Sentence building with optional grammar correction stage.
- React frontend + FastAPI backend for deployable web usage.

## End-to-End Pipeline

1. **Data Collection** (`data_collection.py`)
   - Record gesture sequences from webcam.
   - Extract hand landmarks (126 features per frame: 21 points x 2 hands x xyz).
   - Store training samples under `data/<label>/<sequence>/<frame>.npy`.

2. **Model Training** (`model.py`)
   - Load all labeled landmark sequences.
   - Train a Sequential LSTM + Dense model.
   - Save trained model to `my_model.keras`.

3. **Real-Time Prediction** (`backend_api.py` + `frontend/`)
   - Capture live webcam frames from React.
   - Send frames to FastAPI prediction endpoint.
   - Run MediaPipe + model inference and return:
     - predicted label
     - confidence
     - visualization frame
     - sentence output

## Model and CV Details

- **Vision backbone**: MediaPipe Holistic for landmark extraction.
- **Temporal model**: 3 LSTM layers + Dense classifier.
- **Input shape**: `(10, 126)` sequence window.
- **Prediction smoothing**:
  - confidence threshold (`VOXSIGN_PRED_THRESHOLD`)
  - duplicate cooldown (`VOXSIGN_PRED_COOLDOWN`)
- **CV status feedback**:
  - left/right hand detection status
  - landmark count in current frame

## Tech Stack

- Python, NumPy, TensorFlow/Keras
- OpenCV, MediaPipe
- FastAPI + Uvicorn
- React + Vite

## Prerequisites

- Python 3.10+ recommended
- Node.js 18+ recommended
- Java 8+ (only if you enable full local LanguageTool grammar mode)

## Configure Labels

Edit `labels.json` to define your sign vocabulary:

```json
{
  "actions": ["hello", "thanks", "yes", "no"]
}
```

This is used by data collection and training scripts.

Label tips:
- Use lowercase words.
- Use `_` instead of spaces (example: `thank_you`).
- Keep labels gesture-specific and consistent across recordings.

## Run (Manual)

### Backend

From `VoxSign` folder:

- `pip install -r requirements.txt`
- `python -m uvicorn backend_api:app --host 0.0.0.0 --port 8000`

### Frontend

From `VoxSign/frontend` folder:

- `npm install`
- `npm run dev`

Open `http://localhost:5173`.

If backend runs on another machine, create `frontend/.env`:

- `VITE_API_BASE=http://<BACKEND_HOST>:8000`

## One-Click Windows Workflow

### Collect + Train + Run

- Double click `collect_train_run.bat`

This will:
- collect dataset from webcam,
- train model,
- start backend and frontend.

### Run Existing Model

- Double click `run_all.bat`

This starts backend + frontend using existing `my_model.keras`.

## Dataset-Driven Training

This project now uses a direct dataset-driven flow based on `labels.json`:

- define labels in `labels.json`
- collect data for those exact labels
- train model on collected data
- run realtime inference against the trained label set

## API Endpoints

- `GET /health` - health check
- `POST /session` - create inference session
- `POST /predict` - predict sign from uploaded frame
- `POST /apply-grammar` - sentence grammar/post-processing stage
- `POST /reset` - clear session state

## Deployment Notes

- Backend is deployable via `Procfile` using Uvicorn.
- Frontend can be deployed as static build (`npm run build`) on any static host.
- For production, configure CORS `allow_origins` to trusted frontend domains.

## Current Scope

The current implementation is optimized for **sign-to-text** translation from webcam gesture input. Speech/text to avatar-style sign synthesis can be added as a separate module in a future release.