VoxSign Translator
VoxSign is an intelligent sign language translation project that converts hand gestures into text in real time using computer vision and deep learning. It supports 76 Indian Sign Language (ISL) words using an LSTM-based neural network trained on the Indian Sign Language Words with Landmarks dataset.

Project Vision
Build an intelligent sign language translation system for accessibility.
Apply NLP-style post-processing (sentence cleanup/grammar stage) with vision-based sign recognition.
Implement an end-to-end pipeline combining language processing and computer vision models for real-time translation.
Keep the system deployable and usable on different computers with a modern web stack.
Core Features
76 ISL word recognition from webcam gestures in real time.
Video dataset preprocessing pipeline (preprocess_videos.py) for training from raw video files.
Bidirectional LSTM + Dense neural architecture for gesture sequence classification.
Real-time webcam inference with prediction smoothing and voting.
MediaPipe-based hand landmark extraction (126 features: 21 points x 2 hands x xyz).
Sentence building with optional grammar correction.
React frontend + FastAPI backend for deployable web usage.
Gradio-based web app (web_app.py) as an alternative UI.
Docker support for containerized deployment.
Supported Signs (76 ISL Words)
afternoon, animal, bad, beautiful, big, bird, blind, cat, cheap, clothing, cold, cow, curved, deaf, dog, dress, dry, evening, expensive, famous, fast, female, fish, flat, friday, good, happy, hat, healthy, horse, hot, hour, light, long, loose, loud, minute, monday, month, morning, mouse, narrow, new, night, old, pant, pocket, quiet, sad, saturday, second, shirt, shoes, short, sick, skirt, slow, small, suit, sunday, tall, thursday, time, today, tomorrow, tuesday, t_shirt, ugly, warm, wednesday, week, wet, wide, year, yesterday, young

End-to-End Pipeline
1. Download Dataset
Download the ISL video dataset from Kaggle:

pip install kaggle
kaggle datasets download kaushikyh/indian-sign-language-words-with-landmarks -p ProcessedData_vivit --unzip
This creates ProcessedData_vivit/<label>/<video>.MOV with 76 sign categories.

2. Preprocess Videos (preprocess_videos.py)
Extract hand landmarks from video files into training-ready .npy sequences:

python preprocess_videos.py --input ProcessedData_vivit --output data --frames 10
This creates data/<label>/<sequence>/<frame>.npy files.

3. Train Model (model.py)
Train the Bidirectional LSTM model on extracted landmarks:

python model.py
Loads labels from labels.json (76 ISL words).
90/10 train/test split with stratification.
EarlyStopping and ReduceLROnPlateau callbacks.
Saves best model to my_model.keras.
4. Real-Time Prediction
Option A: React + FastAPI (recommended for deployment)

# Backend
python -m uvicorn backend_api:app --host 0.0.0.0 --port 8000

# Frontend (in VoxSign/frontend/)
npm install && npm run dev
Open http://localhost:5173.

Option B: Gradio Web App (quick demo)

python web_app.py
Open http://localhost:7860.

Model and CV Details
Vision backbone: MediaPipe Hands for landmark extraction.
Temporal model: 2 Bidirectional LSTM layers + Dense classifier with BatchNormalization and Dropout.
Input shape: (10, 126) sequence window (10 frames, 126 landmark features).
Training: Adam optimizer, categorical crossentropy, class-weighted, EarlyStopping.
Prediction smoothing:
Confidence threshold (VOXSIGN_PRED_THRESHOLD, default 0.60)
Duplicate cooldown (VOXSIGN_PRED_COOLDOWN, default 0.8s)
Voting window for stable predictions
Tech Stack
Python 3.10+, NumPy, TensorFlow/Keras
OpenCV, MediaPipe
FastAPI + Uvicorn (backend API)
React + Vite (frontend)
Gradio (alternative web UI)
Docker (containerized deployment)
Quick Start
Prerequisites
Python 3.10+
Node.js 18+
A trained model file my_model.keras (train with model.py or use pre-trained)
Install Dependencies
cd VoxSign
pip install -r requirements.txt
cd frontend && npm install && cd ..
Run Backend + Frontend
# Terminal 1: Backend
python -m uvicorn backend_api:app --host 0.0.0.0 --port 8000

# Terminal 2: Frontend
cd frontend && npm run dev
Open http://localhost:5173.

If backend runs on another machine, create frontend/.env:

VITE_API_BASE=http://<BACKEND_HOST>:8000
Docker Deployment
Build and run:

cd VoxSign
docker build -t voxsign .
docker run -p 7860:7860 voxsign
This runs the Gradio web app on port 7860.

Project Structure
VoxSign/
  backend_api.py          # FastAPI backend with prediction endpoints
  web_app.py              # Gradio web app (alternative UI)
  model.py                # Model training script
  preprocess_videos.py    # Video-to-landmark extraction pipeline
  data_collection.py      # Live webcam data collection
  main.py                 # Standalone real-time prediction (desktop)
  my_functions.py         # Shared MediaPipe utility functions
  action_config.py        # Label loading configuration
  labels.json             # 76 ISL word labels
  label_aliases.json      # Label alias mappings
  requirements.txt        # Python dependencies
  Dockerfile              # Container deployment config
  Procfile                # Render/Heroku deployment
  my_model.keras          # Trained model (not in git, train or download)
  ProcessedData_vivit/    # Raw video dataset (not in git, download from Kaggle)
  data/                   # Extracted landmark sequences (not in git)
  frontend/               # React frontend application
    src/
    package.json
    vite.config.js
    index.html
API Endpoints
GET /health - health check
GET /meta - list available sign labels
POST /session - create inference session
POST /predict - predict sign from uploaded frame
POST /apply-grammar - sentence grammar/post-processing
POST /reset - clear session state
Environment Variables
Variable	Default	Description
VOXSIGN_PRED_THRESHOLD	0.60	Minimum confidence for predictions
VOXSIGN_PRED_COOLDOWN	0.8	Seconds between duplicate predictions
VOXSIGN_MARGIN_THRESHOLD	0.15	Min margin between top-2 predictions
VOXSIGN_VOTE_WINDOW	5	Number of recent predictions for voting
VOXSIGN_VOTE_MIN	3	Min votes needed for stable prediction
VOXSIGN_MIN_LANDMARKS	10	Min landmarks required for inference
VOXSIGN_SEED	42	Random seed for reproducible training
VOXSIGN_EPOCHS	80	Training epochs
Dataset
This project uses the Indian Sign Language Words with Landmarks dataset from Kaggle, containing 1166 video recordings across 76 ISL word categories.
