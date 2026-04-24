🚀 VoxSign Translator
🧠 Real-Time Sign Language to Text using AI

VoxSign is an intelligent sign language translation system that converts hand gestures into meaningful text in real time using Computer Vision + Deep Learning.

🌍 Project Vision

Make communication accessible for the hearing & speech impaired

Combine Computer Vision + NLP-style sentence correction

Build a real-time, deployable web-based system

Create an end-to-end AI pipeline from gesture → text → sentence

✨ Key Features

✅ Real-time ISL gesture recognition (76 words)

✅ Bidirectional LSTM model for sequence learning

✅ MediaPipe-based hand landmark extraction (126 features)

✅ Smart prediction smoothing & voting system

✅ Sentence formation + grammar correction

✅ React + FastAPI web app (production-ready)

✅ Gradio demo UI for quick testing

✅ Docker support for easy deployment

🧠 How It Works (Pipeline)

Video Input → Hand Detection → Landmark Extraction → LSTM Model → Prediction

📊 Model Details

📌 Input Shape: (10 frames, 126 features)

📌 Architecture

Bidirectional LSTM ×2

Dense Layers

Batch Normalization + Dropout

📌 Optimizer: Adam

📌 Loss: Categorical Crossentropy

📌 Training: EarlyStopping + LR Scheduler

🖐️ Supported Signs (76 ISL Words)

Includes words like:
hello, good, bad, happy, sad, morning, night, dog, cat, fast, slow, today, tomorrow, monday...

📌 Covers

Emotions

Animals

Time-related words

Daily conversation vocabulary

🛠️ Tech Stack

🔹 Backend

Python

FastAPI + Uvicorn

TensorFlow / Keras

OpenCV

MediaPipe

🔹 Frontend

React + Vite

🔹 Others

Gradio (demo UI)

Docker (deployment)

📌 1. Install Dependencies

pip install -r requirements.txt
cd frontend
npm install
cd ..

📌 2. Download Dataset
pip install kaggle
kaggle datasets download kaushikyh/indian-sign-language-words-with-landmarks -p ProcessedData_vivit --unzip

📌 3. Preprocess Data
python preprocess_videos.py --input ProcessedData_vivit --output data --frames 10

📌 5. Train Model
python model.py

🚀 Run the Project
🔹 Option 1: Full Web App (Recommended)
Backend
python -m uvicorn backend_api:app --host 0.0.0.0 --port 8000
Frontend
cd frontendnpm run dev
👉 Open: http://localhost:5173

🔹 Option 2: Gradio Demo
python web_app.py
👉 Open: http://localhost:7860

🐳 Docker Deployment
docker build -t voxsign .docker run -p 7860:7860 voxsign


<img width="877" height="540" alt="image" src="https://github.com/user-attachments/assets/c200924e-dd5a-4772-83b8-6bc94f5a6288" />


🔌 API Endpoints


GET /health → Check server status


POST /predict → Predict gesture


POST /apply-grammar → Improve sentence


POST /reset → Reset session



⚡ Key Highlights (For Recruiters 👀)


🔥 End-to-end AI + Web integration project


🔥 Real-time computer vision pipeline


🔥 Combines Deep Learning + NLP concepts


🔥 Deployable using Docker + Web UI


🔥 Solves real-world accessibility problem



📊 Dataset
Indian Sign Language Words with Landmarks (Kaggle)


🎥 1166 videos


🖐️ 76 sign classes



💡 Future Improvements


Add sentence-level translation (full NLP model)


Increase vocabulary beyond 76 words


Mobile app integration


Real-time speech output



❤️ Contribution
Contributions are welcome!
Feel free to fork, improve, and submit a PR 🚀

