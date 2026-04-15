import os
import string
import uuid
import time
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from action_config import load_actions

# --------------- Model loading ---------------
import onnxruntime as ort

actions = np.array(load_actions(default="a,b"))
ort_session = ort.InferenceSession("my_model.onnx")
model_output_size = ort_session.get_outputs()[0].shape[-1]
if len(actions) != model_output_size:
    actions = actions[:model_output_size]
INPUT_NAME = ort_session.get_inputs()[0].name
OUTPUT_NAME = ort_session.get_outputs()[0].name
print(f"[VoxSign] ONNX model loaded: {len(actions)} labels, output size {model_output_size}", flush=True)
# ---------------------------------------------------------

PREDICTION_CONFIDENCE_THRESHOLD = float(os.environ.get("VOXSIGN_PRED_THRESHOLD", "0.60"))
PREDICTION_COOLDOWN_SECONDS = float(os.environ.get("VOXSIGN_PRED_COOLDOWN", "0.8"))
PREDICTION_MARGIN_THRESHOLD = float(os.environ.get("VOXSIGN_MARGIN_THRESHOLD", "0.15"))
PREDICTION_VOTE_WINDOW = int(os.environ.get("VOXSIGN_VOTE_WINDOW", "5"))
PREDICTION_VOTE_MIN = int(os.environ.get("VOXSIGN_VOTE_MIN", "3"))
MIN_LANDMARKS_FOR_INFERENCE = int(os.environ.get("VOXSIGN_MIN_LANDMARKS", "10"))

session_states: Dict[str, Dict[str, object]] = {}


def _new_state() -> Dict[str, object]:
    return {
        "sentence": [],
        "keypoints": [],
        "last_prediction": None,
        "last_prediction_at": 0.0,
        "grammar_result": "",
        "current_prediction": "",
        "current_confidence": 0.0,
        "stable_prediction": "",
        "stable_confidence": 0.0,
        "prediction_history": [],
        "frame_index": 0,
    }


def _merge_letters(sentence: List[str]) -> None:
    if os.environ.get("VOXSIGN_LETTER_MODE", "0") != "1":
        return
    if len(sentence) < 2:
        return
    last = sentence[-1]
    prev = sentence[-2]
    alpha = set(string.ascii_lowercase + string.ascii_uppercase)
    if len(last) == 1 and len(prev) == 1 and last in alpha and prev in alpha:
        sentence[-1] = (prev + last).capitalize()
        sentence.pop(-2)


def _state_for(session_id: str) -> Dict[str, object]:
    if session_id not in session_states:
        session_states[session_id] = _new_state()
    return session_states[session_id]


class SessionPayload(BaseModel):
    session_id: str


class PredictPayload(BaseModel):
    session_id: str
    landmarks: List[float]  # 126-dim vector: [left_hand_63, right_hand_63]
    landmarks_count: int = 0


app = FastAPI(title="VoxSign API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/meta")
def meta() -> Dict[str, object]:
    return {"actions": actions.tolist()}


@app.post("/session")
def create_session() -> Dict[str, str]:
    session_id = str(uuid.uuid4())
    session_states[session_id] = _new_state()
    return {"session_id": session_id}


@app.post("/predict")
def predict(payload: PredictPayload) -> Dict[str, object]:
    state = _state_for(payload.session_id)

    sentence: List[str] = state["sentence"]  # type: ignore[assignment]
    keypoints: List[np.ndarray] = state["keypoints"]  # type: ignore[assignment]
    last_prediction: Optional[str] = state["last_prediction"]  # type: ignore[assignment]
    last_prediction_at: float = state["last_prediction_at"]  # type: ignore[assignment]
    grammar_result: str = state["grammar_result"]  # type: ignore[assignment]
    current_prediction: str = state["current_prediction"]  # type: ignore[assignment]
    current_confidence: float = state["current_confidence"]  # type: ignore[assignment]
    stable_prediction: str = state["stable_prediction"]  # type: ignore[assignment]
    stable_confidence: float = state["stable_confidence"]  # type: ignore[assignment]
    prediction_history: List[str] = state["prediction_history"]  # type: ignore[assignment]
    frame_index: int = state["frame_index"]  # type: ignore[assignment]

    landmarks_count = payload.landmarks_count
    kp = np.array(payload.landmarks, dtype=np.float32)
    if kp.shape[0] != 126:
        raise HTTPException(status_code=400, detail=f"Expected 126 landmarks, got {kp.shape[0]}")

    keypoints.append(kp)
    frame_index += 1
    if len(keypoints) > 10:
        keypoints.pop(0)

    if len(keypoints) == 10 and landmarks_count >= MIN_LANDMARKS_FOR_INFERENCE:
        model_input = np.array(keypoints, dtype=np.float32)[np.newaxis, :, :]
        prediction = ort_session.run([OUTPUT_NAME], {INPUT_NAME: model_input})[0][0]

        current_confidence = float(np.max(prediction))
        current_prediction = str(actions[int(np.argmax(prediction))])
        if len(prediction) > 1:
            sorted_conf = np.sort(prediction)
            confidence_margin = float(sorted_conf[-1] - sorted_conf[-2])
        else:
            confidence_margin = current_confidence

        accepted_now = (
            current_confidence >= PREDICTION_CONFIDENCE_THRESHOLD
            and confidence_margin >= PREDICTION_MARGIN_THRESHOLD
        )
        prediction_history.append(current_prediction if accepted_now else "")
        if len(prediction_history) > PREDICTION_VOTE_WINDOW:
            prediction_history.pop(0)

        counts: Dict[str, int] = {}
        for item in prediction_history:
            if item:
                counts[item] = counts.get(item, 0) + 1
        if counts:
            top_label, top_count = max(counts.items(), key=lambda kv: kv[1])
            if top_count >= PREDICTION_VOTE_MIN:
                stable_prediction = top_label
                stable_confidence = current_confidence

        now = time.time()
        if stable_prediction and stable_prediction != last_prediction and (
            now - last_prediction_at
        ) >= PREDICTION_COOLDOWN_SECONDS:
            sentence.append(stable_prediction)
            last_prediction = stable_prediction
            last_prediction_at = now

    if len(sentence) > 7:
        sentence[:] = sentence[-7:]
    if sentence:
        sentence[0] = sentence[0].capitalize()
    _merge_letters(sentence)

    state["sentence"] = sentence
    state["keypoints"] = keypoints
    state["last_prediction"] = last_prediction
    state["last_prediction_at"] = last_prediction_at
    state["current_prediction"] = current_prediction
    state["current_confidence"] = current_confidence
    state["stable_prediction"] = stable_prediction
    state["stable_confidence"] = stable_confidence
    state["prediction_history"] = prediction_history
    state["frame_index"] = frame_index

    return {
        "sentence": " ".join(sentence),
        "grammar_result": grammar_result,
        "current_prediction": current_prediction,
        "confidence": round(current_confidence, 4),
        "stable_prediction": stable_prediction,
        "stable_confidence": round(stable_confidence, 4),
        "threshold": PREDICTION_CONFIDENCE_THRESHOLD,
    }


@app.post("/apply-grammar")
def apply_grammar(payload: SessionPayload) -> Dict[str, str]:
    state = _state_for(payload.session_id)
    sentence: List[str] = state["sentence"]  # type: ignore[assignment]
    text = " ".join(sentence).strip()
    state["grammar_result"] = text
    return {"grammar_result": text}


@app.post("/reset")
def reset(payload: SessionPayload) -> Dict[str, str]:
    session_states[payload.session_id] = _new_state()
    return {"status": "reset"}
