import os
import json
import string
import cv2
import gradio as gr
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# 1. LOADING CORE ASSETS
with open('labels.json', 'r') as f:
    actions = np.array(json.load(f))

model = load_model("my_model.keras")

# 2. AI MODELS
hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3,
)

def keypoint_extraction(results):
    lh = np.zeros(63)
    rh = np.zeros(63)
    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            label = results.multi_handedness[i].classification[0].label
            # Mirror fix: 1.0 - x
            kp = np.array([[(1.0 - res.x), res.y, res.z] for res in hand_landmarks.landmark]).flatten()
            if label == 'Left': lh = kp
            else: rh = kp
    return np.concatenate([lh, rh])

def _new_state():
    return {"sentence": [], "keypoints": [], "last_prediction": None}

def process_stream(frame, state):
    if state is None: state = _new_state()
    if frame is None: return None, "", "Waiting for Camera...", state

    sentence = state["sentence"]
    keypoints = state["keypoints"]
    last_pred = state["last_prediction"]

    try:
        # Visibility Boost (CLAHE)
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = cv2.createCLAHE(clipLimit=3.0).apply(l)
        bgr_enhanced = cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2BGR)
        
        # AI Detection
        results = hands.process(cv2.cvtColor(bgr_enhanced, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            for hl in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(bgr_enhanced, hl, mp.solutions.hands.HAND_CONNECTIONS)
        
        # Prediction Logic
        keypoints.append(keypoint_extraction(results))
        status = f"Buffer: {len(keypoints)}/6"
        
        if len(keypoints) == 6:
            pred = model.predict(np.array(keypoints)[np.newaxis, :, :], verbose=0)[0]
            keypoints.clear()
            
            # Bias Greetings
            for g in ['Hi', 'Hello']:
                if g in actions: pred[np.where(actions==g)[0][0]] *= 1.2

            prob = float(np.max(pred))
            word = actions[np.argmax(pred)]
            status = f"Last Sign: {word} ({prob*100:.0f}%)"
            
            if prob > 0.35: # Always predict if reasonably sure
                if word != last_pred:
                    sentence.append(word)
                    state["last_prediction"] = word

        if len(sentence) > 5: sentence.pop(0)
        
        # Final UI data
        state["sentence"] = sentence
        state["keypoints"] = keypoints
        
        res_frame = cv2.cvtColor(bgr_enhanced, cv2.COLOR_BGR2RGB)
        return res_frame, " ".join(sentence), status, state

    except Exception as e:
        return frame, " ".join(sentence), f"Error: {e}", state

# 3. GRADIO UI
with gr.Blocks(title="VoxSign Final") as app:
    gr.Markdown("# 🚀 VoxSign Pro: 76-Sign Ultimate Translator")
    state = gr.State(_new_state())
    
    with gr.Row():
        cam = gr.Image(sources="webcam", type="numpy", label="Webcam")
        out = gr.Image(type="numpy", label="AI Output")
    
    text = gr.Textbox(label="Detected Signs", interactive=False, text_align="center")
    stat = gr.Textbox(label="Status", interactive=False)
    btn_clear = gr.Button("Clear History")

    # The Engine
    cam.stream(fn=process_stream, inputs=[cam, state], outputs=[out, text, stat, state], stream_every=0.1)
    btn_clear.click(fn=lambda: ("", "History Cleared", _new_state()), outputs=[text, stat, state])

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
