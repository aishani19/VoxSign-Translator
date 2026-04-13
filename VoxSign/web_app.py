import os
import string
from typing import Dict, List, Optional, Tuple

import cv2
import gradio as gr
import language_tool_python
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

from my_functions import draw_landmarks, image_process, keypoint_extraction


PATH = os.path.join("data")
actions = np.array(sorted(os.listdir(PATH)))
model = load_model("my_model.keras")
grammar_tool = None
grammar_error = ""
holistic = mp.solutions.holistic.Holistic(
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
)


def _new_state() -> Dict[str, object]:
    return {
        "sentence": [],
        "keypoints": [],
        "last_prediction": None,
        "grammar_result": "",
    }


def _merge_letters(sentence: List[str]) -> None:
    if len(sentence) < 2:
        return

    last = sentence[-1]
    prev = sentence[-2]
    alpha = set(string.ascii_lowercase + string.ascii_uppercase)
    known_actions = set(actions.tolist()) | {x.capitalize() for x in actions.tolist()}

    if last in alpha and (prev in alpha or prev not in known_actions):
        sentence[-1] = (prev + last).capitalize()
        sentence.pop(-2)


def process_stream(
    frame: Optional[np.ndarray], state: Optional[Dict[str, object]]
) -> Tuple[Optional[np.ndarray], str, str, str, Dict[str, object]]:
    if state is None:
        state = _new_state()

    if frame is None:
        return None, "", "", "No frame received. Allow webcam access or use Upload Image.", state

    sentence: List[str] = state["sentence"]  # type: ignore[assignment]
    keypoints: List[np.ndarray] = state["keypoints"]  # type: ignore[assignment]
    last_prediction: Optional[str] = state["last_prediction"]  # type: ignore[assignment]
    grammar_result: str = state["grammar_result"]  # type: ignore[assignment]

    try:
        bgr_frame = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)
        results = image_process(bgr_frame, holistic)
        draw_landmarks(bgr_frame, results)
        keypoints.append(keypoint_extraction(results))
    except Exception as exc:
        return None, " ".join(sentence), grammar_result, f"Frame processing error: {exc}", state

    if len(keypoints) == 10:
        model_input = np.array(keypoints)[np.newaxis, :, :]
        prediction = model.predict(model_input, verbose=0)[0]
        keypoints.clear()
        if np.max(prediction) > 0.9:
            predicted_action = actions[int(np.argmax(prediction))]
            if predicted_action != last_prediction:
                sentence.append(predicted_action)
                last_prediction = predicted_action

    if len(sentence) > 7:
        sentence[:] = sentence[-7:]
    if sentence:
        sentence[0] = sentence[0].capitalize()
    _merge_letters(sentence)

    display_text = grammar_result if grammar_result else " ".join(sentence)
    cv2.putText(
        bgr_frame,
        display_text,
        (20, 460),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    state["sentence"] = sentence
    state["keypoints"] = keypoints
    state["last_prediction"] = last_prediction
    state["grammar_result"] = grammar_result

    out_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    return out_frame, " ".join(sentence), grammar_result, "Frame processed.", state


def apply_grammar(state: Optional[Dict[str, object]]) -> Tuple[str, Dict[str, object]]:
    global grammar_tool, grammar_error
    if state is None:
        state = _new_state()
    sentence: List[str] = state["sentence"]  # type: ignore[assignment]
    text = " ".join(sentence).strip()
    if not text:
        state["grammar_result"] = ""
        return "", state

    # Keep grammar optional to avoid blocking the app on network-restricted machines.
    if os.environ.get("VOXSIGN_ENABLE_GRAMMAR", "0") != "1":
        state["grammar_result"] = text
        return state["grammar_result"], state

    if grammar_tool is None and not grammar_error:
        try:
            grammar_tool = language_tool_python.LanguageTool("en-UK")
        except Exception as exc:
            grammar_error = f"Grammar service unavailable: {exc}"

    if grammar_tool is None:
        state["grammar_result"] = grammar_error or "Grammar service unavailable."
    else:
        state["grammar_result"] = grammar_tool.correct(text)
    return state["grammar_result"], state


def reset_all() -> Tuple[str, str, Dict[str, object]]:
    state = _new_state()
    return "", "", state


with gr.Blocks(title="VoxSign Translator (Web)") as app:
    gr.Markdown("# VoxSign Translator (Browser)")
    gr.Markdown(
        "Allow camera access, sign in front of webcam, click **Predict This Frame**. "
        "Grammar button currently returns the same text unless `VOXSIGN_ENABLE_GRAMMAR=1` is set."
    )

    state = gr.State(_new_state())

    with gr.Row():
        webcam = gr.Image(
            sources="webcam",
            type="numpy",
            label="Webcam Input",
        )
        preview = gr.Image(type="numpy", label="Prediction Preview")

    upload_image = gr.Image(type="numpy", label="Upload Image (fallback if webcam blocked)")
    sentence_box = gr.Textbox(label="Predicted Sentence", interactive=False)
    grammar_box = gr.Textbox(label="Grammar Corrected Sentence", interactive=False)
    status_box = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        predict_btn = gr.Button("Predict This Frame")
        grammar_btn = gr.Button("Apply Grammar")
        reset_btn = gr.Button("Reset")

    webcam.change(
        fn=process_stream,
        inputs=[webcam, state],
        outputs=[preview, sentence_box, grammar_box, status_box, state],
    )
    webcam.stream(
        fn=process_stream,
        inputs=[webcam, state],
        outputs=[preview, sentence_box, grammar_box, status_box, state],
        stream_every=0.2,
    )
    predict_btn.click(
        fn=process_stream,
        inputs=[webcam, state],
        outputs=[preview, sentence_box, grammar_box, status_box, state],
    )
    upload_image.change(
        fn=process_stream,
        inputs=[upload_image, state],
        outputs=[preview, sentence_box, grammar_box, status_box, state],
    )
    grammar_btn.click(fn=apply_grammar, inputs=[state], outputs=[grammar_box, state])
    reset_btn.click(
        fn=reset_all,
        inputs=None,
        outputs=[sentence_box, grammar_box, state],
    ).then(
        fn=lambda: "Reset done.",
        inputs=None,
        outputs=[status_box],
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    app.launch(server_name="0.0.0.0", server_port=port, show_error=True)
