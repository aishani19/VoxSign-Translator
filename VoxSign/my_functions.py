import mediapipe as mp
import cv2
import numpy as np

def draw_landmarks(image, results):
    if hasattr(results, 'multi_hand_landmarks') and results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

def image_process(image, model):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(image_rgb)
    return results

def hands_process(image, model):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(image_rgb)
    return results

def keypoint_extraction(results):
    lh = np.zeros(63)
    rh = np.zeros(63)
    
    if hasattr(results, 'multi_hand_landmarks') and results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            label = results.multi_handedness[i].classification[0].label
            # FLIP X coordination to handle mirrored webcam
            keypoints = np.array([[(1.0 - res.x), res.y, res.z] for res in hand_landmarks.landmark]).flatten()
            
            if label == 'Left':
                lh = keypoints
            else:
                rh = keypoints
                
    # MediaPipe Holistic output.
    if hasattr(results, 'left_hand_landmarks') and results.left_hand_landmarks:
        lh = np.array(
            [[1.0 - res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]
        ).flatten()
    if hasattr(results, 'right_hand_landmarks') and results.right_hand_landmarks:
        rh = np.array(
            [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
        ).flatten()

    return np.concatenate([lh, rh])
