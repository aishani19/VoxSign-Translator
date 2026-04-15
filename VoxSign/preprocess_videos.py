"""
preprocess_videos.py

Extracts hand landmarks from video files in ProcessedData_vivit/ and saves them
as .npy sequences in data/<label>/<sequence>/<frame>.npy, matching the format
expected by model.py for training.

Usage:
    python preprocess_videos.py [--input ProcessedData_vivit] [--output data] [--frames 10]
"""

import argparse
import os
import sys

import cv2
import mediapipe as mp
import numpy as np


def extract_landmarks_from_video(video_path, holistic, frames_per_sequence=10):
    """
    Extract hand landmark sequences from a video file.

    Returns a list of sequences, where each sequence is a list of
    `frames_per_sequence` landmark arrays (each 126-dim).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  WARNING: Cannot open {video_path}, skipping.")
        return []

    all_keypoints = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)

        lh = np.zeros(63)
        rh = np.zeros(63)

        if results.left_hand_landmarks:
            lh = np.array(
                [[1.0 - res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]
            ).flatten()
        if results.right_hand_landmarks:
            rh = np.array(
                [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
            ).flatten()

        keypoints = np.concatenate([lh, rh])
        all_keypoints.append(keypoints)

    cap.release()

    # Split collected keypoints into fixed-length sequences
    sequences = []
    for i in range(0, len(all_keypoints) - frames_per_sequence + 1, frames_per_sequence):
        seq = all_keypoints[i : i + frames_per_sequence]
        sequences.append(seq)

    # If we have leftover frames that form at least half a sequence, pad to include them
    remainder = len(all_keypoints) % frames_per_sequence
    if remainder >= frames_per_sequence // 2 and len(all_keypoints) >= frames_per_sequence:
        last_seq = all_keypoints[-frames_per_sequence:]
        if len(sequences) == 0 or not np.array_equal(
            np.array(last_seq), np.array(sequences[-1])
        ):
            sequences.append(last_seq)

    return sequences


def main():
    parser = argparse.ArgumentParser(description="Extract landmarks from sign language videos")
    parser.add_argument("--input", default="ProcessedData_vivit", help="Input video directory")
    parser.add_argument("--output", default="data", help="Output landmark directory")
    parser.add_argument("--frames", type=int, default=10, help="Frames per sequence")
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    frames = args.frames

    if not os.path.isdir(input_dir):
        print(f"ERROR: Input directory '{input_dir}' not found.")
        print("Download the dataset first:")
        print("  kaggle datasets download kaushikyh/indian-sign-language-words-with-landmarks")
        sys.exit(1)

    labels = sorted(
        [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    )
    print(f"Found {len(labels)} sign labels: {labels[:10]}{'...' if len(labels) > 10 else ''}")

    with mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        total_sequences = 0

        for label in labels:
            label_dir = os.path.join(input_dir, label)
            videos = sorted(
                [f for f in os.listdir(label_dir) if f.lower().endswith((".mov", ".mp4", ".avi"))]
            )

            if not videos:
                print(f"  [{label}] No video files found, skipping.")
                continue

            seq_counter = 0
            for video_file in videos:
                video_path = os.path.join(label_dir, video_file)
                sequences = extract_landmarks_from_video(video_path, holistic, frames)

                for seq in sequences:
                    seq_dir = os.path.join(output_dir, label, str(seq_counter))
                    os.makedirs(seq_dir, exist_ok=True)

                    for frame_idx, keypoints in enumerate(seq):
                        np.save(os.path.join(seq_dir, f"{frame_idx}.npy"), keypoints)

                    seq_counter += 1

            total_sequences += seq_counter
            print(f"  [{label}] Extracted {seq_counter} sequences from {len(videos)} videos")

    print(f"\nDone! Total: {total_sequences} sequences across {len(labels)} labels")
    print(f"Output saved to: {output_dir}/")
    print(f"\nNext step: Train the model with:")
    print(f"  python model.py")


if __name__ == "__main__":
    main()
