#!/usr/bin/env python3
"""
Collect a gesture dataset with the webcam.

Keys:
    1..8   select a gesture (in the order of config.GESTURE_DEGREES)
    SPACE  start / stop recording
    d      delete all samples of the selected gesture
    q      quit and save

Tips for good data:
    - About 300 samples per gesture (a few seconds of recording) is enough.
    - While recording, move the hand slightly closer/farther and tilt it a bit;
      also vary the lighting and background.
    - The left hand is accepted too (it is mirrored during normalization).
"""
import argparse
import csv
from collections import Counter

import cv2
import mediapipe as mp
import numpy as np

import config
from hand_features import landmarks_to_array, normalize_landmarks

SAMPLE_EVERY = 2   # take one sample every N frames (reduces near-duplicate samples)


def load_existing():
    rows = []
    if config.DATASET_PATH.exists():
        with open(config.DATASET_PATH, newline="") as f:
            for r in csv.reader(f):
                if r:
                    rows.append((r[0], r[1:]))
    return rows


def save_all(rows):
    config.DATA_DIR.mkdir(exist_ok=True)
    with open(config.DATASET_PATH, "w", newline="") as f:
        w = csv.writer(f)
        for label, vals in rows:
            w.writerow([label, *vals])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--target", type=int, default=300, help="target number of samples per gesture")
    args = ap.parse_args()

    gestures = list(config.GESTURE_DEGREES)
    rows = load_existing()
    counts = Counter(r[0] for r in rows)
    current, recording, frame_i = 0, False, 0

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera; try a different --camera index.")
    hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7,
                                     min_tracking_confidence=0.6)
    drawer = mp.solutions.drawing_utils

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detected = bool(res.multi_hand_landmarks)

        if detected:
            lm = res.multi_hand_landmarks[0]
            drawer.draw_landmarks(frame, lm, mp.solutions.hands.HAND_CONNECTIONS)
            label = res.multi_handedness[0].classification[0].label
            if config.SWAP_HANDEDNESS:
                label = "Left" if label == "Right" else "Right"
            if recording and frame_i % SAMPLE_EVERY == 0:
                norm = normalize_landmarks(landmarks_to_array(lm.landmark),
                                           mirror=(label == "Left"))
                rows.append((gestures[current], [f"{v:.6f}" for v in norm.ravel()]))
                counts[gestures[current]] += 1
        frame_i += 1

        # --- HUD ---
        h, w = frame.shape[:2]
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (300, 30 + 28 * len(gestures)), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
        for i, g in enumerate(gestures):
            n = counts[g]
            color = (0, 255, 0) if n >= args.target else (255, 255, 255)
            mark = ">" if i == current else " "
            cv2.putText(frame, f"{mark}{i + 1}. {g:<10} {n:>4}/{args.target}", (10, 28 + 28 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2 if i == current else 1)
        status = "REC" if recording else "paused"
        col = (0, 0, 255) if recording else (200, 200, 200)
        cv2.putText(frame, f"[{status}] {gestures[current]}   {'hand OK' if detected else 'NO HAND'}",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
        cv2.imshow("Collect gestures", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            recording = not recording
        elif key == ord("d"):
            rows = [r for r in rows if r[0] != gestures[current]]
            counts[gestures[current]] = 0
        elif ord("1") <= key <= ord("0") + len(gestures):
            current, recording = key - ord("1"), False

    cap.release()
    cv2.destroyAllWindows()
    save_all(rows)
    print(f"Saved: {config.DATASET_PATH}  ({len(rows)} samples)")
    for g in gestures:
        print(f"  {g:<10} {counts[g]}")


if __name__ == "__main__":
    main()
