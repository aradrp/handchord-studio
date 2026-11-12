#!/usr/bin/env python3
"""
HandChord Studio - play chords with hand gestures.

Right hand : gesture = chord ; hand height = volume ; swipe = next/prev chord (progression mode)
Left hand  : fingers = chord quality (0 plain, 1 seventh, 2 sus2, 3 sus4) ; hand height = filter brightness

Keys:
  q quit            k next key        m major/minor      [ ] octave
  a arpeggio        t timbre          - =  tempo
  d drums           b bass            g groove           v reverb
  p progression mode                  n next progression
  r record WAV      s save session (CSV + MIDI)
"""
import argparse
import datetime
import time
from collections import Counter, deque

import cv2
import mediapipe as mp
import numpy as np

import config
from classifiers import load_classifier
from dynamic_gestures import SwipeDetector
from engine import AudioEngine
from hand_features import count_fingers, landmarks_to_array
from midi_out import MidiOut
from music import CIRCLE_OF_FIFTHS, NOTE_NAMES, build_chord, midi_name
from progressions import ProgressionPlayer
from session import SessionLog
from visuals import draw_beat, draw_piano, draw_progression_bar


class Stabilizer:
    """Majority voting over the last few frames so the output does not jitter."""

    def __init__(self, window, min_votes, initial=None):
        self.buf = deque(maxlen=window)
        self.min_votes = min_votes
        self.value = initial

    def update(self, x):
        self.buf.append(x)
        label, count = Counter(self.buf).most_common(1)[0]
        if count >= self.min_votes:
            self.value = label
        return self.value


class HandChordApp:
    def __init__(self, args):
        self.args = args
        self.classifier = load_classifier(args.model, args.rule_based)
        self.engine = AudioEngine()
        self.synth = self.engine.synth
        self.midi = MidiOut(args.midi_port, enable=args.midi or bool(args.midi_port))
        self.stab_gesture = Stabilizer(config.VOTE_WINDOW, config.MIN_VOTES)
        self.stab_quality = Stabilizer(config.VOTE_WINDOW, config.MIN_VOTES, initial=0)
        self.swipe = SwipeDetector()
        self.prog = ProgressionPlayer()
        self.prog_mode = False
        self.log = SessionLog()

        self.key_idx, self.mode, self.octave = 0, "major", 0
        self.sounding = None          # (degree, quality, tonic, mode, octave) or None
        self.chord = None
        self.history = deque(maxlen=8)
        self.gesture, self.conf, self.quality = None, 0.0, "triad"
        self.volume, self.fps = 0.8, 0.0
        self.rec_path = None
        self._toast, self._toast_until = "", 0.0
        self._last_t = 0.0

    # ------------------------------------------------------------------
    @property
    def tonic(self):
        return CIRCLE_OF_FIFTHS[self.key_idx % 12]

    def toast(self, text, seconds=1.6):
        self._toast, self._toast_until = text, time.perf_counter() + seconds

    def _detect(self, res):
        right = left = None
        if res.multi_hand_landmarks:
            for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                label = hd.classification[0].label
                if config.SWAP_HANDEDNESS:
                    label = "Left" if label == "Right" else "Right"
                pts = landmarks_to_array(lm.landmark)
                if label == "Right":
                    right = (pts, lm)
                else:
                    left = (pts, lm)
        return right, left

    def _update_state(self, right, left, now=None):
        now = time.perf_counter() if now is None else now
        self._last_t = now

        # --- right hand: gesture, volume, swipe ---
        swipe = None
        if right is None:
            raw = None
            self.conf = 0.0
            self.swipe.reset()
        else:
            label, self.conf = self.classifier.predict(right[0])
            raw = label if self.conf >= config.CONF_THRESHOLD else "unknown"
            y = right[0][0, 1]
            self.volume = float(np.clip(1.0 - (y - 0.2) * (0.8 / 0.65), 0.15, 1.0))
            swipe = self.swipe.update(now, right[0][0, 0], right[0][0, 1])
        self.gesture = self.stab_gesture.update(raw)
        self.engine.set_volume(self.volume)

        if swipe and self.prog_mode:
            self.prog.step(+1 if swipe == "right" else -1)
            self.toast(f"swipe {swipe}")

        # --- left hand: quality (fingers) and brightness (height) ---
        n = count_fingers(left[0]) if left is not None else 0
        q = self.stab_quality.update(n)
        left_quality = config.QUALITY_BY_LEFT_FINGERS.get(q, "triad")
        if left_quality != "triad":
            self.quality = left_quality
        elif self.prog_mode and self.prog.quality:
            self.quality = self.prog.quality
        else:
            self.quality = "triad"
        if left is None:
            self.engine.fx.set_brightness(None)
        else:
            self.engine.fx.set_brightness(1.0 - (left[0][0, 1] - 0.2) / 0.65)

        # --- which chord should sound? ---
        if self.gesture == "unknown":                 # low confidence: keep what is playing
            playing = self.sounding is not None
            gesture_degree = None if self.sounding is None else self.sounding[0]
        else:
            gesture_degree = config.GESTURE_DEGREES.get(self.gesture)
            playing = gesture_degree is not None
        if self.prog_mode:
            degree = self.prog.degree if playing else None
        else:
            degree = gesture_degree

        new = None if degree is None else (degree, self.quality, self.tonic, self.mode, self.octave)
        if new != self.sounding:
            self.sounding = new
            if new is None:
                self.chord = None
                self.engine.set_chord([])
                self.midi.all_off()
            else:
                self.chord = build_chord(new[0], new[2], new[3], new[1], new[4])
                self.engine.set_chord(self.chord.notes)
                self.midi.set_notes(self.chord.notes, int(40 + 87 * self.volume))
                if not self.history or self.history[-1] != self.chord.name:
                    self.history.append(self.chord.name)
            self.log.log(now, self.chord)

    # ------------------------------------------------------------------
    def _progression_names(self):
        p = self.prog.progression
        return [build_chord(d, self.tonic, self.mode, p.quality or "triad").name for d in p.degrees]

    def _draw(self, frame):
        h, w = frame.shape[:2]
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (w, 178), (0, 0, 0), -1)
        cv2.rectangle(ov, (0, h - 40), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
        font = cv2.FONT_HERSHEY_SIMPLEX

        name = self.chord.name if self.chord else "rest"
        cv2.putText(frame, name, (20, 85), font, 2.4, (0, 255, 120) if self.chord else (150, 150, 150), 5)
        g = self.gesture if self.gesture else "no hand"
        cv2.putText(frame, f"gesture: {g} ({self.conf:.0%})   quality: {self.quality}",
                    (20, 118), font, 0.65, (255, 255, 255), 2)
        if self.chord:
            notes = " ".join(midi_name(n) for n in self.chord.notes)
            cv2.putText(frame, notes, (20, 142), font, 0.55, (200, 200, 120), 1)

        band = self.engine.backing
        fx = self.engine.fx
        info = [
            f"Key: {NOTE_NAMES[self.tonic]} {self.mode}   Oct: {self.octave:+d}",
            f"Timbre: {self.synth.timbre}   Tempo: {self.engine.bpm}",
            f"Arp: {'ON' if self.synth.arp else 'off'}   Mode: {'PROGRESSION' if self.prog_mode else 'free'}",
            f"Drums: {'ON' if band.drums_on else 'off'}  Bass: {'ON' if band.bass_on else 'off'}  [{band.groove}]",
            f"Reverb: {'ON' if fx.reverb_on else 'off'}   Filter: {fx.cutoff_now / 1000:.1f} kHz",
            f"MIDI: {self.midi.label}",
            f"FPS: {self.fps:.0f}" + ("   [REC]" if self.engine.recording else ""),
        ]
        for i, t in enumerate(info):
            col = (0, 0, 255) if "[REC]" in t else (255, 255, 255)
            cv2.putText(frame, t, (w - 400, 26 + 22 * i), font, 0.5, col, 1)

        # volume bar
        bar_h = int(120 * self.volume)
        cv2.rectangle(frame, (w - 40, 195), (w - 20, 315), (90, 90, 90), 1)
        cv2.rectangle(frame, (w - 40, 315 - bar_h), (w - 20, 315), (0, 220, 120), -1)

        if self.prog_mode:
            names = self._progression_names()
            draw_progression_bar(frame, names, self.prog.pos, 20, 205,
                                 box_w=68 if len(names) > 8 else 74,
                                 title=self.prog.progression.name + "   (swipe to move)")

        # piano + beat + history
        active = self.chord.notes if self.chord else []
        draw_piano(frame, active, 20, h - 112, min(w - 60, 640), 62)
        draw_beat(frame, self.engine.beat, w - 120, h - 80)
        hist = "  ".join(self.history) if self.history else "-"
        cv2.putText(frame, f"history: {hist}", (15, h - 14), font, 0.6, (255, 255, 255), 1)

        if self._toast and time.perf_counter() < self._toast_until:
            size = cv2.getTextSize(self._toast, font, 1.0, 2)[0]
            cv2.putText(frame, self._toast, ((w - size[0]) // 2, h // 2), font, 1.0, (0, 255, 255), 2)

    def _handle_key(self, key):
        eng, band = self.engine, self.engine.backing
        if key == ord("k"):
            self.key_idx += 1
        elif key == ord("m"):
            self.mode = "minor" if self.mode == "major" else "major"
        elif key == ord("a"):
            self.synth.arp = not self.synth.arp
        elif key == ord("t"):
            self.synth.cycle_timbre()
        elif key == ord("["):
            self.octave = max(-1, self.octave - 1)
        elif key == ord("]"):
            self.octave = min(1, self.octave + 1)
        elif key == ord("-"):
            eng.set_bpm(eng.bpm - 5)
        elif key == ord("="):
            eng.set_bpm(eng.bpm + 5)
        elif key == ord("d"):
            band.drums_on = not band.drums_on
        elif key == ord("b"):
            band.bass_on = not band.bass_on
        elif key == ord("g"):
            band.cycle_groove()
            self.toast(f"groove: {band.groove}")
        elif key == ord("v"):
            eng.fx.toggle_reverb()
        elif key == ord("p"):
            self.prog_mode = not self.prog_mode
            self.toast("progression mode ON: swipe to change chord, fist = mute" if self.prog_mode
                       else "free mode")
        elif key == ord("n"):
            self.prog.next_progression()
            self.toast(self.prog.progression.name)
        elif key == ord("s"):
            self._save_session()
        elif key == ord("r"):
            if eng.recording:
                if eng.stop_recording(self.rec_path):
                    print(f"Recording saved: {self.rec_path}")
                    self.toast("recording saved")
            else:
                config.REC_DIR.mkdir(exist_ok=True)
                stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                self.rec_path = config.REC_DIR / f"handchord_{stamp}.wav"
                eng.start_recording()

    def _save_session(self):
        if not len(self.log):
            self.toast("nothing to save yet")
            return
        config.SESSION_DIR.mkdir(exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base = config.SESSION_DIR / f"session_{stamp}"
        end = self._last_t - self.log.t0
        self.log.export_csv(base.with_suffix(".csv"))
        self.log.export_midi(base.with_suffix(".mid"), bpm=self.engine.bpm, end_time=end)
        print(f"Session saved: {base}.csv / .mid  ({len(self.log)} chords)")
        self.toast("session saved (CSV + MIDI)")

    # ------------------------------------------------------------------
    def run(self):
        print("Classifier:", self.classifier.name)
        cap = cv2.VideoCapture(self.args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        if not cap.isOpened():
            raise SystemExit("Could not open the camera; try a different --camera index.")

        stream = None
        if not self.args.no_audio:
            import sounddevice as sd
            stream = sd.OutputStream(samplerate=config.SAMPLE_RATE, channels=1,
                                     blocksize=config.BLOCK_SIZE, dtype="float32",
                                     callback=self.engine.callback)
            stream.start()

        hands = mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.6,
                                         min_tracking_confidence=0.5)
        drawer = mp.solutions.drawing_utils
        t_prev = time.perf_counter()
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)    # mirrored -> Left/Right labels are correct
                res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                right, left = self._detect(res)
                for hand in (right, left):
                    if hand is not None:
                        drawer.draw_landmarks(frame, hand[1], mp.solutions.hands.HAND_CONNECTIONS)

                now = time.perf_counter()
                self._update_state(right, left, now)
                self.fps = 0.9 * self.fps + 0.1 / max(now - t_prev, 1e-6)
                t_prev = now

                self._draw(frame)
                cv2.imshow("HandChord Studio", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                self._handle_key(key)
        finally:
            if self.engine.recording and self.rec_path:
                self.engine.stop_recording(self.rec_path)
                print(f"Recording saved: {self.rec_path}")
            if len(self.log):
                self._save_session()
            if stream:
                stream.stop()
                stream.close()
            self.midi.close()
            cap.release()
            cv2.destroyAllWindows()


def parse_args():
    ap = argparse.ArgumentParser(description="HandChord Studio")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--model", default=None, help="model path (default: models/gesture_model.joblib)")
    ap.add_argument("--rule-based", action="store_true", help="no model; just count fingers")
    ap.add_argument("--midi", action="store_true", help="enable MIDI output")
    ap.add_argument("--midi-port", default=None, help="part of the MIDI port name")
    ap.add_argument("--list-midi", action="store_true", help="list MIDI ports and exit")
    ap.add_argument("--no-audio", action="store_true", help="no built-in sound (MIDI only)")
    return ap.parse_args()


if __name__ == "__main__":
    a = parse_args()
    if a.list_midi:
        print(MidiOut.list_ports() or "No MIDI ports found")
    else:
        HandChordApp(a).run()
