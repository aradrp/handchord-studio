# 🎹 HandChord Studio

Play chords with hand gestures, using image processing (MediaPipe) + machine learning (SVM / Random Forest) + a live synthesizer with a backing band.

## Features

- **Right hand:** gesture = chord, hand height = volume, **swipe = next/previous chord** (progression mode)
- **Left hand:** fingers = chord quality (plain / seventh / sus2 / sus4), **hand height = filter brightness**
- **Custom, trainable gestures** (collect your own dataset) - or no training at all with finger counting
- **Backing band:** synthesized drums + bass that follow the tempo and your chords (3 grooves)
- **Effects:** reverb and a resonant low-pass filter
- **Progression mode:** 6 preset progressions (pop, 50s, sad, canon, jazz ii-V-I, 12-bar blues)
- **On-screen piano**, beat indicator and progression bar
- **Session export:** every performance is saved as CSV + standard MIDI file
- Arpeggio, 4 timbres, 12 keys, major/minor, WAV recording, optional live MIDI output
- **Evaluation tools:** k-fold cross-validation, per-class recall, latency benchmark

## Project structure

| File | Purpose |
|---|---|
| `app.py` | Main real-time application |
| `collect_data.py` | Collect a gesture dataset with the webcam |
| `train_model.py` | Train the model (SVM and Random Forest) |
| `evaluate_model.py` | Cross-validation, confusion matrix, latency benchmark |
| `config.py` | Gesture→chord mapping and all settings |
| `hand_features.py` | Landmark normalization, features, augmentation, finger counting |
| `classifiers.py` | ML classifier and rule-based classifier |
| `dynamic_gestures.py` | Swipe detection from the wrist trajectory |
| `progressions.py` | Preset chord progressions |
| `music.py` | Chord theory + polyphonic synthesizer |
| `backing.py` | Drums + bass generator (16-step grid) |
| `effects.py` | Reverb + low-pass filter |
| `engine.py` | Audio engine: shared clock, mixer, WAV recording |
| `session.py` | Performance log → CSV + MIDI file |
| `visuals.py` | Piano keyboard, progression bar, beat dots |
| `midi_out.py` | Optional live MIDI output |
| `demo_offline.py` | Render `demo.wav` (full band) without a camera |
| `selftest.py` | Automated test without a camera |

## Installation

Python 3.9 to 3.12 (required by mediapipe):

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python selftest.py                # every line should print OK
```

> ⚠️ The mediapipe version is intentionally pinned to `0.10.14`; newer releases removed the legacy `mp.solutions` API.

Make sure your terminal is inside the project folder (the one containing `app.py` and `requirements.txt`) before running the commands.

## Running

### Mode 1 - Quick start (no training)
```bash
python app.py --rule-based
```
Gestures = number of extended fingers on the right hand:

| Gesture | Chord (C major) |
|---|---|
| Fist | Silence |
| 1 finger | C (I) |
| 2 fingers | F (IV) |
| 3 fingers | G (V) |
| 4 fingers | Am (vi) |
| Open palm | Dm (ii) |

### Mode 2 - Custom gestures with ML (recommended)
```bash
python collect_data.py      # 1) record data
python train_model.py       # 2) train
python evaluate_model.py    # 3) (optional) cross-validation + latency, for your report
python app.py               # 4) run (the model is loaded automatically)
```
In `collect_data.py`: keys `1..8` select a gesture, `SPACE` starts/stops recording, `d` clears the current gesture, `q` saves and quits. Record about 300 samples per gesture and **move the hand slightly closer/farther, tilt it and shift it while recording** so the model generalizes.

Default gestures: `fist, one, two, three, four, open, thumbs_up, rock` - rename or add your own in `config.py`.

### Left hand
| Left hand | Quality | Example on C |
|---|---|---|
| Fist / no hand | Plain | C |
| 1 finger | Seventh | Cmaj7 (on G it becomes G7) |
| 2 fingers | sus2 | Csus2 |
| 3 fingers | sus4 | Csus4 |

Raise the left hand for a **bright** sound, lower it for a **darker** (filtered) sound.

### Progression mode (`p`)
Press `p`, then:
- any gesture except a fist = play the current chord of the progression (fist = mute)
- **swipe right / left** with the right hand = next / previous chord
- `n` = next preset progression

A swipe is a fast, mostly horizontal wrist movement (about 22% of the frame width in 0.35 s). A 0.7 s cooldown stops the hand's return movement from counting as a second swipe - return your hand a bit slower than the flick.

### Band and effects
`d` drums · `b` bass · `g` next groove (rock / disco / halftime) · `v` reverb · `-` `=` tempo. Try: `d`, `b`, `v`, then `p` and swipe through a 12-bar blues.

### All keys
| Key | Action | Key | Action |
|---|---|---|---|
| `q` | quit | `p` | progression mode |
| `k` | next key (circle of fifths) | `n` | next progression |
| `m` | major / minor | `d` | drums on/off |
| `[` `]` | octave | `b` | bass on/off |
| `a` | arpeggio | `g` | next groove |
| `t` | timbre | `v` | reverb on/off |
| `-` `=` | tempo | `s` | save session (CSV + MIDI) |
| `r` | record WAV | | |

Recordings go to `recordings/`, sessions to `sessions/` (also saved automatically when you quit).

### MIDI (optional)
```bash
pip install mido python-rtmidi
python app.py --list-midi
python app.py --midi --midi-port "IAC"        # or part of any port name
python app.py --midi --no-audio               # MIDI only, no built-in synth
```
macOS: IAC Driver · Windows: loopMIDI · Linux: a virtual port is created automatically. Live MIDI sends the chords only (no arpeggio, drums or bass); the exported session `.mid` contains the chords as well.

Check the sound without a camera: `python demo_offline.py` (renders plain chords, arpeggio and the full band).

## How it works (for your project report)

```
Webcam → mirror flip → MediaPipe Hands (21 landmarks per hand)
   ├─ right hand → normalization → 78 features → SVM/RF → (gesture, confidence) → 7-frame voting
   │        ├→ wrist height → volume
   │        └→ wrist trajectory → SwipeDetector → next/previous chord (progression mode)
   └─ left hand  → finger counting → chord quality
                 → wrist height → filter brightness
gesture/progression + quality + key + mode → build_chord (diatonic theory)
        → AudioEngine: [ChordSynth + BackingBand] → low-pass → reverb → soft clip → speakers / WAV
        → SessionLog → CSV + MIDI      → optional live MIDI
```

**Normalization:** the wrist is moved to the origin, coordinates are divided by the wrist→middle-finger-base length (independent of camera distance and hand size), and the left hand is mirrored.

**Features (78-dimensional):** 63 coordinates + 10 fingertip-to-fingertip distances + 5 fingertip-to-wrist distances.

**Augmentation:** rotation ±25°, scale ±10%, Gaussian noise; applied to the training folds only and *after* the split (prevents data leakage).

**Dynamic gesture (swipe):** `|dx| ≥ 0.22` and `|dy| ≤ 0.6·|dx|` inside a 0.35 s window of wrist positions, plus a cooldown. It is rule-based on purpose (explainable, no extra training data); an LSTM or DTW model is the natural next step.

**Diatonic chords:** built by stacking thirds on a scale degree, so quality (maj/min/dim) and name come out of the scale automatically. `dom7` gives true dominant sevenths for the blues.

**Audio engine:** a single sample clock drives the arpeggiator, drums and bass, so everything stays in time. Drums are synthesized (pitch-swept kick, filtered-noise snare and hat); the bass follows the chord root (C2-B2). The reverb is a Schroeder design (4 comb + 2 all-pass filters); the filter is a biquad low-pass with the cutoff smoothed in the log domain. The whole chain renders a 11.6 ms block in about 1 ms.

**Evaluation (`evaluate_model.py`):** stratified k-fold cross-validation of SVM / Random Forest / k-NN, per-class recall, a pooled confusion matrix (`models/cv_confusion_matrix.png`) and the per-hand classification latency.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Could not open requirements file` | Your terminal is not in the project folder; `cd` into the folder that contains `requirements.txt` |
| Left and right are swapped | In `config.py` set `SWAP_HANDEDNESS = True` |
| Camera does not open | Use `--camera 1` (or 2) |
| Gestures get confused | Record more varied data; replace look-alike gestures (e.g. 3 vs 4 fingers) with distinct ones; check `confusion_matrix.png` |
| Chords change too slowly | Lower `MIN_VOTES` or `VOTE_WINDOW` in `config.py` (more responsive, less stable) |
| Swipes trigger by accident / not at all | Adjust `min_dx`, `window_s`, `cooldown_s` in `dynamic_gestures.py` |
| Audio crackles | Set `BLOCK_SIZE` to 1024; close heavy applications |
| `mp.solutions` error | The mediapipe version must be `0.10.14` |
| sounddevice error on Linux | `sudo apt install libportaudio2` |

## More ideas
- LSTM / DTW for richer dynamic gestures (circle, zig-zag, two-hand pinch)
- Chord changes quantized to the beat or bar when the band is playing
- More grooves, drum fills, strumming/comping patterns
- Per-user calibration screen (reach, hand size, thresholds)
- User study: latency, accuracy under different lighting, several users, learning curve
