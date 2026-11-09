#!/usr/bin/env python3
"""
Automated test that needs no camera or sound card: chord theory, synth,
feature extraction, finger counting and the whole training pipeline on
synthetic hands.
Run:  python selftest.py
"""
import io
import time
import types

import numpy as np

from hand_features import count_fingers, extract_features, normalize_landmarks, augment
from music import ChordSynth, build_chord
import train_model


def fake_hand(ext, rng, noise=0.004):
    """Synthetic hand: ext = [thumb, index, middle, ring, pinky] (1 = extended)"""
    pts = np.zeros((21, 3))
    pts[0] = [0.50, 0.80, 0]
    bases = {1: (0.46, 0.60), 2: (0.50, 0.58), 3: (0.54, 0.60), 4: (0.58, 0.64)}
    for f, (bx, by) in bases.items():
        i0 = 1 + 4 * f
        pts[i0] = [bx, by, 0]
        if ext[f]:
            for k in range(1, 4):
                pts[i0 + k] = [bx, by - 0.07 * k, 0]
        else:
            pts[i0 + 1] = [bx, by - 0.04, 0]
            pts[i0 + 2] = [bx, by - 0.01, 0]
            pts[i0 + 3] = [bx, by + 0.02, 0]
    pts[1] = [0.45, 0.75, 0]
    pts[2] = [0.40, 0.70, 0]
    pts[3] = [0.36, 0.65, 0]
    pts[4] = [0.30, 0.60, 0] if ext[0] else [0.47, 0.66, 0]
    # random rotation / scale / translation + noise
    c = pts[0, :2].copy()
    ang = np.deg2rad(rng.uniform(-20, 20))
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    pts[:, :2] = (pts[:, :2] - c) @ R.T * rng.uniform(0.8, 1.2) + c + rng.uniform(-0.1, 0.1, 2)
    return pts + rng.normal(0, noise, pts.shape)


def main():
    # 1) chord theory
    names = [build_chord(d).name for d in range(7)]
    assert names == ["C", "Dm", "Em", "F", "G", "Am", "Bdim"], names
    assert build_chord(4, quality="7th").name == "G7"
    assert build_chord(0, quality="7th").name == "Cmaj7"
    assert build_chord(1, quality="7th").name == "Dm7"
    assert build_chord(6, quality="7th").name == "Bm7b5"
    assert build_chord(0, quality="sus4").name == "Csus4"
    assert build_chord(0, mode="minor").name == "Cm"
    assert build_chord(3, tonic=7).name == "C"            # IV in G major
    assert build_chord(0, quality="dom7").name == "C7" and build_chord(3, quality="dom7").name == "F7"
    print("OK  chord theory")

    # 2) synth
    s = ChordSynth()
    silent = s.render(512)
    assert np.abs(silent).max() == 0
    s.set_chord(build_chord(0).notes)
    sig = np.concatenate([s.render(512) for _ in range(40)])
    assert 0.05 < np.abs(sig).max() <= 1.0
    assert np.abs(np.diff(sig)).max() < 0.5, "sudden jump (click) in the audio"
    s.set_chord([])
    for _ in range(200):
        s.render(512)
    assert not s._voices, "voices must be freed after release"
    s.arp = True
    s.set_chord(build_chord(0).notes)
    assert np.abs(np.concatenate([s.render(512) for _ in range(100)])).max() > 0.05
    print("OK  synth (no clicks, voices freed, arp)")

    # 3) features
    rng = np.random.default_rng(0)
    h = fake_hand([1, 1, 1, 1, 1], rng)
    n = normalize_landmarks(h)
    assert n.shape == (21, 3) and extract_features(h).shape == (78,)
    assert np.allclose(normalize_landmarks(h * 1.0 + 0.0)[0], 0)
    print("OK  features (78-dim)")

    # 4) finger counting
    patterns = {0: [0, 0, 0, 0, 0], 1: [0, 1, 0, 0, 0], 2: [0, 1, 1, 0, 0],
                3: [0, 1, 1, 1, 0], 4: [0, 1, 1, 1, 1], 5: [1, 1, 1, 1, 1]}
    ok = 0
    total = 0
    for want, p in patterns.items():
        for _ in range(30):
            total += 1
            ok += count_fingers(fake_hand(p, rng)) == want
    assert ok / total > 0.95, ok / total
    print(f"OK  finger counting ({ok}/{total})")

    # 5) training pipeline (SVM / RF) on synthetic gestures
    gest = {"fist": [0, 0, 0, 0, 0], "one": [0, 1, 0, 0, 0], "two": [0, 1, 1, 0, 0],
            "rock": [0, 1, 0, 0, 1], "thumbs_up": [1, 0, 0, 0, 0], "open": [1, 1, 1, 1, 1]}
    norms, labels = [], []
    for g, p in gest.items():
        for _ in range(60):
            norms.append(normalize_landmarks(fake_hand(p, rng)))
            labels.append(g)
    labels = np.array(labels)
    model, best, acc, classes, cm = train_model.train(norms, labels, n_aug=2, verbose=False)
    assert acc > 0.9, acc
    x = extract_features(fake_hand(gest["rock"], rng))[None, :]
    assert model.predict(x)[0] == "rock"
    print(f"OK  training pipeline (best={best}, test acc={acc:.3f})")
    # 6) effects: reverb is stable + decays, low-pass removes highs, block-size independent
    from effects import EffectsChain
    fx = EffectsChain()
    fx.reverb_on = True
    imp = np.zeros(44100 * 3)
    imp[0] = 1.0
    outs = np.concatenate([fx.process(imp[i:i + 1000]) for i in range(0, len(imp), 1000)])
    e_early = np.sum(outs[2000:12000] ** 2)
    e_late = np.sum(outs[-20000:] ** 2)
    assert np.isfinite(outs).all() and e_early > 0 and e_late < 0.2 * e_early, (e_early, e_late)
    fx2 = EffectsChain()
    fx2.cutoff_now = fx2.cutoff_target = 800.0
    noise = np.random.default_rng(0).normal(0, 1, 44100)
    y = np.concatenate([fx2.process(noise[i:i + 512]) for i in range(0, 44100 - 512, 512)])
    spec = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1 / 44100)
    assert spec[freqs > 6000].mean() < 0.05 * spec[freqs < 400].mean()
    print("OK  effects (reverb tail decays, low-pass filters, any block size)")

    # 7) backing band: kick lands on the beat, bass follows the chord, no NaN
    from engine import AudioEngine
    eng = AudioEngine()
    eng.set_bpm(120)
    eng.backing.drums_on = True
    eng.backing.groove = "disco"
    audio = np.concatenate([eng.render(441) for _ in range(200)])          # 2 s
    beat = 22050                                                           # 0.5 s per beat at 120 bpm
    for b in range(4):
        w = audio[b * beat:b * beat + 3000]
        assert np.abs(w).max() > 0.2, f"no kick on beat {b}"
    eng2 = AudioEngine()
    eng2.backing.bass_on = True
    eng2.set_chord(build_chord(4).notes)                                   # G -> bass note G2
    bass = np.concatenate([eng2.render(512) for _ in range(120)])
    eng2.synth.set_chord([])
    f0 = np.fft.rfftfreq(len(bass), 1 / 44100)[np.argmax(np.abs(np.fft.rfft(bass)[:2000]))]
    assert abs(f0 - 98.0) < 3.0, f0                                       # G2 = 98 Hz
    assert np.isfinite(audio).all() and np.abs(audio).max() <= 1.0
    print(f"OK  backing band (kick on beats, bass root = {f0:.0f} Hz)")

    # 8) swipe detection
    from dynamic_gestures import SwipeDetector
    def run_path(xs, ys, fps=30):
        d, hit = SwipeDetector(), []
        for i, (x, y) in enumerate(zip(xs, ys)):
            r = d.update(i / fps, x, y)
            if r:
                hit.append(r)
        return hit
    assert run_path(np.linspace(0.3, 0.7, 10), np.full(10, 0.5)) == ["right"]
    assert run_path(np.linspace(0.7, 0.3, 10), np.full(10, 0.5)) == ["left"]
    assert run_path(np.linspace(0.45, 0.55, 60), np.full(60, 0.5)) == []                 # slow drift
    assert run_path(np.full(10, 0.5), np.linspace(0.2, 0.8, 10)) == []                   # vertical
    assert run_path(np.concatenate([np.linspace(0.3, 0.7, 10), np.linspace(0.7, 0.3, 4)]),
                    np.full(14, 0.5)) == ["right"]                                       # cooldown
    print("OK  swipe detector (left/right, ignores drift/vertical, cooldown)")

    # 9) progressions
    from progressions import ProgressionPlayer
    pp = ProgressionPlayer()
    names = [build_chord(d).name for d in pp.progression.degrees]
    assert names == ["C", "G", "Am", "F"], names
    pp.step(-1)
    assert pp.degree == 3
    blues = [p for p in pp.progressions if "blues" in p.name][0]
    assert [build_chord(d, quality=blues.quality).name for d in blues.degrees[:5]] == ["C7"] * 4 + ["F7"]
    pp.next_progression()
    assert pp.pos == 0 and pp.progression.name.startswith("50s")
    print("OK  progressions (pop I-V-vi-IV = C G Am F, wrap-around)")

    # 10) session export: CSV + a MIDI file that a real parser accepts
    from session import SessionLog
    import tempfile, os
    log = SessionLog()
    log.log(10.0, build_chord(0))
    log.log(12.0, build_chord(4, quality="7th"))
    log.log(13.5, None)
    log.log(14.0, build_chord(5))
    data = log.to_midi_bytes(bpm=120, end_time=6.0)   # times are relative to the first event
    assert data[:4] == b"MThd" and b"MTrk" in data
    try:
        import mido                                                       # optional: strict parse
        mid = mido.MidiFile(file=io.BytesIO(data))
        ons = [m for m in mid.tracks[0] if m.type == "note_on"]
        offs = [m for m in mid.tracks[0] if m.type == "note_off"]
        assert len(ons) == len(offs) == 4 + 5 + 4, (len(ons), len(offs))  # C(4) + G7(5) + Am(4)
        assert abs(mid.length - 6.0) < 0.05, mid.length                   # 6 s of music
        strict = "parsed by mido"
    except ImportError:
        strict = "header check only; pip install mido for a strict parse"
    tmp = tempfile.mkdtemp()
    assert log.export_csv(os.path.join(tmp, "s.csv")) and log.export_midi(os.path.join(tmp, "s.mid"))
    assert len(log) == 3
    print(f"OK  session export (.mid {strict}, CSV written)")

    # 11) cross-validation + latency benchmark on synthetic data
    import evaluate_model
    cv = evaluate_model.cross_validate(norms, labels, k=3, n_aug=1)
    for name, r in cv.items():
        assert np.mean(r["acc"]) > 0.85, (name, r["acc"])
    ms = evaluate_model.latency_benchmark(model, norms, n=50)
    print(f"OK  evaluation (3-fold CV {', '.join(f'{n} {np.mean(r['acc']):.2f}' for n, r in cv.items())}; "
          f"classify latency {ms:.2f} ms)")

    # 12) the whole app logic with synthetic hands (no camera): modes, swipes, HUD drawing
    from app import HandChordApp
    args = types.SimpleNamespace(model=None, rule_based=True, midi=False, midi_port=None,
                                 camera=0, no_audio=True)
    app = HandChordApp(args)
    t = 0.0
    def feed(pattern, left=None, n=8, hand=None):
        global_t = feed.t
        for _ in range(n):
            feed.t += 1 / 30
            app._update_state((hand if hand is not None else fake_hand(pattern, rng), None),
                              None if left is None else (fake_hand(left, rng), None), feed.t)
    feed.t = 0.0
    feed([0, 1, 0, 0, 0]); assert app.chord.name == "C"
    feed([0, 1, 1, 0, 0]); assert app.chord.name == "F"
    feed([0, 1, 1, 1, 0], left=[0, 1, 0, 0, 0]); assert app.chord.name == "G7"
    feed([0, 0, 0, 0, 0]); assert app.chord is None
    app.prog_mode = True                                                   # pop: C G Am F
    feed([0, 1, 0, 0, 0]); assert app.chord.name == "C"
    h0 = fake_hand([0, 1, 0, 0, 0], rng)
    def shifted(dx):
        h = h0.copy()
        h[:, 0] += dx
        return h
    for expect in ("G", "Am", "F"):                                        # 3 swipes to the right
        for i in range(10):                                                # fast flick: 0.4 in 10 frames
            feed(None, n=1, hand=shifted(0.04 * i))
        for i in range(60):                                                # slow return: must NOT swipe left
            feed(None, n=1, hand=shifted(0.36 * (1 - i / 59)))
        assert app.chord.name == expect, (app.chord.name, expect)
    frame = np.zeros((540, 960, 3), np.uint8)
    app.engine.backing.drums_on = True
    app._draw(frame)
    assert frame.sum() > 0
    print("OK  full app logic (gestures, left-hand quality, mute, progression + swipes, HUD)")

    # 13) performance: full chain must use far less than the audio block budget
    eng3 = AudioEngine()
    eng3.backing.drums_on = eng3.backing.bass_on = True
    eng3.fx.reverb_on = True
    eng3.set_chord(build_chord(0, quality="7th").notes)
    t0 = time.perf_counter()
    for _ in range(200):
        eng3.render(512)
    ms = (time.perf_counter() - t0) / 200 * 1000
    assert ms < 0.5 * (512 / 44100 * 1000), ms
    print(f"OK  audio performance ({ms:.2f} ms per 11.6 ms block)")

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
