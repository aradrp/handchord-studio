"""
Backing band: synthesized drums + bass line that follow the tempo and the
chord you are playing. Pure numpy; sample-accurate scheduling on a 16-step grid.
"""
import math
import threading

import numpy as np

import config

STEPS = 16   # 16th notes per bar (4/4)

# kick/snare/hat: lists of steps (0..15).  bass: list of (step, semitone offset)
GROOVES = {
    "rock": {
        "kick": [0, 8, 10], "snare": [4, 12], "hat": list(range(0, 16, 2)),
        "bass": [(s, 0) for s in range(0, 16, 2)],
    },
    "disco": {
        "kick": [0, 4, 8, 12], "snare": [4, 12], "hat": [2, 6, 10, 14],
        "bass": [(s, 0 if (s // 2) % 2 == 0 else 12) for s in range(0, 16, 2)],
    },
    "halftime": {
        "kick": [0, 10], "snare": [8], "hat": list(range(0, 16, 2)),
        "bass": [(0, 0), (6, 0), (10, 0), (14, 7)],
    },
}


def _make_drums(sr):
    rng = np.random.default_rng(7)
    t = np.arange(int(0.30 * sr)) / sr
    freq = 48 + 110 * np.exp(-t * 28)                      # pitch sweep
    kick = np.sin(2 * np.pi * np.cumsum(freq) / sr) * np.exp(-t * 9)

    t = np.arange(int(0.22 * sr)) / sr
    noise = np.diff(rng.normal(0, 1, len(t) + 1))          # emphasize highs
    snare = 0.55 * noise * np.exp(-t * 22) + 0.5 * np.sin(2 * np.pi * 185 * t) * np.exp(-t * 30)

    t = np.arange(int(0.06 * sr)) / sr
    hat = np.diff(rng.normal(0, 1, len(t) + 1)) * np.exp(-t * 90)
    return kick * 0.9, snare * 0.6, hat * 0.30


class BackingBand:
    def __init__(self, sr=config.SAMPLE_RATE):
        self.sr = sr
        self.bpm = config.DEFAULT_BPM
        self.drums_on = False
        self.bass_on = False
        self.groove = "rock"
        self.level = 0.5
        self._lock = threading.Lock()
        self._bass_root = None
        self._active = []          # [array, read index] of hits still ringing
        self._cache = {}
        self._kick, self._snare, self._hat = _make_drums(sr)

    # --- control (main thread) -------------------------------------------------
    def set_chord(self, notes):
        with self._lock:
            self._bass_root = (36 + notes[0] % 12) if notes else None   # bass range C2..B2

    def cycle_groove(self):
        names = list(GROOVES)
        self.groove = names[(names.index(self.groove) + 1) % len(names)]

    # --- synthesis -------------------------------------------------------------
    def _bass_note(self, midi, length):
        key = (midi, length)
        if key not in self._cache:
            t = np.arange(length) / self.sr
            f = 440.0 * 2 ** ((midi - 69) / 12)
            w = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(4 * np.pi * f * t) + 0.12 * np.sin(6 * np.pi * f * t)
            env = np.minimum(1.0, t / 0.008) * np.exp(-t * 3.0)
            env *= np.minimum(1.0, (length - np.arange(length)) / (0.015 * self.sr))   # taper end
            self._cache[key] = 0.6 * w * env
        return self._cache[key]

    # --- audio thread ----------------------------------------------------------
    def render(self, frames, pos):
        with self._lock:
            bass_root = self._bass_root
        out = np.zeros(frames)

        # hits that started in earlier blocks
        still = []
        for arr, idx in self._active:
            n = min(len(arr) - idx, frames)
            out[:n] += arr[idx:idx + n]
            if idx + n < len(arr):
                still.append([arr, idx + n])
        self._active = still

        if not (self.drums_on or self.bass_on):
            return out

        groove = GROOVES[self.groove]
        step_len = self.sr * 60.0 / self.bpm / 4.0
        k = math.ceil(pos / step_len)
        while k * step_len < pos + frames:
            off = min(frames - 1, max(0, int(round(k * step_len - pos))))
            step = k % STEPS
            hits = []
            if self.drums_on:
                if step in groove["kick"]:
                    hits.append(self._kick)
                if step in groove["snare"]:
                    hits.append(self._snare)
                if step in groove["hat"]:
                    hits.append(self._hat)
            if self.bass_on and bass_root is not None:
                for s, semis in groove["bass"]:
                    if s == step:
                        hits.append(self._bass_note(bass_root + semis, int(step_len * 2.6)))
            for arr in hits:
                n = min(len(arr), frames - off)
                out[off:off + n] += arr[:n]
                if n < len(arr):
                    self._active.append([arr, n])
            k += 1
        return out * self.level
