"""
Music engine: chord theory + polyphonic synthesizer.
This file depends on neither the camera nor a sound card (rendering is pure
numpy), so it can be tested offline.
"""
import threading
from dataclasses import dataclass

import numpy as np

import config

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJOR = [0, 2, 4, 5, 7, 9, 11]
MINOR = [0, 2, 3, 5, 7, 8, 10]   # natural minor
SCALES = {"major": MAJOR, "minor": MINOR}
CIRCLE_OF_FIFTHS = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]

# Harmonic amplitudes for each timbre
TIMBRES = {
    "pad":    [1.0, 0.35, 0.15, 0.05],
    "organ":  [1.0, 0.8, 0.6, 0.5, 0.3, 0.2],
    "bright": [1.0, 0.6, 0.45, 0.3, 0.2, 0.1, 0.05],
    "soft":   [1.0, 0.1],
}


def midi_name(n):
    return f"{NOTE_NAMES[n % 12]}{n // 12 - 1}"


def midi_to_freq(n):
    return 440.0 * 2 ** ((n - 69) / 12)


@dataclass
class Chord:
    name: str
    notes: list      # MIDI notes; the first one is the bass note


def _suffix(iv, quality):
    if quality == "sus2":
        return "sus2"
    if quality == "sus4":
        return "sus4"
    third, fifth = iv[1], iv[2]
    if quality == "triad":
        return {(4, 7): "", (3, 7): "m", (3, 6): "dim", (4, 8): "aug"}.get((third, fifth), "")
    seventh = iv[3]
    table = {
        (4, 7, 11): "maj7", (4, 7, 10): "7", (3, 7, 10): "m7",
        (3, 6, 10): "m7b5", (3, 7, 11): "m(maj7)", (3, 6, 9): "dim7",
    }
    return table.get((third, fifth, seventh), "7")


def build_chord(degree, tonic=0, mode="major", quality="triad", octave=0):
    """
    Build a diatonic chord from a scale degree.
    quality: triad | 7th (diatonic seventh) | dom7 (dominant seventh) | sus2 | sus4
    """
    scale = SCALES[mode]

    def step(k):  # distance in semitones, k scale steps above the root
        idx = degree + k
        return scale[idx % 7] + 12 * (idx // 7) - scale[degree]

    third, fifth, seventh = step(2), step(4), step(6)
    if quality == "sus2":
        iv = [0, 2, 7]
    elif quality == "sus4":
        iv = [0, 5, 7]
    elif quality == "7th":
        iv = [0, third, fifth, seventh]
    elif quality == "dom7":
        iv = [0, 4, 7, 10]
    else:
        iv = [0, third, fifth]

    root_pc = (tonic + scale[degree]) % 12
    root_midi = 48 + root_pc + 12 * octave
    notes = [root_midi - 12] + [root_midi + i for i in iv]   # bass + chord
    return Chord(NOTE_NAMES[root_pc] + _suffix(iv, quality), notes)


class ChordSynth:
    """
    Polyphonic synth: each note uses two slightly detuned oscillators plus a
    smooth envelope (no clicks).
    Arpeggio mode: the chord notes are played up and down, one per eighth note.
    """

    def __init__(self, sample_rate=config.SAMPLE_RATE):
        self.sr = sample_rate
        self._lock = threading.Lock()
        self._chord = []
        self._voices = {}              # midi -> dict(amp, ph1, ph2)
        self.timbre = "pad"
        self.arp = False
        self.bpm = config.DEFAULT_BPM
        self._vol_target = 0.8
        self._vol_now = 0.8
        self._pos = 0

    # --- control from the main thread ---------------------------------------
    def set_chord(self, notes):
        with self._lock:
            self._chord = list(notes)

    def set_volume(self, v):
        self._vol_target = float(np.clip(v, 0.0, 1.0))

    def cycle_timbre(self):
        names = list(TIMBRES)
        self.timbre = names[(names.index(self.timbre) + 1) % len(names)]

    # --- audio thread -------------------------------------------------------
    def render(self, frames, pos=None):
        """Render one block. `pos` = absolute sample position of the block start
        (shared clock from the audio engine); if None an internal clock is used."""
        with self._lock:
            chord = list(self._chord)
        if pos is None:
            pos = self._pos
            self._pos += frames
        timbre = TIMBRES[self.timbre]

        if self.arp and chord:
            seq = sorted(chord)
            seq = seq + seq[-2:0:-1]                     # up and down
            step_len = max(1, int(self.sr * 60 / self.bpm / 2))
            active = {seq[(pos // step_len) % len(seq)]}
            attack, release = 0.004, 0.14
        else:
            active = set(chord)
            attack, release = 0.03, 0.30

        t = np.arange(1, frames + 1)
        harm = np.arange(1, len(timbre) + 1)[:, None]
        amps = np.array(timbre)[:, None]
        norm = amps.sum()
        detune = 2 ** (3 / 1200)
        out = np.zeros(frames)

        for n in list(active | set(self._voices)):
            v = self._voices.setdefault(n, {"amp": 0.0, "ph1": 0.0, "ph2": 0.0})
            a0 = v["amp"]
            if n in active:
                a1 = min(1.0, a0 + frames / (self.sr * attack))
            else:
                a1 = max(0.0, a0 - frames / (self.sr * release))
            f = midi_to_freq(n)
            p1 = v["ph1"] + 2 * np.pi * f * t / self.sr
            p2 = v["ph2"] + 2 * np.pi * f * detune * t / self.sr
            w1 = (amps * np.sin(harm * p1[None, :])).sum(0)
            w2 = (amps * np.sin(harm * p2[None, :])).sum(0)
            out += np.linspace(a0, a1, frames) * (w1 + w2) / (2 * norm)
            v["ph1"], v["ph2"], v["amp"] = p1[-1] % (2 * np.pi), p2[-1] % (2 * np.pi), a1
            if a1 == 0.0 and n not in active:
                del self._voices[n]

        vol = np.linspace(self._vol_now, self._vol_target, frames)
        self._vol_now = self._vol_target
        return np.tanh(out * 0.30 * vol).astype(np.float32)

    def callback(self, outdata, frames, time_info, status):
        outdata[:, 0] = self.render(frames)
