"""
AudioEngine: one shared clock + mixer for synth, backing band and effects.
Also handles WAV recording of the final mix.
"""
import wave

import numpy as np

import config
from backing import BackingBand
from effects import EffectsChain
from music import ChordSynth


class AudioEngine:
    def __init__(self, sr=config.SAMPLE_RATE):
        self.sr = sr
        self.synth = ChordSynth(sr)
        self.backing = BackingBand(sr)
        self.fx = EffectsChain(sr)
        self.bpm = config.DEFAULT_BPM
        self._pos = 0
        self._rec = None
        self.set_bpm(self.bpm)

    # --- control -------------------------------------------------------------
    def set_bpm(self, bpm):
        self.bpm = int(np.clip(bpm, 50, 200))
        self.synth.bpm = self.bpm
        self.backing.bpm = self.bpm

    def set_chord(self, notes):
        self.synth.set_chord(notes)
        self.backing.set_chord(notes)

    def set_volume(self, v):
        self.synth.set_volume(v)

    @property
    def beat(self):
        """Position inside the bar in beats (0.0 .. 4.0)."""
        return (self._pos / (self.sr * 60.0 / self.bpm)) % 4.0

    # --- recording -------------------------------------------------------------
    def start_recording(self):
        self._rec = []

    @property
    def recording(self):
        return self._rec is not None

    def stop_recording(self, path):
        data, self._rec = self._rec, None
        if not data:
            return False
        pcm = (np.clip(np.concatenate(data), -1, 1) * 32767).astype(np.int16)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sr)
            w.writeframes(pcm.tobytes())
        return True

    # --- audio thread ----------------------------------------------------------
    def render(self, frames):
        pos = self._pos
        self._pos += frames
        mix = self.synth.render(frames, pos) + self.backing.render(frames, pos)
        mix = self.fx.process(mix)
        block = np.tanh(mix).astype(np.float32)
        if self._rec is not None:
            self._rec.append(block.copy())
        return block

    def callback(self, outdata, frames, time_info, status):
        outdata[:, 0] = self.render(frames)
