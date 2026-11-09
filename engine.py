"""
AudioEngine: one shared clock + mixer for synth, backing band and effects.
Also handles WAV recording of the final mix.
"""
import tempfile
import threading
import wave
from pathlib import Path

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
        # One sixteenth note is sr * 60 units. Each sample advances 4 * BPM units.
        self._step_units = sr * 60
        self._phase = 0
        self._next_step = 0
        self._rec_lock = threading.Lock()
        self._rec = None
        self._pending_wav = None
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
        return self._phase / (4 * self._step_units) % 4.0

    # --- recording -------------------------------------------------------------
    def start_recording(self):
        with self._rec_lock:
            if not self.has_recording:
                self._rec = []

    @property
    def recording(self):
        return self._rec is not None

    @property
    def has_recording(self):
        return self.recording or self._pending_wav is not None

    def stop_recording(self, path):
        # Detach the buffer before doing any work that could stall playback.
        with self._rec_lock:
            if self._rec is not None:
                self._pending_wav = self._rec or None
                self._rec = None
        if self._pending_wav is None:
            return False
        pcm = (np.clip(np.concatenate(self._pending_wav), -1, 1) * 32767).astype(np.int16)
        path = Path(path)
        path.parent.mkdir(exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".wav.tmp", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            with wave.open(str(tmp_path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(self.sr)
                w.writeframes(pcm.tobytes())
            tmp_path.replace(path)
        finally:
            tmp_path.unlink(missing_ok=True)
        self._pending_wav = None
        return True

    # --- audio thread ----------------------------------------------------------
    def render(self, frames):
        pos = self._pos
        self._pos += frames
        bpm = self.bpm
        rate = 4 * bpm
        phase = self._phase
        self._phase += frames * rate
        steps = []
        while self._next_step * self._step_units <= self._phase - rate:
            distance = self._next_step * self._step_units - phase
            # Schedule on the first sample at or after the boundary.
            offset = max(0, (distance + rate - 1) // rate)
            steps.append((offset, self._next_step))
            self._next_step += 1
        step = phase // self._step_units
        mix = self.synth.render(frames, pos, step=step, boundaries=steps)
        mix = mix + self.backing.render(frames, pos, boundaries=steps, bpm=bpm)
        mix = self.fx.process(mix)
        block = np.tanh(mix).astype(np.float32)
        with self._rec_lock:
            if self._rec is not None:
                self._rec.append(block.copy())
        return block

    def callback(self, outdata, frames, time_info, status):
        outdata[:, 0] = self.render(frames)
