"""
Audio effects: Schroeder reverb + resonant low-pass filter.

Both run block by block (state is kept between blocks) and use only numpy/scipy.
The reverb is a classic Schroeder design: 4 parallel feedback comb filters
followed by 2 series all-pass filters.
"""
import numpy as np
from scipy.signal import lfilter

import config


class _Comb:
    """y[n] = x[n] + g * y[n - D]   (valid for block length <= D)."""

    def __init__(self, delay, feedback):
        self.delay = delay
        self.buf = np.zeros(delay)
        self.i = 0
        self.g = feedback

    def process(self, x):
        n, d = len(x), len(self.buf)
        idx = (self.i + np.arange(n)) % d
        y = x + self.g * self.buf[idx]
        self.buf[idx] = y
        self.i = (self.i + n) % d
        return y


class _Allpass:
    """y[n] = -g*x[n] + x[n-D] + g*y[n-D]   (valid for block length <= D)."""

    def __init__(self, delay, g):
        self.delay = delay
        self.xb = np.zeros(delay)
        self.yb = np.zeros(delay)
        self.i = 0
        self.g = g

    def process(self, x):
        n, d = len(x), len(self.xb)
        idx = (self.i + np.arange(n)) % d
        y = -self.g * x + self.xb[idx] + self.g * self.yb[idx]
        self.xb[idx] = x
        self.yb[idx] = y
        self.i = (self.i + n) % d
        return y


class Reverb:
    COMB_DELAYS = (1116, 1188, 1277, 1356)     # samples at 44.1 kHz
    ALLPASS_DELAYS = (556, 441)

    def __init__(self, sr=config.SAMPLE_RATE, feedback=0.82):
        k = sr / 44100.0
        self.combs = [_Comb(max(8, int(d * k)), feedback) for d in self.COMB_DELAYS]
        self.allpass = [_Allpass(max(8, int(d * k)), 0.5) for d in self.ALLPASS_DELAYS]
        self.gain = 1.0 - feedback                       # normalizes sustained tones
        self.chunk = min(c.delay for c in self.combs + self.allpass)

    def process(self, x):
        """Return the wet signal only."""
        out = np.empty_like(x)
        for a in range(0, len(x), self.chunk):         # keep chunk <= shortest delay
            seg = x[a:a + self.chunk]
            wet = sum(c.process(seg) for c in self.combs) * (self.gain / len(self.combs) * 1.6)
            for ap in self.allpass:
                wet = ap.process(wet)
            out[a:a + len(seg)] = wet
        return out


class EffectsChain:
    """Low-pass filter (brightness) -> reverb (wet/dry fades smoothly on toggle)."""

    MIN_CUTOFF, MAX_CUTOFF = 400.0, 18000.0

    def __init__(self, sr=config.SAMPLE_RATE, reverb_mix=0.30, q=0.8):
        self.sr = sr
        self.reverb = Reverb(sr)
        self.reverb_on = False
        self.reverb_mix = reverb_mix
        self._mix_now = 0.0
        self.q = q
        self.cutoff_target = self.MAX_CUTOFF
        self.cutoff_now = self.MAX_CUTOFF
        self._zi = np.zeros(2)

    def set_brightness(self, b):
        """b in [0, 1] (0 = dark, 1 = open). None means fully open."""
        b = 1.0 if b is None else float(np.clip(b, 0.0, 1.0))
        self.cutoff_target = self.MIN_CUTOFF * (self.MAX_CUTOFF / self.MIN_CUTOFF) ** b

    def toggle_reverb(self):
        self.reverb_on = not self.reverb_on

    def _lowpass_coeffs(self, fc):
        w0 = 2 * np.pi * fc / self.sr
        alpha = np.sin(w0) / (2 * self.q)
        cw = np.cos(w0)
        b = np.array([(1 - cw) / 2, 1 - cw, (1 - cw) / 2])
        a = np.array([1 + alpha, -2 * cw, 1 - alpha])
        return b / a[0], a / a[0]

    def process(self, x):
        # smooth the cutoff in the log domain (avoids zipper noise)
        lc, lt = np.log(self.cutoff_now), np.log(self.cutoff_target)
        self.cutoff_now = float(np.exp(lc + 0.25 * (lt - lc)))
        b, a = self._lowpass_coeffs(min(self.cutoff_now, 0.45 * self.sr))
        y, self._zi = lfilter(b, a, x, zi=self._zi)

        wet = self.reverb.process(y)
        target = self.reverb_mix if self.reverb_on else 0.0
        m = np.linspace(self._mix_now, target, len(y))
        self._mix_now = target
        return y * (1 - 0.5 * m) + wet * m
