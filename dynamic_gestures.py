"""
Dynamic (motion-based) gesture detection.

SwipeDetector watches the wrist trajectory of one hand over a short time window
and reports a horizontal swipe ("left" / "right" on the mirrored screen, which
matches the user's own left/right).  It is deliberately simple and explainable:

    swipe  <=>  |dx| >= min_dx  and  |dy| <= max_dy_ratio * |dx|   inside `window_s`

with a cooldown so one physical swipe triggers exactly one event.
"""
from collections import deque


class SwipeDetector:
    def __init__(self, window_s=0.35, min_dx=0.22, max_dy_ratio=0.6, cooldown_s=0.7):
        self.window_s = window_s
        self.min_dx = min_dx
        self.max_dy_ratio = max_dy_ratio
        self.cooldown_s = cooldown_s
        self._hist = deque()
        self._blocked_until = -1e9

    def reset(self):
        """Call when the hand is lost."""
        self._hist.clear()

    def update(self, t, x, y):
        """Feed one wrist sample (t in seconds, x/y in 0..1 image coordinates).
        Returns "left", "right" or None."""
        self._hist.append((t, x, y))
        while self._hist and t - self._hist[0][0] > self.window_s:
            self._hist.popleft()
        if t < self._blocked_until or len(self._hist) < 3:
            return None
        dx = self._hist[-1][1] - self._hist[0][1]
        dy = self._hist[-1][2] - self._hist[0][2]
        if abs(dx) >= self.min_dx and abs(dy) <= self.max_dy_ratio * abs(dx):
            self._hist.clear()
            self._blocked_until = t + self.cooldown_s
            return "right" if dx > 0 else "left"
        return None
