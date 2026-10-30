"""Preset chord progressions for "progression mode" (swipe = next / previous chord)."""
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class Progression:
    name: str
    degrees: Tuple[int, ...]            # scale degrees 0..6 (0 = I)
    quality: Optional[str] = None       # default chord quality, e.g. "7th" or "dom7"


PROGRESSIONS = [
    Progression("Pop  I-V-vi-IV", (0, 4, 5, 3)),
    Progression("50s  I-vi-IV-V", (0, 5, 3, 4)),
    Progression("Sad  vi-IV-I-V", (5, 3, 0, 4)),
    Progression("Canon", (0, 4, 5, 2, 3, 0, 3, 4)),
    Progression("Jazz  ii-V-I", (1, 4, 0, 0), "7th"),
    Progression("12-bar blues", (0, 0, 0, 0, 3, 3, 0, 0, 4, 3, 0, 4), "dom7"),
]


class ProgressionPlayer:
    def __init__(self, progressions=None):
        self.progressions = list(progressions or PROGRESSIONS)
        self.p_idx = 0
        self.pos = 0

    @property
    def progression(self):
        return self.progressions[self.p_idx]

    @property
    def degree(self):
        return self.progression.degrees[self.pos]

    @property
    def quality(self):
        return self.progression.quality

    def step(self, direction):
        """+1 = next chord, -1 = previous chord (wraps around)."""
        self.pos = (self.pos + direction) % len(self.progression.degrees)

    def next_progression(self):
        self.p_idx = (self.p_idx + 1) % len(self.progressions)
        self.pos = 0
