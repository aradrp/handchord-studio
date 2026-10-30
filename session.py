"""
Session logging: remember which chords were played and when, then export a CSV
and a standard MIDI file (no extra library needed - the SMF writer is built in).
"""
import csv
import struct

TPQ = 480            # MIDI ticks per quarter note


def _vlq(n):
    """MIDI variable-length quantity."""
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    return bytes(reversed(out))


class SessionLog:
    def __init__(self):
        self.t0 = None
        self.events = []     # (time_s, chord name or None, [midi notes])

    def log(self, t, chord):
        """chord: music.Chord, or None for silence."""
        if self.t0 is None:
            self.t0 = t
        name = chord.name if chord else None
        notes = list(chord.notes) if chord else []
        if self.events and self.events[-1][1] == name and self.events[-1][2] == notes:
            return
        self.events.append((t - self.t0, name, notes))

    def __len__(self):
        return sum(1 for e in self.events if e[1])

    def export_csv(self, path):
        if not self.events:
            return False
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "chord", "midi_notes"])
            for t, name, notes in self.events:
                w.writerow([f"{t:.3f}", name or "rest", " ".join(map(str, notes))])
        return True

    def to_midi_bytes(self, bpm=120, end_time=None):
        if not self.events:
            return None
        if end_time is None:
            end_time = self.events[-1][0] + 1.0
        to_tick = lambda s: int(round(s * bpm / 60.0 * TPQ))

        msgs = []     # (tick, priority, bytes)  priority 0 = note off, 1 = note on
        for i, (t, name, notes) in enumerate(self.events):
            end = self.events[i + 1][0] if i + 1 < len(self.events) else end_time
            if not notes or end <= t:
                continue
            on, off = to_tick(t), max(to_tick(t) + 1, to_tick(end))
            for n in notes:
                msgs.append((on, 1, bytes([0x90, n & 0x7F, 90])))
                msgs.append((off, 0, bytes([0x80, n & 0x7F, 0])))
        msgs.sort(key=lambda m: (m[0], m[1]))

        tempo = int(60_000_000 / bpm)
        track = b"\x00\xff\x51\x03" + tempo.to_bytes(3, "big")
        last = 0
        for tick, _, data in msgs:
            track += _vlq(tick - last) + data
            last = tick
        track += b"\x00\xff\x2f\x00"
        header = b"MThd" + struct.pack(">IHHH", 6, 0, 1, TPQ)
        return header + b"MTrk" + struct.pack(">I", len(track)) + track

    def export_midi(self, path, bpm=120, end_time=None):
        data = self.to_midi_bytes(bpm, end_time)
        if data is None:
            return False
        with open(path, "wb") as f:
            f.write(data)
        return True
