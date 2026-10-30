#!/usr/bin/env python3
"""
No camera and no sound card needed: renders a demo to demo.wav that shows
  1) plain chords with different qualities,
  2) the arpeggiator,
  3) the full band (drums + bass + reverb) over the I-V-vi-IV progression.
Handy for checking sound quality, timbres and grooves.
"""
import wave

import numpy as np

import config
from engine import AudioEngine
from music import build_chord

PLAIN = [(0, "triad"), (4, "triad"), (5, "7th"), (3, "sus2"),
         (0, "7th"), (4, "sus4"), (4, "triad"), (0, "triad")]
POP = [(0, "triad"), (4, "triad"), (5, "triad"), (3, "triad")]


def play(engine, progression, seconds_per_chord, repeats=1):
    blocks, frames = [], config.BLOCK_SIZE
    for _ in range(repeats):
        for degree, quality in progression:
            chord = build_chord(degree, quality=quality)
            print(f"  {chord.name}")
            engine.set_chord(chord.notes)
            for _ in range(int(seconds_per_chord * engine.sr / frames)):
                blocks.append(engine.render(frames))
    engine.set_chord([])
    for _ in range(int(0.8 * engine.sr / frames)):      # let the tail ring out
        blocks.append(engine.render(frames))
    return np.concatenate(blocks)


def main():
    eng = AudioEngine()
    print("Part 1: plain chords")
    a = play(eng, PLAIN, 1.4)

    print("Part 2: arpeggio")
    eng.synth.arp, eng.synth.timbre = True, "bright"
    b = play(eng, PLAIN[:4], 1.4)

    print("Part 3: full band (rock groove, reverb)")
    eng.synth.arp, eng.synth.timbre = False, "pad"
    eng.set_bpm(100)
    eng.backing.drums_on = eng.backing.bass_on = True
    eng.fx.reverb_on = True
    c = play(eng, POP, 60 / 100 * 4, repeats=2)          # one bar per chord

    audio = np.concatenate([a, b, c])
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open("demo.wav", "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(eng.sr)
        w.writeframes(pcm.tobytes())
    print(f"demo.wav created ({len(audio) / eng.sr:.1f} seconds)")


if __name__ == "__main__":
    main()
