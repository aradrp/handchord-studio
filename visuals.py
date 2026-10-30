"""On-screen widgets drawn with OpenCV: piano keyboard, progression bar, beat dots."""
import cv2

_WHITE_PCS = (0, 2, 4, 5, 7, 9, 11)
_HAS_BLACK_AFTER = (0, 2, 5, 7, 9)


def draw_piano(frame, active, x, y, w, h, lo=24, hi=83):
    """Piano keyboard from MIDI note `lo` to `hi`; notes in `active` are highlighted."""
    active = set(active)
    whites = [n for n in range(lo, hi + 1) if n % 12 in _WHITE_PCS]
    kw = w / len(whites)
    for i, n in enumerate(whites):
        x0, x1 = int(x + i * kw), int(x + (i + 1) * kw)
        color = (110, 230, 120) if n in active else (238, 238, 238)
        cv2.rectangle(frame, (x0, y), (x1, y + h), color, -1)
        cv2.rectangle(frame, (x0, y), (x1, y + h), (60, 60, 60), 1)
    for i, n in enumerate(whites):
        if n % 12 in _HAS_BLACK_AFTER and n + 1 <= hi:
            x0 = int(x + (i + 1) * kw - kw * 0.30)
            x1 = int(x + (i + 1) * kw + kw * 0.30)
            color = (60, 170, 60) if (n + 1) in active else (28, 28, 28)
            cv2.rectangle(frame, (x0, y), (x1, y + int(h * 0.62)), color, -1)


def draw_progression_bar(frame, names, current, x, y, box_w=74, box_h=34, title=""):
    """Row of chord boxes; the current one is highlighted."""
    if title:
        cv2.putText(frame, title, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    for i, name in enumerate(names):
        x0 = x + i * (box_w + 4)
        on = i == current
        cv2.rectangle(frame, (x0, y), (x0 + box_w, y + box_h), (0, 200, 110) if on else (70, 70, 70), -1 if on else 1)
        cv2.putText(frame, name, (x0 + 8, y + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0) if on else (230, 230, 230), 2 if on else 1)


def draw_beat(frame, beat, x, y, r=9):
    """Four dots for the beats of the bar; the current beat is filled."""
    cur = int(beat) % 4
    for i in range(4):
        c = (x + i * (2 * r + 8), y)
        cv2.circle(frame, c, r, (0, 220, 255) if i == cur else (110, 110, 110), -1 if i == cur else 2)
