"""
Feature extraction from hand landmarks (MediaPipe's 21 points).

Data flow:
    raw landmarks (x, y, z in the 0..1 image range)
        -> normalize_landmarks      : independent of position, hand size and
                                      (with mirror) of left/right hand
        -> features_from_normalized : final feature vector for the model (78 values)
"""
from itertools import combinations
import numpy as np

TIPS = [4, 8, 12, 16, 20]
_PAIRS = np.array(list(combinations(range(5), 2)))


def landmarks_to_array(landmarks):
    """MediaPipe landmark list -> array of shape (21, 3)."""
    return np.array([[p.x, p.y, p.z] for p in landmarks], dtype=float)


def normalize_landmarks(pts, mirror=False):
    """
    1) move the wrist to the origin  2) mirror (for the left hand)
    3) divide by the palm size.
    Returns an array of shape (21, 3).
    """
    pts = np.asarray(pts, dtype=float).copy()
    pts -= pts[0]
    if mirror:
        pts[:, 0] *= -1
    scale = np.linalg.norm(pts[9, :2])  # wrist -> base of the middle finger
    if scale < 1e-6:
        scale = 1.0
    return pts / scale


def features_from_normalized(norm):
    """63 coordinates + 10 fingertip-to-fingertip distances
    + 5 fingertip-to-wrist distances = 78 features."""
    tips = norm[TIPS]
    pair_d = np.linalg.norm(tips[_PAIRS[:, 0]] - tips[_PAIRS[:, 1]], axis=1)
    tip_wrist = np.linalg.norm(tips, axis=1)
    return np.concatenate([norm.ravel(), pair_d, tip_wrist])


def extract_features(pts, mirror=False):
    return features_from_normalized(normalize_landmarks(pts, mirror))


def augment(norm, rng):
    """
    Data augmentation: small in-plane rotation, scale change and noise.
    Makes the model robust to hand angle even with few training samples.
    """
    out = norm.copy()
    ang = np.deg2rad(rng.uniform(-25, 25))
    c, s = np.cos(ang), np.sin(ang)
    rot = np.array([[c, -s], [s, c]])
    out[:, :2] = out[:, :2] @ rot.T
    out[:, :2] *= rng.uniform(0.9, 1.1)
    out += rng.normal(0, 0.02, out.shape)
    return out


def count_fingers(pts):
    """
    Rule-based count of extended fingers (0 to 5).
    Compares distances from the wrist, so it is insensitive to hand rotation.
    """
    xy = np.asarray(pts)[:, :2]
    wrist = xy[0]
    d = lambda a, b: float(np.linalg.norm(xy[a] - xy[b]))
    count = 0
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        if np.linalg.norm(xy[tip] - wrist) > np.linalg.norm(xy[pip] - wrist) * 1.1:
            count += 1
    if d(4, 17) > d(3, 17) * 1.05:   # thumb
        count += 1
    return count
