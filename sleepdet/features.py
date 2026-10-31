"""Per-frame signal extraction: face detection, identity, eye openness (EAR), head pitch, motion."""
import glob
import os

import cv2
import mediapipe as mp
import numpy as np

from .models import ensure_models

# MediaPipe FaceMesh landmark indices (p1..p6 for the eye aspect ratio)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
# Points for head-pose estimation: nose tip, chin, left eye corner, right eye corner, mouth corners
POSE_IDX = [1, 152, 263, 33, 291, 61]
POSE_3D = np.array([
    [0.0, 0.0, 0.0],
    [0.0, -63.6, -12.5],
    [-43.3, 32.7, -26.0],
    [43.3, 32.7, -26.0],
    [-28.9, -28.9, -24.1],
    [28.9, -28.9, -24.1],
], dtype=np.float64)

SFACE_MATCH = 0.363  # cosine-similarity threshold recommended by OpenCV for SFace


def eye_aspect_ratio(pts):
    p1, p2, p3, p4, p5, p6 = pts
    v = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    h = 2.0 * np.linalg.norm(p1 - p4)
    return float(v / h) if h > 1e-6 else 0.0


def head_pitch(lm2d, w, h):
    """Pitch in degrees (positive = head tilted down) via solvePnP on 6 landmarks."""
    img_pts = np.array([lm2d[i] for i in POSE_IDX], dtype=np.float64)
    f = float(w)
    cam = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(POSE_3D, img_pts, cam, np.zeros(4), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return float("nan")
    R, _ = cv2.Rodrigues(rvec)
    pitch = np.degrees(np.arctan2(-R[2, 1], R[2, 2]))
    # normalise to [-90, 90]
    if pitch > 90:
        pitch -= 180
    elif pitch < -90:
        pitch += 180
    return float(-pitch)


class Gallery:
    """Reference faces: gallery/<Name>/*.jpg (or gallery/<Name>.jpg). Names are used as person labels."""

    def __init__(self, folder, detector, recognizer):
        self.names, self.embs = [], []
        if not folder or not os.path.isdir(folder):
            return
        entries = []
        for p in sorted(glob.glob(os.path.join(folder, "*"))):
            if os.path.isdir(p):
                for img in sorted(glob.glob(os.path.join(p, "*"))):
                    entries.append((os.path.basename(p), img))
            elif p.lower().endswith((".jpg", ".jpeg", ".png")):
                entries.append((os.path.splitext(os.path.basename(p))[0], p))
        for name, path in entries:
            img = cv2.imread(path)
            if img is None:
                continue
            h, w = img.shape[:2]
            detector.setInputSize((w, h))
            _, faces = detector.detect(img)
            if faces is None or len(faces) == 0:
                print(f"[sleepdet] warning: no face in gallery image {path}")
                continue
            face = max(faces, key=lambda f: f[2] * f[3])
            emb = recognizer.feature(recognizer.alignCrop(img, face)).flatten()
            self.names.append(name)
            self.embs.append(emb / np.linalg.norm(emb))
        if self.names:
            print(f"[sleepdet] gallery: {sorted(set(self.names))}")

    def match(self, emb):
        if not self.embs:
            return None, 0.0
        e = emb / np.linalg.norm(emb)
        sims = np.array(self.embs) @ e
        i = int(np.argmax(sims))
        return (self.names[i], float(sims[i])) if sims[i] >= SFACE_MATCH else (None, float(sims[i]))


class Tracker:
    """Keeps identities stable across frames: gallery name when known, otherwise IoU-matched 'Person N'."""

    def __init__(self, max_missing=30):
        self.tracks = {}  # id -> dict(box, missing, votes, last_emb)
        self.next_id = 1
        self.max_missing = max_missing

    @staticmethod
    def iou(a, b):
        ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
        iw = max(0, min(ax2, bx2) - max(a[0], b[0]))
        ih = max(0, min(ay2, by2) - max(a[1], b[1]))
        inter = iw * ih
        u = a[2] * a[3] + b[2] * b[3] - inter
        return inter / u if u > 0 else 0.0

    def update(self, dets):
        """dets: list of dict(box, name, emb). Returns list of track labels aligned with dets."""
        labels = [None] * len(dets)
        used = set()
        # 1) match by IoU to existing tracks
        pairs = []
        for di, d in enumerate(dets):
            for tid, t in self.tracks.items():
                pairs.append((self.iou(d["box"], t["box"]), di, tid))
        for score, di, tid in sorted(pairs, reverse=True):
            if score < 0.2 or labels[di] is not None or tid in used:
                continue
            labels[di] = tid
            used.add(tid)
        # 2) re-identify unmatched detections by embedding similarity to lost tracks
        for di, d in enumerate(dets):
            if labels[di] is not None:
                continue
            best, best_sim = None, SFACE_MATCH
            for tid, t in self.tracks.items():
                if tid in used or t["emb"] is None:
                    continue
                sim = float(np.dot(d["emb"], t["emb"]) / (np.linalg.norm(d["emb"]) * np.linalg.norm(t["emb"])))
                if sim > best_sim:
                    best, best_sim = tid, sim
            if best is None:
                best = self.next_id
                self.next_id += 1
                self.tracks[best] = {"box": d["box"], "missing": 0, "votes": {}, "emb": None}
            labels[di] = best
            used.add(best)
        # 3) update state
        for di, tid in enumerate(labels):
            t = self.tracks[tid]
            t["box"], t["missing"] = dets[di]["box"], 0
            t["emb"] = dets[di]["emb"] if t["emb"] is None else 0.9 * t["emb"] + 0.1 * dets[di]["emb"]
            if dets[di]["name"]:
                t["votes"][dets[di]["name"]] = t["votes"].get(dets[di]["name"], 0) + 1
        for tid in list(self.tracks):
            if tid not in used:
                self.tracks[tid]["missing"] += 1
        return labels

    def name_of(self, tid):
        v = self.tracks[tid]["votes"]
        return max(v, key=v.get) if v else f"Person {tid}"

    def lost_tracks(self):
        return [tid for tid, t in self.tracks.items() if 0 < t["missing"] <= self.max_missing]


def extract(video_path, gallery_dir=None, sample_fps=10.0, annotate_path=None, max_faces=4):
    """Returns (rows, fps_used, duration). One row per sampled frame per visible/recently-seen person."""
    paths = ensure_models()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(src_fps / sample_fps)))

    detector = cv2.FaceDetectorYN.create(paths["yunet.onnx"], "", (320, 320), 0.7, 0.3, 5000)
    recognizer = cv2.FaceRecognizerSF.create(paths["sface.onnx"], "")
    gallery = Gallery(gallery_dir, detector, recognizer)
    mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                                           min_detection_confidence=0.4)
    tracker = Tracker(max_missing=int(3 * sample_fps))
    writer = None
    prev_gray = None
    rows = []
    idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % step:
            continue
        t = idx / src_fps
        h, w = frame.shape[:2]
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        diff = cv2.absdiff(gray, prev_gray) if prev_gray is not None else None
        prev_gray = gray

        detector.setInputSize((w, h))
        _, faces = detector.detect(frame)
        faces = [] if faces is None else sorted(faces, key=lambda f: -f[2] * f[3])[:max_faces]
        dets = []
        for f in faces:
            box = [float(v) for v in f[:4]]
            emb = recognizer.feature(recognizer.alignCrop(frame, f)).flatten()
            name, _ = gallery.match(emb)
            dets.append({"box": box, "name": name, "emb": emb, "face": f})
        labels = tracker.update(dets)

        for d, tid in zip(dets, labels):
            x, y, bw, bh = d["box"]
            m = 0.35
            x0, y0 = int(max(0, x - m * bw)), int(max(0, y - m * bh))
            x1, y1 = int(min(w, x + (1 + m) * bw)), int(min(h, y + (1 + m) * bh))
            crop = frame[y0:y1, x0:x1]
            ear = pitch = float("nan")
            if crop.size:
                res = mesh.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                if res.multi_face_landmarks:
                    ch, cw = crop.shape[:2]
                    lm = np.array([[p.x * cw, p.y * ch] for p in res.multi_face_landmarks[0].landmark])
                    ear = 0.5 * (eye_aspect_ratio(lm[LEFT_EYE]) + eye_aspect_ratio(lm[RIGHT_EYE]))
                    pitch = head_pitch(lm, cw, ch)
            rows.append(dict(t=t, track=tid, face=1, ear=ear, pitch=pitch,
                             motion=_motion(diff, d["box"], w, h), box=d["box"]))
        # persons temporarily without a detectable face (head dropped, turned away, covered)
        for tid in tracker.lost_tracks():
            box = tracker.tracks[tid]["box"]
            rows.append(dict(t=t, track=tid, face=0, ear=float("nan"), pitch=float("nan"),
                             motion=_motion(diff, box, w, h), box=box))

        if annotate_path:
            if writer is None:
                writer = cv2.VideoWriter(annotate_path, cv2.VideoWriter_fourcc(*"mp4v"), src_fps / step, (w, h))
            frame_rows = [r for r in rows if r["t"] == t]
            for r in frame_rows:
                x, y, bw, bh = [int(v) for v in r["box"]]
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0) if r["face"] else (0, 165, 255), 2)
                txt = f"{tracker.name_of(r['track'])} EAR={r['ear']:.2f}" if r["face"] else f"{tracker.name_of(r['track'])} (no face)"
                cv2.putText(frame, txt, (x, max(15, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            writer.write(frame)
    cap.release()
    if writer:
        writer.release()
    names = {tid: tracker.name_of(tid) for tid in tracker.tracks}
    for r in rows:
        r["person"] = names[r["track"]]
        r.pop("box", None)
    duration = (n_frames / src_fps) if n_frames else (rows[-1]["t"] if rows else 0.0)
    return rows, src_fps / step, duration


def _motion(diff, box, w, h):
    """Mean absolute frame difference over the person's head + upper-body region (0-255 scale)."""
    if diff is None:
        return float("nan")
    x, y, bw, bh = box
    x0, x1 = int(max(0, x - bw)), int(min(w, x + 2 * bw))
    y0, y1 = int(max(0, y - 0.3 * bh)), int(min(h, y + 3 * bh))
    roi = diff[y0:y1, x0:x1]
    return float(roi.mean()) if roi.size else float("nan")
