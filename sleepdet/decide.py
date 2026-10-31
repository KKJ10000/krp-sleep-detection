"""Turns per-frame signals into a per-second state per person, then into sleep intervals.

Operational definition (see WRITEUP.md):
  A person is ASLEEP from the start of an eyes-closed run once their eyes have been closed
  (PERCLOS >= 0.7 within each second) for at least `onset_s` consecutive seconds AND their
  head/upper body is still (motion below a per-person adaptive threshold).
  They stay asleep through short eye openings (< `wake_s`) and through stretches where the
  face is hidden (head slumped / turned) as long as they remain still; they wake on
  `wake_s` seconds of open eyes or on a burst of motion.
"""
import numpy as np
import pandas as pd

STATES = ("awake", "drowsy", "asleep", "unknown")


def calibrate(df):
    """Per-person eye-closed threshold from that person's own open-eye EAR, and a motion threshold."""
    cal = {}
    for person, g in df.groupby("person"):
        ear = g["ear"].dropna()
        open_ear = float(np.percentile(ear, 90)) if len(ear) >= 10 else 0.28
        thr = float(np.clip(0.65 * open_ear, 0.12, 0.22))
        mot = g["motion"].dropna()
        still = float(np.percentile(mot, 30)) if len(mot) >= 10 else 2.0
        cal[person] = {"open_ear": open_ear, "closed_thr": thr, "motion_thr": max(3.0, 2.5 * still)}
    return cal


def per_second(df, duration, cal):
    df = df.copy()
    df["sec"] = df["t"].astype(int)
    df["closed"] = np.where(df["ear"].isna(), np.nan,
                            (df["ear"] < df["person"].map(lambda p: cal[p]["closed_thr"])).astype(float))
    agg = df.groupby(["person", "sec"]).agg(
        face_visible=("face", "mean"), perclos=("closed", "mean"), ear=("ear", "mean"),
        pitch=("pitch", "median"), motion=("motion", "mean")).reset_index()
    n_sec = int(np.ceil(duration)) if duration else int(agg["sec"].max() + 1)
    out = []
    for person, g in agg.groupby("person"):
        g = g.set_index("sec").reindex(range(n_sec))
        g["person"] = person
        out.append(g.reset_index().rename(columns={"index": "sec"}))
    return pd.concat(out, ignore_index=True) if out else agg


def classify(sec_df, cal, onset_s=5, wake_s=2, max_hidden_s=60, pitch_down=20.0):
    res = []
    for person, g in sec_df.groupby("person", sort=False):
        g = g.sort_values("sec").reset_index(drop=True)
        c = cal[person]
        n = len(g)
        eyes = []  # 'closed' | 'open' | None (no face)
        for _, r in g.iterrows():
            if pd.isna(r["perclos"]) or (r["face_visible"] if not pd.isna(r["face_visible"]) else 0) < 0.3:
                eyes.append(None)
            else:
                eyes.append("closed" if r["perclos"] >= 0.7 else "open")
        still = [(not pd.isna(m)) and m < c["motion_thr"] for m in g["motion"]]
        state = ["awake"] * n
        asleep = False
        closed_run = open_run = hidden_run = 0
        for i in range(n):
            e = eyes[i]
            if e == "closed":
                closed_run += 1; open_run = 0; hidden_run = 0
            elif e == "open":
                open_run += 1; closed_run = 0; hidden_run = 0
            else:
                hidden_run += 1
            moving = (not pd.isna(g.at[i, "motion"])) and not still[i]
            if not asleep:
                if e == "closed" and closed_run >= onset_s and sum(still[i - closed_run + 1:i + 1]) >= 0.8 * closed_run:
                    asleep = True
                    for j in range(i - closed_run + 1, i + 1):  # backdate onset to start of the closure
                        state[j] = "asleep"
                    continue
                if e is None:
                    state[i] = "unknown"  # face hidden; may be upgraded by the pitch rule below
                elif e == "closed":
                    state[i] = "drowsy"
                else:
                    state[i] = "awake"
            else:
                if e == "open" and open_run >= wake_s:
                    asleep = False
                    for j in range(i - open_run + 1, i + 1):
                        state[j] = "awake"
                    continue
                if moving and e != "closed":
                    asleep = False
                    state[i] = "awake"
                    continue
                if e is None and hidden_run > max_hidden_s:
                    asleep = False
                    state[i] = "unknown"
                    continue
                state[i] = "asleep"
        # pitch evidence: a still person whose face disappeared while head was pitched down after drowsiness
        for i in range(1, n):
            if state[i] == "unknown" and state[i - 1] in ("drowsy", "asleep") and still[i]:
                p = g.at[i - 1, "pitch"]
                if not pd.isna(p) and p > pitch_down:
                    state[i] = state[i - 1]
        g["state"] = state
        res.append(g)
    return pd.concat(res, ignore_index=True)


def intervals(cls_df, min_len=1):
    out = {}
    for person, g in cls_df.groupby("person", sort=False):
        ivs, start = [], None
        secs = list(g["sec"])
        states = list(g["state"])
        for s, st in zip(secs + [None], states + ["end"]):
            if st == "asleep" and start is None:
                start = s
            elif st != "asleep" and start is not None:
                end = s if s is not None else secs[-1] + 1
                if end - start >= min_len:
                    ivs.append((int(start), int(end)))
                start = None
        out[person] = ivs
    return out
