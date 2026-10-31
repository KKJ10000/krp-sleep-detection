import argparse
import json
import os

import pandas as pd

from .decide import calibrate, classify, intervals, per_second
from .features import extract


def fmt(s):
    return f"{int(s) // 60:02d}:{int(s) % 60:02d}"


def main():
    ap = argparse.ArgumentParser(description="Per-second sleep detection for one or more people in a video.")
    ap.add_argument("--video", required=True)
    ap.add_argument("--gallery", default="gallery", help="folder of reference faces: gallery/<Name>/*.jpg")
    ap.add_argument("--out", default="output")
    ap.add_argument("--fps", type=float, default=10.0, help="frames analysed per second")
    ap.add_argument("--onset", type=int, default=5, help="seconds of continuous eye closure before 'asleep'")
    ap.add_argument("--wake", type=int, default=2, help="seconds of open eyes that end a sleep interval")
    ap.add_argument("--annotate", action="store_true", help="also write an annotated video")
    a = ap.parse_args()

    name = os.path.splitext(os.path.basename(a.video))[0]
    out = os.path.join(a.out, name)
    os.makedirs(out, exist_ok=True)
    rows, _, duration = extract(a.video, a.gallery, a.fps, os.path.join(out, "annotated.mp4") if a.annotate else None)
    if not rows:
        print("No faces found in the video.")
        return
    frames = pd.DataFrame(rows)
    frames.to_csv(os.path.join(out, "frames.csv"), index=False)
    cal = calibrate(frames)
    sec = classify(per_second(frames, duration, cal), cal, onset_s=a.onset, wake_s=a.wake)
    cols = ["person", "sec", "state", "perclos", "ear", "pitch", "motion", "face_visible"]
    sec[cols].round(3).to_csv(os.path.join(out, "per_second.csv"), index=False)
    ivs = intervals(sec)

    summary = {"video": a.video, "duration_s": round(duration, 1), "params": {"onset_s": a.onset, "wake_s": a.wake},
               "calibration": cal, "people": {}}
    lines = [f"Video: {a.video} ({duration:.1f}s)"]
    for person, iv in ivs.items():
        total = sum(e - s for s, e in iv)
        summary["people"][person] = {"slept": bool(iv), "asleep_seconds": total,
                                     "intervals": [{"start": s, "end": e, "start_mmss": fmt(s), "end_mmss": fmt(e)} for s, e in iv]}
        verdict = f"{person}: ASLEEP for {total}s -> " + ", ".join(f"{fmt(s)}-{fmt(e)}" for s, e in iv) if iv else f"{person}: not asleep"
        lines.append(verdict)
    sleepers = [p for p, v in summary["people"].items() if v["slept"]]
    awake = [p for p, v in summary["people"].items() if not v["slept"]]
    if len(summary["people"]) > 1:
        lines.append(f"Verdict: sleeping = {sleepers or 'nobody'}; not sleeping = {awake or 'nobody'}")
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nOutputs in {out}/ (per_second.csv, summary.json, summary.txt{', annotated.mp4' if a.annotate else ''})")


if __name__ == "__main__":
    main()
