"""Per-second evaluation against hand-labelled ground truth, compared with two simpler baselines.

Ground truth CSV: person,start,end   (seconds; one row per sleep interval; a person with no rows never slept)
Usage: python scripts/evaluate.py --pred output/<video>/per_second.csv --gt labels.csv
"""
import argparse

import numpy as np
import pandas as pd


def metrics(y, p):
    tp = int(((y == 1) & (p == 1)).sum()); fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum()); tn = int(((y == 0) & (p == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(accuracy=round((tp + tn) / max(1, len(y)), 3), precision=round(prec, 3), recall=round(rec, 3),
                f1=round(f1, 3), false_pos_s=fp, missed_s=fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    a = ap.parse_args()
    pred = pd.read_csv(a.pred)
    gt = pd.read_csv(a.gt)
    pred["y"] = 0
    for _, r in gt.iterrows():
        m = (pred["person"] == r["person"]) & (pred["sec"] >= r["start"]) & (pred["sec"] < r["end"])
        pred.loc[m, "y"] = 1
    y = pred["y"].values
    systems = {
        "this system (PERCLOS + duration + stillness + hysteresis)": (pred["state"] == "asleep").astype(int).values,
        "baseline: eyes closed this second (PERCLOS>=0.5)": (pred["perclos"].fillna(0) >= 0.5).astype(int).values,
        "baseline: any closed frame (blink-level EAR)": (pred["perclos"].fillna(0) > 0).astype(int).values,
    }
    rows = [dict(system=k, **metrics(y, v)) for k, v in systems.items()]
    print(pd.DataFrame(rows).to_string(index=False))
    # identity check for multi-person clips: did we name the right sleeper?
    slept_pred = set(pred.loc[pred["state"] == "asleep", "person"])
    slept_gt = set(gt["person"])
    print(f"\nWho slept - predicted: {sorted(slept_pred)}  ground truth: {sorted(slept_gt)}  "
          f"{'MATCH' if slept_pred == slept_gt else 'MISMATCH'}")


if __name__ == "__main__":
    main()
