# Sleep detection from video (KRP Solutions take-home)

Given a video, outputs a **per-second judgement** (`awake` / `drowsy` / `asleep` / `unknown`) for every person in it,
plus a summary of the intervals where each person was asleep. With 2+ people it says **by name** who was sleeping,
using a small gallery of reference photos.

## Run it (one command)

Requires Python 3.10-3.12 (tested on 3.12, CPU only).

```bash
./run.sh path/to/video.mp4                 # task 1: one person
./run.sh path/to/two_people.mp4 gallery    # task 2: gallery/<Name>/*.jpg gives each person their name
```

The first run creates `.venv/` with the pinned dependencies (`requirements.txt`) and downloads two small
OpenCV Zoo models (YuNet face detector 0.2 MB, SFace face embedder 37 MB) into `models/`.

Docker alternative:

```bash
docker build -t sleepdet .
docker run --rm -v "$PWD:/data" sleepdet --video /data/video.mp4 --gallery /data/gallery --out /data/output --annotate
```

Try it on the bundled demo clips (synthetic, see write-up):

```bash
./run.sh tests/single.mp4 tests/gallery_demo
./run.sh tests/two.mp4 tests/gallery_demo
```

## Output (`output/<video_name>/`)

| File | What it is |
| --- | --- |
| `summary.txt` | Human-readable verdict, e.g. `Lena: ASLEEP for 12s -> 00:05-00:17` / `Messi: not asleep` |
| `summary.json` | Same, machine-readable, with per-person calibration values |
| `per_second.csv` | One row per person per second: state, PERCLOS, mean EAR, head pitch, motion, face visibility |
| `frames.csv` | Raw per-frame signals (for debugging or re-tuning without re-running video) |
| `annotated.mp4` | Video with boxes, names and eye-openness overlay |

Options: `--fps` (frames analysed per second, default 10), `--onset` (seconds of continuous closure before
"asleep", default 5), `--wake` (seconds of open eyes that end sleep, default 2).

## Evaluate against labels

```bash
.venv/bin/python scripts/evaluate.py --pred output/two/per_second.csv --gt tests/two_labels.csv
```

Labels are `person,start,end` sleep intervals in seconds. The script reports per-second accuracy, precision,
recall, F1, false-positive seconds and missed seconds for this system and two simpler baselines, and checks
whether the right person was named as the sleeper.

## How it works (short)

1. **Faces**: YuNet detects every face; SFace embeds it; a gallery match (cosine >= 0.363) gives the name, and an
   IoU + embedding tracker keeps identities stable when a face is briefly lost.
2. **Signals per face**: MediaPipe FaceMesh (478 landmarks) -> Eye Aspect Ratio (both eyes), head pitch via
   `solvePnP`; frame differencing over head + upper body -> motion.
3. **Per-person calibration**: closed-eye threshold = 65% of that person's own open-eye EAR (90th percentile),
   clipped to [0.12, 0.22], so narrow eyes / glasses don't count as "closed".
4. **Per-second state machine**: PERCLOS >= 0.7 in a second = eyes closed. Closed for `onset` consecutive seconds
   while still -> asleep (backdated to the start of the closure). Stays asleep through brief openings, and through
   a hidden face if the person stays still; wakes on `wake` seconds of open eyes or a motion burst.

See `WRITEUP.pdf` for the operational definition, signal choices, evaluation and failure modes.

## Layout

```
sleepdet/features.py   per-frame detection, identity, EAR, pitch, motion
sleepdet/decide.py     calibration, per-second aggregation, sleep state machine, intervals
sleepdet/__main__.py   CLI and output files
scripts/evaluate.py    metrics vs ground truth and baselines
tests/                 synthetic demo clips, labels, demo gallery
```
