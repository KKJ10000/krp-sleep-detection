---
title: "Sleep Detection from Video: Write-up"
author: "Kshitij Kumar Jha · KRP Solutions, Computer Vision & AI Internship take-home"
---

## 1. What "sleeping" means here

Sleep can't be measured from a camera, but its visible behaviour can. I define a person as **asleep** when:

> **their eyes stay closed for at least 5 consecutive seconds (eyes count as closed in a second when they're shut in ≥ 70% of that second's frames, i.e. PERCLOS ≥ 0.7) while their head and upper body stay still.**

The interval is **backdated** to when the closure began. It **ends** when the eyes are open for 2 consecutive seconds or there's a clear burst of movement. It **continues** through openings shorter than 2 s (a sleeper's eyelid flutter) and through stretches where the face is hidden (head slumped forward or turned into a pillow) as long as the person stays still, for up to 60 s.

Why these numbers: a blink lasts 0.1–0.4 s and a slow "microsleep" blink about 0.5–2 s, so 5 s sits far above both and filters every blink. PERCLOS (percentage of eyelid closure over time) is the standard drowsiness measure in driver-monitoring research, more robust than a single-frame threshold. Stillness separates sleeping from just resting eyes while moving, like rubbing the eyes, laughing, or listening with eyes shut and head nodding along. Between 1 and 5 s of closure I output **drowsy** rather than awake, so the per-second timeline shows the run-up to sleep. Every threshold is a CLI flag (`--onset`, `--wake`), because the right value depends on the use case: a driver needs about 2 s, a classroom or night-shift guard 10 s or more.

## 2. Signals and why

| Signal | How | Why |
| --- | --- | --- |
| Eye Aspect Ratio (EAR) | MediaPipe FaceMesh, 6 landmarks per eye, averaged over both eyes | Works on any face without training; it's a scale-invariant ratio, so distance to the camera doesn't matter |
| Per-person calibration | Closed threshold = 65% of the person's own open-eye EAR (90th percentile), clipped to [0.12, 0.22] | A fixed 0.2 threshold misreads narrow or hooded eyes and glasses as closed; a per-person baseline avoids that |
| Head pitch | `solvePnP` on 6 landmarks | Head dropping forward is a strong secondary cue, and it keeps the sleep state going when the face is no longer visible |
| Motion | Frame difference over the head and upper body, threshold = 2.5× the person's own quiet level | Separates "eyes closed but active" from real sleep, and wakes the state on movement |
| Identity | YuNet face detector + SFace embeddings matched to a small gallery (`gallery/<Name>/*.jpg`), plus an IoU/embedding tracker | Task 2 needs a name rather than a box. The tracker keeps the name when the face turns or is briefly missed. Without a gallery, people become Person 1 and Person 2 |

All models are small, CPU-only and pretrained, with no training data needed. The pipeline runs at 10 analysed frames per second by default.

## 3. How I decided it works

**Ground truth:** per-second labels as `person,start,end` sleep intervals. `scripts/evaluate.py` computes accuracy, precision, recall, F1, false-positive seconds and missed seconds, and checks whether the right person was named as the sleeper.

**Baselines compared:**

- (a) *Blink-level*: any closed frame in a second counts as asleep. This is the naive EAR threshold.
- (b) *Per-second PERCLOS ≥ 0.5* with no duration, stillness or hysteresis.

**Bundled demo clips:** I made two controlled clips with exact ground truth: one person with blinks at 3 s and 6 s and eyes closed from 8–20 s, and two people side by side with only one asleep from 5–17 s. Results:

| System | Single-person F1 | False-positive seconds | Two-person: right sleeper named? |
| --- | --- | --- | --- |
| Blink-level baseline | 0.857 | 4 (the blinks) | yes |
| PERCLOS-only baseline | 1.000 | 0 | yes |
| **This system** | **1.000** | **0** | **yes** |

These clips check correctness of the pipeline (calibration, timing, backdating, naming) rather than real-world accuracy: synthetic eyelids are cleaner than real ones. **For real accuracy, the next step is a labelled set of real clips:** about 20 clips of 1–3 minutes across lighting, glasses, camera angles and 1–2 people, covering hard negatives (looking down at a phone, reading, long blinks, eyes closed while listening to music). Report per-second F1 plus *event-level* metrics (did we catch each sleep episode, and how late was the onset).

## 4. Where it fails, and what I'd do next

**False positives:**

1. **Looking down** at a phone, book or keyboard. The eyelids cover the iris from the camera's angle, so EAR drops. Stillness helps but isn't enough when someone is reading quietly. *Fix:* combine pitch with iris visibility from the refined FaceMesh iris landmarks, and train a small temporal classifier on labelled clips.
2. **Meditating, praying, or resting the eyes while awake.** These look the same as sleep at the eyelid level. *Fix:* add posture cues (head sway, neck muscle tone, slumped shoulders from a pose model), respiration rate from chest motion, and a longer onset (60 s or more) where it matters.
3. **Sunglasses, heavy glare, IR night mode, a face too small** (under about 40 px). *Fix:* detect "eyes not observable" and report `unknown` instead of guessing, which the system already does when no landmarks are found. Add an IR-trained eye-state CNN.

**Missed detections:** someone asleep face-down, or with their back to the camera, never shows their eyes, so they're only covered by the hidden-face rule and come out as `unknown`. *Fix:* a body-pose (MediaPipe Pose or YOLO-pose) stillness model for when the face isn't visible.

**Identity errors:** two similar-looking people at low resolution, or a gallery photo very different from the video (lighting, beard). *Fix:* add frames from the video to the gallery, and use a stronger embedder such as ArcFace/InsightFace.

**With more time:** a labelled real dataset (public options: NTHU-DDD, UTA-RLDD, plus self-recorded clips); a learned eye-state model and a small temporal model (GRU over per-second features) in place of hand-set thresholds; ONNX/TensorRT export to handle many camera streams on one edge box; and confidence scores on every output second.

## 5. Long-term vision

A privacy-first **attention and alertness layer for existing CCTV**: no new hardware, with the processing on an edge box and only events leaving the site.

- **Workplaces:** detect sleeping guards and control-room operators on night shifts, with alerts after N minutes and per-shift reports.
- **Driver and operator monitoring:** forklift and crane operators, fleet cabins, and earlier drowsiness alerts from PERCLOS trends before full sleep.
- **Healthcare and elder care:** sleep and wake logs, sudden falls, and patients out of bed at night.
- **Classrooms and exam halls:** engagement analytics, aggregated with no faces stored.

**Features to integrate:** multi-camera re-identification; yawning and head-nod detection (drowsiness before sleep); pose-based falls and "person down" alerts; a dashboard with timelines and clip export; alerts to WhatsApp, Slack or SMS; per-site threshold tuning; privacy controls (face blurring, embeddings only, on-device processing); and an active-learning loop where operators confirm or reject alerts to retrain the models.
