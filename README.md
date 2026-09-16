# YOLO11n Drone Detector Starter

This scaffold trains and evaluates a one-class `drone` detector using Ultralytics YOLO11n.
It is intentionally separate from the turret controller: validate detection on recorded footage
before allowing predictions to influence motion or any laser state.

## 1. Install

Use Python 3.11 in a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install the CUDA-enabled PyTorch build appropriate for the host before `requirements.txt` if the
default PyTorch package does not detect the RTX 4070. Verify with:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 2. Dataset layout

Place images and YOLO text labels in this structure:

```text
datasets/drone/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

Every positive image needs a matching label file with the same stem. For example,
`images/train/frame_001.jpg` uses `labels/train/frame_001.txt`.

Each label row is:

```text
class_id x_center y_center width height
```

Coordinates are normalized to `[0, 1]`. Since this is a one-class model, `class_id` is always `0`.
An image with no drone may have either an empty label file or no label file.

Keep complete recording sessions together. Do not randomly split neighboring video frames across
train, validation, and test; that leaks nearly identical images into evaluation. A good initial
split is roughly 70/15/15 by recording session.

## 3. Validate the dataset

```powershell
python validate_dataset.py --data dataset.yaml
```

The validator checks missing images, malformed rows, invalid class IDs, and coordinates outside
the YOLO range. Missing labels are reported as negative images, not errors.

## 4. Train

```powershell
python train.py
```

The default command fine-tunes `yolo11n.pt` at `imgsz=1280`. The first run downloads the pretrained
checkpoint. Results are written under `runs/drone/yolo11n_baseline/`.

Useful overrides:

```powershell
python train.py --epochs 100 --batch 8 --device 0
python train.py --model yolo11s.pt --name yolo11s_comparison
```

Start with YOLO11n. Compare YOLO11s only after the full pipeline works and only keep it if held-out
recall/center stability improves enough to justify its latency.

## 5. Evaluate the untouched test split

```powershell
python evaluate.py --weights runs/drone/yolo11n_baseline/weights/best.pt
```

Do not repeatedly tune against the test set. Use validation results during development and run the
test split when choosing a release candidate.

## 6. Run on a video or webcam

```powershell
python predict.py --weights runs/drone/yolo11n_baseline/weights/best.pt --source path\to\video.mp4 --show
python predict.py --weights runs/drone/yolo11n_baseline/weights/best.pt --source 0 --show
```

Predictions are saved by default. Add `--no-save` when only displaying results. This script is for
offline detector testing; it does not send commands to the turret.

## Data-collection reminders

- Capture through the actual cameras with the final focus, exposure, gain, and IR-light policy.
- Include 2–5 m distances, different orientations, motion blur, hand occlusion, and frame edges.
- Collect negative frames containing hands, faces, backpacks, phones, fans, and empty classrooms.
- Train primarily on laser-off frames so the model cannot learn the laser dot as a shortcut.
- Label consistently: use the visible drone boundary for every partially occluded example.
- After the first model, run it on new recordings and label its false positives and missed drones.

