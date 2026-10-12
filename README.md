# Adapting a YOLOv8n Hand Detector to Webcam Input for Gestural Music Control


AIML339 final project, Samantha Hatchwell, Victoria University of Wellington.

A YOLOv8n hand detector fine-tuned in three stages (EgoHands → webcam → webcam with face in frame), used in a live system that plays notes in Sonic Pi when a hand enters one of four screen zones.

## Contents

- `training_and_tests.ipynb`: training and evaluation of all models (Google Colab)
- `live_system/`: `camera.py` (live detection), `listener.rb` (Sonic Pi ruby file), `face_model_seed0.pt` (Model 3, seed 0)
- `results/`: settings (`args.yaml`), training logs (`results.csv`), plots and weights (`best.pt`) for every run, plus test outputs.


## Dependencies

- Training: Google Colab (T4 GPU), `ultralytics` 8.4.175
- Live system: Python 3, Sonic Pi, and `pip install ultralytics opencv-python python-osc`. `camera.py` uses the Apple Silicon GPU (`"mps"`); change to `"cpu"` on other machines.

## Data

Datasets are not included.
- **EgoHands:** YOLOv8 export from Roboflow Universe, https://universe.roboflow.com/brad-dwyer/egohands-public/dataset/9 (CC BY 4.0)
- **Webcam** and **Extended Webcam:** collected and labelled by the author; not shared, as they contain images of the author.

## Training and evaluation

Run the notebook in Colab. Model 1 was trained once (seed 0) and is loaded from saved weights. Models 2 and 3 were each trained with seeds 0–4 (AdamW, learning rate 0.001, up to 30 epochs). All models are evaluated on each dataset's test split, with Models 2 and 3 reported as mean ± standard deviation over the five seeds.

## Results in the report

- **Table II** (Model 1): notebook "Test Model 1" cells; `results/egohands/tests/`
- **Table III** (Model 2, 5 seeds): notebook "Test model 2" cells and `print_mean_std` outputs; `results/webcam/tests/`
- **Table IV** (Models 2 and 3, Extended Webcam, 5 seeds): notebook "extended webcam" test cells; `results/webcam/tests/` and `results/face/tests/`
- **Fig. 1** (Model 2 training curves, seed 0): `results/webcam/webcam_model_seed0/results.png`
- **Detection latency** (13.7 ms per frame): `live_system/camera.py` with Model 3 (seed 0)
- **Live false-detection test:** informal count with `camera.py`, Models 2 and 3 (seed 0), confidence 0.6
- **Training settings and seeds:** `args.yaml` in each run folder



## Live system

1. Run `listener.rb` in Sonic Pi.
2. Run `python camera.py` from the `live_system` folder.
3. Press `q` to quit; the mean detection time per frame is printed.

## Licence

AGPL-3.0, as required by Ultralytics YOLOv8 (see `LICENSE`).
