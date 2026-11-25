# SortFlow: Waste Image Classifier with an End-to-End MLOps Pipeline

SortFlow classifies a photo of a waste item into one of six material classes (`cardboard`, `glass`, `metal`, `organic`, `paper`, `plastic`) so it can be sorted into the right bin. It is built as a complete ML cycle: data acquisition, preprocessing, model training, evaluation, a prediction API, a web UI, bulk data upload with one-click retraining, Docker deployment on AWS, and load testing with Locust.

| | |
|---|---|
| **Video demo** | `<ADD YOUTUBE LINK>` |
| **Live app (AWS EC2)** | UI: `http://<PUBLIC_IP>:8501` |
| **Live API docs** | `http://<PUBLIC_IP>:8000/docs` |
| **Notebook** | [`notebook/waste_classifier.ipynb`](notebook/waste_classifier.ipynb) |

## Why this project

Waste looks different from place to place: local packaging, lighting and backgrounds all differ from public datasets. A model trained once will drift when it meets local photos. SortFlow lets anyone upload new labelled images and retrain the model from the UI, so the model can keep improving without touching the code.

## Features

- **Predict:** upload one image and get the class, the confidence, and the probability for every class.
- **Visualizations:** class distribution of the training data, live predictions per class, and a confidence histogram, each with an interpretation of what the data shows.
- **Upload data:** upload many images at once for a chosen class. Files are saved to disk and recorded in a SQLite database.
- **Retrain:** a button starts retraining. Retraining also starts automatically when the number of new images reaches a threshold (default 50).
- **Monitoring:** an uptime page shows API status, uptime, whether the model is loaded, whether retraining is running, and average prediction latency. A history table lists every retraining run with its test accuracy and model version.

## Results

Test set: 988 held-out images. Both models are trained from scratch (no pretrained weights).

| Model | Loss | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---|---|---|---|---|
| Baseline CNN (no optimization) | 2.199 | 0.593 | 0.571 | 0.573 | 0.568 |
| **Custom CNN (final)** | **0.542** | **0.817** | **0.819** | **0.799** | **0.805** |

Per-class F1 for the custom model: cardboard 0.876, glass 0.831, metal 0.667, organic 0.918, paper 0.810, plastic 0.728.

**Optimization techniques:** data augmentation, batch normalization, dropout, L2 regularization, Adam, learning-rate scheduling (`ReduceLROnPlateau`), early stopping with best-weights restore, and checkpointing. An earlier version with heavier regularization and class weights reached only 62.8% accuracy. The notebook documents what changed.

**Known limitations**

- The dataset contains augmented and near-duplicate images, so test accuracy is probably optimistic compared to real-world photos.
- The model only knows six classes. Other objects are forced into the closest one, so low-confidence predictions should be treated with care.
- One item per photo is assumed.

## Architecture

```
            ┌────────────┐      ┌──────────────┐      ┌───────────────────┐
 Browser ──▶│ Streamlit  │─────▶│    nginx     │─────▶│ FastAPI container │ x N
            │ UI :8501   │      │ load balancer│      │ (TensorFlow model)│
            └────────────┘      │    :8000     │      └───────────────────┘
                                └──────────────┘        │            │
                                                  SQLite (uploads,   models/ (versioned
                                                  predictions, runs)  .h5 files)
```

### Retraining pipeline

1. **Upload:** images are sent to `/upload` with a class label and saved to `data/uploads/<class>/`. Each file is recorded in SQLite.
2. **Preprocess:** `src/preprocessing.py` resizes images to 224x224 and applies augmentation, the same code used for training.
3. **Retrain:** `src/retrain.py` loads the saved `waste_cnn.h5` as the pre-trained model and fine-tunes it (learning rate 1e-4) on the new uploads mixed with the original training data, to avoid forgetting.
4. **Evaluate and version:** the model is scored on the test set, saved as `models/versions/waste_cnn_vN.h5`, and becomes the active model. The run is logged in the database.
5. **Trigger:** manual (button or `POST /retrain`) or automatic when pending uploads reach `RETRAIN_THRESHOLD`.

## Repository structure

```
SortFlow/
├── README.md
├── notebook/
│   └── waste_classifier.ipynb     # preprocessing, training, evaluation
├── src/
│   ├── preprocessing.py           # resize, augmentation, class weights
│   ├── model.py                   # baseline and custom CNN
│   ├── prediction.py              # model loading and single-image prediction
│   ├── retrain.py                 # retraining pipeline
│   └── db.py                      # SQLite helpers
├── api/main.py                    # FastAPI endpoints
├── ui/app.py                      # Streamlit UI
├── data/
│   ├── train/                     # small sample of the training images
│   └── test/                      # small sample of the test images
├── models/
│   ├── waste_cnn.h5               # active model
│   └── class_names.json           # label order
├── tests/sample.jpg               # image used by Locust
├── locustfile.py                  # load test
├── Dockerfile.api
├── Dockerfile.ui
├── docker-compose.yaml
├── nginx.conf
└── requirements-docker.txt
```

The repository contains only a small sample of the images. The full dataset is downloaded from Kaggle (see below).

## Setup

### Option 1: Docker Compose (recommended)

Requires Docker with the Compose plugin.

```bash
git clone https://github.com/ShamiPlacide/SortFlow.git
cd SortFlow
docker compose up -d --build
```

The first build takes several minutes because TensorFlow is large. Then open:

- UI: http://localhost:8501
- API docs (Swagger): http://localhost:8000/docs

Useful commands:

```bash
docker compose ps                         # container status
docker compose logs -f api                # API logs (wait for "Application startup complete")
docker compose up -d --scale api=3        # run 3 API containers behind nginx
docker compose down                       # stop everything
```

Settings (environment variables on the `api` service): `RETRAIN_THRESHOLD` is the number of new images that triggers automatic retraining (default 50). The UI service reads `API_URL`, which already points at the nginx service in `docker-compose.yaml`.

### Option 2: Run without Docker

Use Python 3.11 or 3.12 and TensorFlow 2.21.0. TensorFlow does not run on older Intel Macs (macOS 10.15), so use Docker there.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-docker.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000        # terminal 1
pip install streamlit pandas                            # UI only, separate environment if you see protobuf conflicts
streamlit run ui/app.py                                 # terminal 2
```

### Reproduce the model

1. Open the notebook on Kaggle (or Colab) with a GPU and add the dataset [Garbage Classification (mostafaabla)](https://www.kaggle.com/datasets/mostafaabla/garbage-classification).
2. Run all cells. The notebook merges the 12 original classes into 6, splits 70/15/15, trains the baseline and the custom CNN, and evaluates both on the test set.
3. Download `waste_cnn.h5` and `class_names.json` into `models/`.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Status, uptime in seconds, model loaded, retraining flag |
| POST | `/predict` | Multipart `file` (image). Returns class, confidence, all probabilities |
| POST | `/upload` | Form field `label` plus multiple `files`. Saves images and logs them. Starts retraining automatically at the threshold |
| POST | `/retrain` | Start retraining in the background (409 if one is already running) |
| GET | `/status` | Retraining flag, pending uploads, threshold, run history |
| GET | `/stats` | Uploads per class and prediction statistics |

Example:

```bash
curl -X POST http://localhost:8000/predict -F "file=@tests/sample.jpg"
```

## Load test (Locust)

The test sends a mix of `/predict` (weight 5) and `/health` (weight 1) requests with 30 simulated users, spawn rate 5 per second, for 60 seconds. Nginx balances requests across the API containers. Each run waited about a minute after scaling so the new containers finished loading TensorFlow.

```bash
docker compose up -d --scale api=1     # then 2, then 3
locust -f locustfile.py --host http://localhost:8000 --headless -u 30 -r 5 -t 60s --csv results_1
```

For an interactive view with charts, run `locust -f locustfile.py --host http://localhost:8000` and open port 8089.

### Results

**Codespaces (load generator and API on the same machine)**

| API containers | `/predict` requests | Failures | Avg latency (ms) | Median (ms) | p95 (ms) | Requests/s |
|---|---|---|---|---|---|---|
| 1 | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<fill>` |
| 2 | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<fill>` |
| 3 | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<fill>` |

**AWS EC2 (`t3.medium`), load generated from Codespaces**

| API containers | `/predict` requests | Failures | Avg latency (ms) | Median (ms) | p95 (ms) | Requests/s |
|---|---|---|---|---|---|---|
| 1 | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<fill>` |
| 2 | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<fill>` |
| 3 | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<fill>` |

Screenshots: `<ADD LINKS OR IMAGES FROM THE LOCUST UI>`

**Interpretation:** `<WRITE 2-3 SENTENCES: how latency and throughput changed with more containers, and why. Inference is CPU-bound, so containers sharing the same few cores stop helping once the CPU is saturated.>`

Each API container has its own SQLite file and upload folder, so uploads and retraining are not shared between replicas. The retraining demo uses a single API container.

## Deployment on AWS

The app runs on one **EC2 `t3.medium`** instance (Ubuntu 24.04, 30 GB disk) using the same `docker-compose.yaml` as local development. The security group allows SSH from one IP and ports 8000 and 8501. An Elastic IP gives the app a stable address.

```bash
sudo apt update && sudo apt install -y docker.io docker-compose-v2 git
git clone https://github.com/ShamiPlacide/SortFlow.git && cd SortFlow
docker compose up -d --build
```

Compose volumes keep `data/uploads`, the SQLite database and `models/` across container restarts, so retrained models are not lost.

## Tech stack

Python, TensorFlow/Keras 2.21, FastAPI, Uvicorn, Streamlit, SQLite, Docker and Docker Compose, nginx, Locust, AWS EC2.

## Data and license

Training data: [Garbage Classification](https://www.kaggle.com/datasets/mostafaabla/garbage-classification) by mostafaabla on Kaggle, licensed under ODbL-1.0.

## Author

Shami Placide