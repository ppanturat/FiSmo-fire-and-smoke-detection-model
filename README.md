# FiSmo Fire & Smoke Detection Model

**Notebook:** `model.ipynb`
**Platform:** Google Colab (Python 3, GPU Runtime — NVIDIA T4 recommended)
**Framework:** YOLOv8 (Ultralytics) + PyTorch
**Dataset:** FiSmo-Images (BoWFire subset for benchmarking + 180 custom-annotated Flickr-FireSmoke images for training)

---

## How to Set Up

### What to Download
Download the dataset folder from the shared Google Drive link below and place it in your Google Drive as `fismo-detection/`:

> **Dataset & Weights:** [Google Drive Link](YOUR_GOOGLE_DRIVE_LINK_HERE)

The folder contains:
- `raw_annotated_data/` — 180 annotated images and their MakeSense.ai label files
- `bowfire/` — BoWFire images (used for File I/O benchmarking only)
- `data/yolo/dataset.yaml` — YOLOv8 data config

All required libraries (`ultralytics`, `torch`, `cv2`, `psutil`, `matplotlib`, `pandas`) are either pre-installed on Colab or installed automatically by the notebook.

### Environment Setup
1. Open `model.ipynb` in **Google Colab**
2. Go to **Runtime → Change runtime type** and select **T4 GPU**
3. Run **Cell 0 (Project Setup)** first — it mounts your Google Drive and automatically creates all required folders

### File Setup
After placing the downloaded folder in your Google Drive, your structure should look like this:
```
MyDrive/
└── fismo-detection/
    ├── raw_annotated_data/
    │   ├── images/               # 665 raw images
    │   └── annotations/          # 180 matched label files (.txt)
    ├── bowfire/
    │   └── img/                  # BoWFire images (benchmark only)
    ├── data/
    │   └── yolo/
    │       ├── dataset.yaml
    │       ├── images/
    │       │   ├── train/        # Auto-populated (144 images)
    │       │   └── val/          # Auto-populated (36 images)
    │       └── labels/
    │           ├── train/        # Auto-populated (144 labels)
    │           └── val/          # Auto-populated (36 labels)
    └── models/
        └── fismo_global_model/
            └── weights/
                ├── best.pt       # Best model weights (saved after training)
                └── last.pt       # Last epoch weights
```

---

## Running the Code

Run all cells in order from top to bottom.

### 1. Directory Inspection
Inspects the project folder structure using `ls -al` to verify everything is in place before processing begins.

### 2. OS Implementation

**File I/O Benchmark**
Five different file reading methods are benchmarked on a BoWFire image over 100 iterations using `timeit`: High-level Text, High-level Binary, Low-level OS, Buffered (16KB), and Memory Mapped. Low-level OS and Buffered reads were the fastest (~2.9ms), while High-level Text was slowest (~4.3ms) due to encoding overhead.

**CPU vs GPU Benchmark**
Matrix multiplication (10,000×10,000) is timed on both CPU and GPU to demonstrate OS-level hardware resource allocation. The Tesla T4 GPU was typically 10–50× faster than CPU.

**Multiprocessing**
100 simulated image preprocessing tasks are run sequentially vs. in parallel using `multiprocessing.Pool(processes=2)`, demonstrating ~1.9× speedup by bypassing the Python GIL.

**Multithreading & Synchronization**
100 threads simulate concurrent file writes, gated by a `threading.Semaphore(4)` to prevent I/O saturation. A `threading.Lock()` prevents interleaved console output, and `os.fsync()` ensures each write is physically committed to disk.

### 3. Model Training
The dataset preparation cell scans the 180 annotated images, pairs each with its label file, shuffles with `seed=42`, and splits 80/20 into Training (144 images) and Validation (36 images). The 5 categories are: `orange_fire` (32), `red_fire` (15), `light_smoke` (26), `dense_smoke` (32), and `neither` (114).

A **YOLOv8n** model is then trained for up to 150 epochs with GPU device, batch size 32, 4 multiprocessing workers, RAM caching, and early stopping at patience=25. Data augmentations (HSV jitter, rotation, scaling, horizontal flip) are applied to combat overfitting and improve recall.

### 4. Results

**Performance Summary**
Reads the `results.csv` log and displays the final epoch's Precision, Recall, and mAP@50 scores.

**Detected Images**
3 random validation images are run through the trained model at `conf=0.25` and displayed with bounding boxes drawn around detected fire and smoke regions.

**Video Demo**
A video from the raw data folder is processed frame-by-frame, re-encoded to H.264 via FFmpeg, and displayed inline in the notebook.

**Visualization Dashboard**
Four report-ready charts are generated: File I/O performance comparison, Multiprocessing speedup, Training loss curves (train vs. val), and Precision/Recall/mAP curves over epochs.
