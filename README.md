# FiSmo Fire & Smoke Detection Model

**Notebook:** `model.ipynb`  
**Platform:** Google Colab (Python 3, GPU Runtime — NVIDIA T4 recommended)  
**Framework:** YOLOv8 (Ultralytics) + PyTorch  
**Dataset:** FiSmo-Images (BoWFire subset for benchmarking + 180 custom-annotated Flickr-FireSmoke images for training)  

---

## How to Set Up

### What to Download
Download the dataset folder from the shared Google Drive link below and place it in your Google Drive as `fismo-detection/`:

> **Dataset & Weights:** [Google Drive Link](https://drive.google.com/drive/folders/1No2B1Et_NdHIB1tlfbOChgDx_ZSrW77b?usp=sharing)

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


## Running the Code

Run all cells in order from top to bottom.

--- 

### 1. Directory Inspection
Verifies the Google Drive is mounted correctly and all required folders are in place before any processing begins.

---

### 2. OS Implementation
Demonstrates core OS concepts through four benchmarks, each tied directly to the image processing pipeline.

**File I/O:** Five file reading methods (High-level Text, High-level Binary, Low-level OS, Buffered 16KB, and Memory Mapped) are benchmarked over 100 iterations on a BoWFire image using `timeit`. Low-level OS and Buffered reads were the fastest (~2.9ms); High-level Text was the slowest (~4.3ms) due to encoding conversion overhead.

**CPU vs GPU:** A 10,000×10,000 matrix multiplication is timed on both devices to demonstrate OS-level hardware allocation. The Tesla T4 GPU was consistently 10–50× faster, with `torch.cuda.synchronize()` used for accurate async timing.

**Multiprocessing:** 100 simulated preprocessing tasks are run sequentially then in parallel via `multiprocessing.Pool(processes=2)`, achieving ~1.9× speedup by bypassing the Python GIL.

**Multithreading & Synchronization:** 100 threads simulate concurrent disk writes, gated by `threading.Semaphore(4)` to cap I/O concurrency at 4 threads at a time, it was chosen because allowing all 100 threads to write simultaneously would saturate the disk and cause I/O errors or significant slowdown. A `threading.Lock()` ensures safe console output, and `os.fsync()` guarantees each write is physically committed to disk before the thread exits.

---

### 3. Model Training
Covers dataset preparation and model training end to end.

The 180 annotated images are paired with their label files, shuffled with `seed=42` for reproducibility, and split 80/20 into Training (144 images) and Validation (36 images) across 5 categories: `orange_fire` (32), `red_fire` (15), `light_smoke` (26), `dense_smoke` (32), and `neither` (114).

A **YOLOv8n** model is trained for up to 150 epochs using GPU device, batch size 32, 4 multiprocessing data loader workers, and full RAM caching to eliminate disk I/O bottlenecks during training. Early stopping at `patience=25` prevents over-running. HSV jitter, rotation, scaling, and horizontal flip augmentations are applied to combat overfitting and improve recall on the minority fire classes.

---

### 4. Results
Evaluates the trained model through three lenses: metrics, images, and video.

**Performance Summary:** Reads `results.csv` and reports the final epoch's Precision, Recall, and mAP@50.

**Detected Images:** 3 random validation images are passed through the model at `conf=0.25` and displayed with bounding boxes annotating detected fire and smoke regions.

**Video Demo:** A raw video is processed frame-by-frame with bounding box overlays, re-encoded to H.264 via FFmpeg, and played back inline in the notebook.

**Visualization Dashboard:** Four charts summarising the project: File I/O read times, Multiprocessing speedup, Training vs Validation loss curves, and Precision/Recall/mAP progression over epochs.
