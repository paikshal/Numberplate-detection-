# 🚗 Indian Number Plate Detection System

A real-time Indian number plate detection and OCR system built using **YOLOv8**, **Haarcascade**, and **EasyOCR**. Also includes a **Faster R-CNN** training pipeline on a custom annotated dataset.

---

## 📌 Features

- 🔍 **Vehicle Detection** using YOLOv8 (COCO pretrained — car, motorcycle, bus, truck)
- 🪪 **Number Plate Localization** using Haarcascade classifier
- 🔤 **OCR Text Extraction** using EasyOCR with image preprocessing
- 📸 **Two Modes** — Image mode & Live Camera mode
- 💾 **Auto Save** detected plates as image + text files
- 🏋️ **Custom Model Training** using Faster R-CNN on annotated Indian plate dataset

---

## 📂 Project Structure

```
number_platedection/
│
├── plate.py                  # Main detection & OCR script (YOLOv8 + EasyOCR)
├── train_rcnn.py             # Faster R-CNN training script
├── yolov8n.pt                # YOLOv8 Nano pretrained model weights
├── requirements.txt          # Python dependencies
├── SRS.md                    # Software Requirements Specification
│
├── Indian_Number_Plates/     # Sample training images dataset
│   └── Sample_Images/
│
├── Annotations/              # XML annotations (Pascal VOC format)
│   └── Annotations/
│
├── number_plate_images_ocr/  # OCR-focused dataset images
├── number_plate_annos_ocr/   # OCR-focused dataset annotations
│
└── cs/                       # Output folder (saved detected plates)
```

---

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/paikshal/Numberplate-detection-.git
cd Numberplate-detection-
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install PyTorch (CPU)
```bash
pip install torch torchvision
```
> For GPU (CUDA) support, visit [pytorch.org](https://pytorch.org/get-started/locally/) for the correct command.

### 4. Install YOLOv8
```bash
pip install ultralytics
```

---

## 🚀 Running the Project

### ▶️ Detection & OCR (`plate.py`)
```bash
python plate.py
```

You will be prompted to choose a mode:

| Choice | Mode | Description |
|--------|------|-------------|
| `1` | Image Mode | Provide path to an image file |
| `2` | Camera Mode | Uses webcam for live detection |

**Camera Mode Controls:**
- Press `s` — Capture and scan current frame
- Press `q` — Quit camera mode

**Output** is saved automatically in the `cs/` folder as:
- `plate_<timestamp>.jpg` — Cropped plate image
- `plate_<timestamp>.txt` — Detected number plate text

---

## 🏋️ Training Custom Model (`train_rcnn.py`)

Train a custom Faster R-CNN model on the included annotated dataset:

```bash
python train_rcnn.py
```

- Uses **Faster R-CNN ResNet50 FPN** backbone
- Trains on Indian number plate images with Pascal VOC XML annotations
- Saves trained weights to `custom_plate_model.pth`

---

## 🧠 How It Works

```
Input (Image/Camera)
        │
        ▼
  YOLOv8 Detection
  (Detect Vehicles)
        │
        ▼
  Haarcascade Classifier
  (Locate Plate inside Vehicle)
        │
        ▼
  Image Preprocessing
  (Resize → Sharpen → Grayscale)
        │
        ▼
  EasyOCR
  (Extract Plate Text)
        │
        ▼
  Save Result (Image + Text)
```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `easyocr` | 1.7.1 | OCR text recognition |
| `opencv-python` | 4.10.0.84 | Image processing & display |
| `imutils` | 0.5.4 | Image utility functions |
| `numpy` | <2 | Numerical operations |
| `scipy` | <2 | Scientific computing |
| `Pillow` | 10.4.0 | Image handling |
| `ultralytics` | latest | YOLOv8 model |
| `torch` + `torchvision` | latest | Deep learning (Faster R-CNN) |

---

## 🖼️ Dataset

- **Indian_Number_Plates/** — 27 sample images of Indian vehicles with number plates
- **Annotations/** — Pascal VOC format XML annotations for each image
- **number_plate_images_ocr/** — Additional images focused on OCR tasks
- **number_plate_annos_ocr/** — Corresponding annotations for OCR dataset

---

## 👤 Author

**Paikshal Prajapati**  
GitHub: [@paikshal](https://github.com/paikshal)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
