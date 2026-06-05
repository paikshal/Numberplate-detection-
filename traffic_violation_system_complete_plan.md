# Traffic Violation Detection System
## Complete Research, Architecture & Implementation Plan

---

## 1. PROJECT OVERVIEW

**Goal:** Automatically detect traffic violations (triple riding, no helmet) using cameras, extract number plates via OCR, and generate challans — with minimum hardware cost and free tools only.

**Core Pipeline:**
```
Camera Feed → AI Detection → Rule Engine → OCR → Backend → Challan → SMS to Owner
```

---

## 2. SYSTEM DESIGN ARCHITECTURE

### 2.1 Layer-by-Layer Breakdown

```
┌─────────────────────────────────────────────────────────┐
│  LAYER 1: CAMERA / EDGE DEVICE                          │
│  Raspberry Pi 4 (4GB) + 1080p USB/IP Camera             │
│  Runs AI inference locally — no raw video to cloud      │
└───────────────────┬─────────────────────────────────────┘
                    │ frames (15-20 fps)
                    ▼
┌─────────────────────────────────────────────────────────┐
│  LAYER 2: AI DETECTION PIPELINE                         │
│  ┌──────────┐  ┌────────────┐  ┌──────────┐  ┌───────┐ │
│  │ YOLOv8n  │  │ Helmet     │  │ Rider    │  │Paddle │ │
│  │ Bike +   │→ │ Detector   │→ │ Counter  │→ │ OCR   │ │
│  │ Person   │  │ Custom     │  │ ≥3 flag  │  │ Plate │ │
│  └──────────┘  └────────────┘  └──────────┘  └───────┘ │
└───────────────────┬─────────────────────────────────────┘
                    │ violation_event JSON
                    ▼
┌─────────────────────────────────────────────────────────┐
│  LAYER 3: RULE ENGINE (Python)                          │
│  IF riders ≥ 3 → TRIPLE_RIDING                         │
│  IF helmet == False (any rider) → NO_HELMET             │
│  IF both → BOTH                                         │
│  confidence < 0.65 → SKIP                               │
│  confidence 0.65-0.85 → MANUAL_REVIEW                   │
│  confidence > 0.85 → AUTO_CHALLAN                       │
└───────────────────┬─────────────────────────────────────┘
                    │ HTTP POST (REST API)
                    ▼
┌─────────────────────────────────────────────────────────┐
│  LAYER 4: FASTAPI BACKEND                               │
│  ┌──────────────┐ ┌─────────────┐ ┌──────────────────┐ │
│  │ Violation    │ │ OCR         │ │ Challan          │ │
│  │ Service      │ │ Validator   │ │ Generator        │ │
│  │ save + dedup │ │ plate regex │ │ PDF + SMS        │ │
│  └──────────────┘ └─────────────┘ └──────────────────┘ │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│  LAYER 5: STORAGE                                       │
│  PostgreSQL (Supabase free)  → challans, vehicles, logs │
│  Redis (Upstash free)        → dedup cache (60 min TTL) │
│  Cloudinary (free 25GB)      → violation images/clips   │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│  LAYER 6: ADMIN DASHBOARD (React + Tailwind)            │
│  Live feed | Challan list | Manual review | Analytics   │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│  LAYER 7: NOTIFICATION                                  │
│  Twilio / MSG91 → SMS to vehicle owner                  │
└─────────────────────────────────────────────────────────┘
```

---

### 2.2 Database Design (PostgreSQL)

```sql
-- Table 1: Vehicles (populate from RTO data or manually)
CREATE TABLE vehicles (
    plate_number    VARCHAR(15)  PRIMARY KEY,
    owner_name      VARCHAR(100) NOT NULL,
    owner_phone     VARCHAR(15),
    owner_email     VARCHAR(100),
    vehicle_type    VARCHAR(20),
    registered_city VARCHAR(50),
    created_at      TIMESTAMP DEFAULT NOW()
);

-- Table 2: Cameras
CREATE TABLE cameras (
    id              VARCHAR(20)  PRIMARY KEY,  -- e.g. "CAM_MG_ROAD_01"
    location_name   VARCHAR(100),
    latitude        DECIMAL(9,6),
    longitude       DECIMAL(9,6),
    is_active       BOOLEAN DEFAULT TRUE
);

-- Table 3: Violations
CREATE TABLE violations (
    id              UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    plate_number    VARCHAR(15)  REFERENCES vehicles(plate_number),
    violation_type  VARCHAR(20)  CHECK (violation_type IN ('triple_riding','no_helmet','both')),
    camera_id       VARCHAR(20)  REFERENCES cameras(id),
    captured_at     TIMESTAMP    NOT NULL,
    image_path      TEXT,                      -- Cloudinary URL
    confidence      FLOAT,                     -- AI confidence 0-1
    rider_count     INTEGER,
    helmet_detected BOOLEAN,
    status          VARCHAR(20)  DEFAULT 'pending'
                    CHECK (status IN ('pending','challan_issued','disputed','resolved','skipped'))
);

-- Table 4: Challans
CREATE TABLE challans (
    id              UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    violation_id    UUID         REFERENCES violations(id),
    plate_number    VARCHAR(15),
    amount          INTEGER      NOT NULL,     -- in rupees
    violation_type  VARCHAR(20),
    issued_at       TIMESTAMP    DEFAULT NOW(),
    due_date        DATE,
    paid            BOOLEAN      DEFAULT FALSE,
    pdf_url         TEXT,
    sms_sent        BOOLEAN      DEFAULT FALSE
);

-- Table 5: Manual Review Queue
CREATE TABLE review_queue (
    id              UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    violation_id    UUID         REFERENCES violations(id),
    confidence      FLOAT,
    assigned_to     VARCHAR(50),
    reviewed_at     TIMESTAMP,
    decision        VARCHAR(20)  CHECK (decision IN ('approve','reject','escalate'))
);

-- Indexes for performance
CREATE INDEX idx_violations_plate    ON violations(plate_number);
CREATE INDEX idx_violations_captured ON violations(captured_at);
CREATE INDEX idx_challans_plate      ON challans(plate_number);
CREATE INDEX idx_challans_paid       ON challans(paid);
```

### 2.3 Redis Schema (Deduplication)

```
KEY FORMAT:  seen:{plate_number}:{camera_id}
VALUE:       violation_id (UUID)
TTL:         3600 seconds (1 hour)

EXAMPLE:     seen:MH12AB1234:CAM_MG_ROAD_01
LOGIC:       Same plate + same camera within 1 hour = skip challan
```

---

## 3. AI MODELS — COMPLETE DETAILS

### 3.1 Models Required (3 separate models)

| Model | Purpose | Architecture | Size | FPS on Pi 4 |
|---|---|---|---|---|
| `helmet_detector.onnx` | Detect helmet / no helmet | YOLOv8n custom | ~6MB | 15-20 |
| `rider_counter.onnx` | Count people on bike | YOLOv8n (COCO pretrained) | ~6MB | 15-20 |
| PaddleOCR (built-in) | Extract plate text | PaddleOCR v4 | ~8MB | 10-15 |

**Note:** PaddleOCR ko train nahi karna — already Indian plates pe kaam karta hai.

### 3.2 Model Classes

```yaml
# helmet_detector/data.yaml
names:
  0: helmet
  1: no_helmet
nc: 2

# rider_counter/data.yaml  
names:
  0: motorcycle
  1: person
nc: 2
```

---

## 4. DATASET — SOURCES (ALL FREE & PUBLIC)

### 4.1 Ready-Made Datasets

| Dataset | Platform | Images | Download Link | Format |
|---|---|---|---|---|
| Helmet Detection India | Roboflow Universe | 5,200+ | universe.roboflow.com/search?q=helmet+india | YOLOv8 |
| Indian Number Plate Detection | Roboflow Universe | 8,400+ | universe.roboflow.com/search?q=number+plate+india | YOLOv8 |
| Motorcycle Rider Detection | Roboflow Universe | 3,100+ | universe.roboflow.com/search?q=motorcycle+rider | YOLOv8 |
| Indian Vehicle Number Plate OCR | Kaggle | 15,000+ | kaggle.com/datasets/search?q=indian+number+plate | Images |
| Helmet vs No Helmet Classification | Kaggle | 7,200+ | kaggle.com/datasets/search?q=helmet+detection | YOLOv8 |
| COCO Dataset (person + vehicle) | COCO | 118,000+ | cocodataset.org (pretrained weights use karo) | — |

### 4.2 Roboflow Se Dataset Download Kaise Karo

```python
# Step 1: Free account banao at roboflow.com
# Step 2: API key copy karo (Settings > API)

pip install roboflow

from roboflow import Roboflow

rf = Roboflow(api_key="TUMHARA_API_KEY_YAHAN")

# Helmet dataset
project = rf.workspace().project("helmet-detection-india-xxxx")
dataset  = project.version(1).download("yolov8")
# Downloads to: ./helmet-detection-india-1/

# Number plate dataset
project2 = rf.workspace().project("indian-number-plate-detection-xxxx")
dataset2  = project2.version(1).download("yolov8")
```

### 4.3 Apna Data Collect Karna (Accuracy Boost ke liye)

**Minimum images required:**

| Class | Min Images | Source |
|---|---|---|
| helmet | 500 | YouTube Indian traffic videos, screenshots |
| no_helmet | 500 | Same |
| single rider | 300 | Same |
| double rider | 300 | Same |
| triple rider | 300 | Same — rare, isliye 300 kaafi |
| number plate clear | 1,000 | Road photos, any time |
| number plate blurry/angled | 500 | Movement blur simulate |

**Free annotation tool:**
- **Roboflow** (browser based, recommended): roboflow.com → New Project → Upload → Annotate
- **LabelImg** (offline): `pip install labelImg` → `labelImg` command se run karo

**Data augmentation (Roboflow mein free):**
Set karo yeh augmentations — 500 images effectively 2,000+ ban jaate hain:
- Brightness: -30% to +30%
- Blur: up to 2px
- Horizontal Flip
- Rotation: -10° to +10°
- Noise: 2%
- Crop: 0% to 20%

---

## 5. GPU & COMPUTE REQUIREMENTS

### 5.1 Training Requirements

| Platform | GPU | VRAM | Cost | Training Time (5k imgs, 50 epochs) | Recommended? |
|---|---|---|---|---|---|
| Google Colab Free | T4 | 16GB | FREE | 2-3 hours | ✅ Best for beginners |
| Kaggle Notebooks | T4 / P100 | 16GB | FREE | 3-4 hours | ✅ Good backup |
| Google Colab Pro | A100 | 40GB | ₹900/month | 40-50 min | Optional |
| Local PC (RTX 3060) | — | 12GB | Already owned | 1-2 hours | ✅ If available |
| Local PC (CPU only) | — | — | — | 20-40 hours | ❌ Mat karo |

**Verdict: Google Colab free T4 GPU bilkul kaafi hai YOLOv8n ke liye.**

### 5.2 Inference (Camera pe Run Karna) Requirements

| Device | RAM | Cost | FPS (YOLOv8n) | Usable? |
|---|---|---|---|---|
| Raspberry Pi 4 (4GB) | 4GB RAM | ~₹5,000 | 15-20 fps | ✅ Good |
| Raspberry Pi 5 (8GB) | 8GB RAM | ~₹7,500 | 25-30 fps | ✅ Better |
| Jetson Nano (4GB) | 4GB RAM + GPU | ~₹9,000 | 35-45 fps | ✅ Best budget |
| Laptop (no GPU) | 8GB RAM | Already owned | 5-8 fps | ⚠️ Slow |
| PC with any GPU | 8GB RAM | Already owned | 60+ fps | ✅ |

**Minimum viable:** Raspberry Pi 4 (4GB) + YOLOv8n = 15-20 fps = kaafi hai real traffic ke liye.

---

## 6. COMPLETE FREE TECH STACK

| Layer | Tool | Free Limit | Link |
|---|---|---|---|
| AI Framework | Ultralytics YOLOv8 | Free (open source) | github.com/ultralytics/ultralytics |
| OCR | PaddleOCR | Free (open source) | github.com/PaddlePaddle/PaddleOCR |
| Training Platform | Google Colab | 15 hrs GPU/day free | colab.research.google.com |
| Dataset Platform | Roboflow | 10k images free | roboflow.com |
| Dataset Platform 2 | Kaggle | Unlimited free | kaggle.com |
| Backend Framework | FastAPI (Python) | Open source | fastapi.tiangolo.com |
| Database | Supabase (PostgreSQL) | 500MB free | supabase.com |
| Cache | Upstash (Redis) | 10k req/day free | upstash.com |
| Image Storage | Cloudinary | 25GB free | cloudinary.com |
| Backend Hosting | Railway.app | 500 hrs/month free | railway.app |
| Frontend | React + Tailwind | Open source | — |
| Frontend Hosting | Vercel | Free forever | vercel.com |
| SMS | Twilio | $15 free credit | twilio.com |
| Model Export | ONNX Runtime | Open source | onnxruntime.ai |

**Total Monthly Cost: ₹0 (MVP ke liye)**

---

## 7. MODEL TRAINING — STEP BY STEP CODE

### 7.1 Google Colab Setup

```python
# Cell 1: Install dependencies
!pip install ultralytics roboflow paddlepaddle paddleocr opencv-python

# Cell 2: Check GPU
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
# Output should show: Tesla T4
```

### 7.2 Download Dataset

```python
# Cell 3: Download helmet dataset from Roboflow
from roboflow import Roboflow

rf = Roboflow(api_key="APNI_KEY_YAHAN_DALO")
project = rf.workspace("your-workspace").project("helmet-detection-india")
dataset  = project.version(1).download("yolov8")

print("Dataset path:", dataset.location)
# e.g. /content/helmet-detection-india-1/
```

### 7.3 Train YOLOv8n

```python
# Cell 4: Train helmet detection model
from ultralytics import YOLO

model = YOLO('yolov8n.pt')   # Downloads 6MB pretrained weights

results = model.train(
    data    = '/content/helmet-detection-india-1/data.yaml',
    epochs  = 50,            # 50 epochs kaafi hai
    imgsz   = 640,           # standard input size
    batch   = 16,            # T4 ke liye 16 fit hoga
    device  = 0,             # use GPU
    patience = 10,           # early stop if no improvement for 10 epochs
    augment  = True,         # auto augmentation ON
    name    = 'helmet_v1',
    project = '/content/models'
)

# Cell 5: Check accuracy
print(f"mAP50:    {results.results_dict['metrics/mAP50(B)']:.3f}")
print(f"mAP50-95: {results.results_dict['metrics/mAP50-95(B)']:.3f}")
# Target: mAP50 > 0.85
```

### 7.4 Model Validation

```python
# Cell 6: Validate on test set
model = YOLO('/content/models/helmet_v1/weights/best.pt')
metrics = model.val()

print(f"Precision: {metrics.box.mp:.3f}")
print(f"Recall:    {metrics.box.mr:.3f}")
print(f"mAP50:     {metrics.box.map50:.3f}")

# Targets:
# mAP50     > 0.85   = good
# mAP50     > 0.90   = excellent
# Precision > 0.88   = low false positives
# Recall    > 0.82   = low missed detections
```

### 7.5 Export for Raspberry Pi

```python
# Cell 7: Export to ONNX (best for Raspberry Pi)
model.export(
    format   = 'onnx',
    imgsz    = 640,
    optimize = True,
    simplify = True
)
# Output: /content/models/helmet_v1/weights/best.onnx

# Download to local machine
from google.colab import files
files.download('/content/models/helmet_v1/weights/best.onnx')
```

---

## 8. LIVE DETECTION SCRIPT (Raspberry Pi / Laptop)

```python
# detection_engine.py
import cv2
import numpy as np
import requests
import redis
from ultralytics import YOLO
from paddleocr import PaddleOCR
from datetime import datetime
import uuid

# Load models
helmet_model = YOLO('models/helmet_v1.onnx')
rider_model  = YOLO('models/rider_counter.onnx')
ocr          = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

# Redis for dedup
r = redis.Redis(host='localhost', port=6379, db=0)

CAMERA_ID  = "CAM_MAIN_ROAD_01"
BACKEND_URL = "http://localhost:8000/api/violations"

CONF_AUTO   = 0.85   # Auto challan threshold
CONF_REVIEW = 0.65   # Manual review threshold

def extract_plate_text(frame, plate_box):
    x1, y1, x2, y2 = map(int, plate_box)
    plate_crop = frame[y1:y2, x1:x2]
    result = ocr.ocr(plate_crop, cls=True)
    if result and result[0]:
        text = result[0][0][1][0]
        return text.replace(" ", "").upper()  # e.g. "MH12AB1234"
    return None

def check_dedup(plate, camera_id):
    key = f"seen:{plate}:{camera_id}"
    return r.exists(key)

def mark_seen(plate, camera_id, violation_id):
    key = f"seen:{plate}:{camera_id}"
    r.setex(key, 3600, violation_id)  # TTL = 1 hour

def process_frame(frame):
    # Step 1: Detect bikes + riders
    rider_results = rider_model(frame, conf=0.5, verbose=False)[0]
    
    bikes   = [b for b in rider_results.boxes if int(b.cls) == 0]  # motorcycle
    persons = [p for p in rider_results.boxes if int(p.cls) == 1]  # person

    for bike in bikes:
        bx1, by1, bx2, by2 = map(int, bike.xyxy[0])
        
        # Count riders ON this bike (person bbox overlaps bike bbox)
        riders_on_bike = []
        for person in persons:
            px1, py1, px2, py2 = map(int, person.xyxy[0])
            overlap_x = max(0, min(bx2, px2) - max(bx1, px1))
            overlap_y = max(0, min(by2, py2) - max(by1, py1))
            if overlap_x > 20 and overlap_y > 20:
                riders_on_bike.append(person)
        
        rider_count = len(riders_on_bike)
        
        # Step 2: Helmet detection on bike region
        bike_crop = frame[by1:by2, bx1:bx2]
        helmet_results = helmet_model(bike_crop, conf=0.5, verbose=False)[0]
        
        no_helmet_detected = any(
            int(h.cls) == 1 for h in helmet_results.boxes
        )
        helmet_conf = float(max(
            (h.conf for h in helmet_results.boxes), default=0
        ))
        
        # Step 3: Determine violation
        violation_type = None
        if rider_count >= 3 and no_helmet_detected:
            violation_type = "both"
        elif rider_count >= 3:
            violation_type = "triple_riding"
        elif no_helmet_detected:
            violation_type = "no_helmet"
        
        if not violation_type:
            continue
        
        # Step 4: Only proceed if confidence sufficient
        if helmet_conf < CONF_REVIEW:
            continue
        
        status = "auto_challan" if helmet_conf >= CONF_AUTO else "manual_review"
        
        # Step 5: OCR number plate
        # (Assume plate detection model or crop bottom of bike)
        plate_text = "UNKNOWN"  # Replace with actual plate detection
        
        if plate_text == "UNKNOWN":
            continue
        
        # Step 6: Dedup check
        if check_dedup(plate_text, CAMERA_ID):
            print(f"Skipping duplicate: {plate_text}")
            continue
        
        # Step 7: Send to backend
        violation_id = str(uuid.uuid4())
        payload = {
            "id":              violation_id,
            "plate_number":    plate_text,
            "violation_type":  violation_type,
            "camera_id":       CAMERA_ID,
            "captured_at":     datetime.now().isoformat(),
            "confidence":      helmet_conf,
            "rider_count":     rider_count,
            "helmet_detected": not no_helmet_detected,
            "status":          status
        }
        
        try:
            response = requests.post(BACKEND_URL, json=payload, timeout=3)
            if response.status_code == 201:
                mark_seen(plate_text, CAMERA_ID, violation_id)
                print(f"Violation logged: {violation_type} | Plate: {plate_text} | Conf: {helmet_conf:.2f}")
        except requests.exceptions.RequestException as e:
            print(f"Backend error: {e}")

def main():
    cap = cv2.VideoCapture(0)   # 0 = default webcam, or RTSP URL for IP camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    frame_skip = 3   # Process every 3rd frame — saves CPU
    frame_count = 0
    
    print("Detection started. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue
        
        process_frame(frame)
        
        cv2.imshow("Traffic Violation Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

---

## 9. FASTAPI BACKEND (Core Endpoints)

```python
# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime, timedelta
import asyncpg
import redis.asyncio as aioredis
import cloudinary.uploader
import uuid

app  = FastAPI(title="Traffic Violation System")
pool = None   # asyncpg connection pool
rc   = None   # redis client

CHALLAN_AMOUNTS = {
    "no_helmet":    500,
    "triple_riding": 1000,
    "both":         1500
}

class ViolationIn(BaseModel):
    id:              str
    plate_number:    str
    violation_type:  str
    camera_id:       str
    captured_at:     str
    confidence:      float
    rider_count:     int
    helmet_detected: bool
    status:          str

@app.post("/api/violations", status_code=201)
async def create_violation(v: ViolationIn):
    # 1. Dedup check (Redis)
    key = f"seen:{v.plate_number}:{v.camera_id}"
    if await rc.exists(key):
        return {"message": "duplicate skipped"}
    
    # 2. Save violation to PostgreSQL
    await pool.execute("""
        INSERT INTO violations 
        (id, plate_number, violation_type, camera_id, captured_at, 
         confidence, rider_count, helmet_detected, status)
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
    """, v.id, v.plate_number, v.violation_type, v.camera_id,
        datetime.fromisoformat(v.captured_at), v.confidence,
        v.rider_count, v.helmet_detected, v.status)
    
    # 3. Mark in Redis (1 hour TTL)
    await rc.setex(key, 3600, v.id)
    
    # 4. Auto challan if confidence high
    if v.status == "auto_challan":
        challan_id = str(uuid.uuid4())
        amount     = CHALLAN_AMOUNTS.get(v.violation_type, 500)
        due_date   = (datetime.now() + timedelta(days=30)).date()
        
        await pool.execute("""
            INSERT INTO challans
            (id, violation_id, plate_number, amount, violation_type, due_date)
            VALUES ($1,$2,$3,$4,$5,$6)
        """, challan_id, v.id, v.plate_number, amount, v.violation_type, due_date)
        
        # Send SMS (async, non-blocking)
        await send_sms_notification(v.plate_number, amount, challan_id)
    
    return {"message": "violation created", "id": v.id}

@app.get("/api/challans")
async def list_challans(page: int = 1, limit: int = 50):
    offset = (page - 1) * limit
    rows = await pool.fetch("""
        SELECT c.*, v.image_path, v.captured_at
        FROM challans c
        JOIN violations v ON c.violation_id = v.id
        ORDER BY c.issued_at DESC
        LIMIT $1 OFFSET $2
    """, limit, offset)
    return [dict(r) for r in rows]

@app.get("/api/review-queue")
async def get_review_queue():
    rows = await pool.fetch("""
        SELECT * FROM violations
        WHERE status = 'manual_review'
        ORDER BY captured_at DESC
        LIMIT 100
    """)
    return [dict(r) for r in rows]

@app.put("/api/violations/{violation_id}/review")
async def review_violation(violation_id: str, decision: str):
    if decision not in ["approve", "reject"]:
        raise HTTPException(400, "decision must be approve or reject")
    
    new_status = "challan_issued" if decision == "approve" else "skipped"
    await pool.execute(
        "UPDATE violations SET status=$1 WHERE id=$2",
        new_status, violation_id
    )
    return {"message": f"violation {decision}d"}
```

---

## 10. PROJECT FOLDER STRUCTURE

```
traffic-violation-system/
│
├── edge/                          # Runs on Raspberry Pi / camera device
│   ├── detection_engine.py        # Main detection loop
│   ├── models/
│   │   ├── helmet_v1.onnx         # Trained helmet model
│   │   └── rider_counter.onnx     # Rider count model
│   ├── config.py                  # Camera URL, backend URL, thresholds
│   └── requirements_edge.txt
│
├── backend/                       # FastAPI server
│   ├── main.py                    # FastAPI app + endpoints
│   ├── database.py                # asyncpg pool setup
│   ├── models.py                  # Pydantic schemas
│   ├── sms.py                     # Twilio SMS helper
│   ├── challan_pdf.py             # Generate PDF challan
│   └── requirements.txt
│
├── frontend/                      # React dashboard
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx      # Live stats + recent violations
│   │   │   ├── Challans.jsx       # Challan list + search
│   │   │   ├── ReviewQueue.jsx    # Manual review interface
│   │   │   └── Analytics.jsx      # Heatmaps, charts
│   │   └── components/
│   └── package.json
│
├── training/                      # Google Colab notebooks
│   ├── train_helmet_model.ipynb   # Helmet detection training
│   ├── train_rider_model.ipynb    # Rider count training
│   └── test_paddleocr.ipynb       # OCR accuracy testing
│
└── database/
    └── schema.sql                  # PostgreSQL schema (from Section 2.2)
```

---

## 11. ACCURACY TARGETS & BENCHMARKS

| Metric | Minimum | Good | Excellent |
|---|---|---|---|
| Helmet mAP50 | 0.80 | 0.87 | 0.93+ |
| Rider Count Accuracy | 85% | 92% | 97% |
| Number Plate OCR Accuracy | 80% | 90% | 95%+ |
| End-to-End False Positive Rate | <15% | <8% | <3% |

**Factors that kill accuracy (avoid these):**
- Poor camera placement (angle > 30° from plate face)
- Night without IR lighting
- Dirty/covered plates
- Low contrast (overcast weather, fog)

**Fix:**
- Camera height: 3-5 meters
- Angle: 15-20° downward
- IR spotlight ₹200 for night
- Confidence threshold > 0.85 for auto-challan (manual review for 0.65-0.85)

---

## 12. WEEK-BY-WEEK IMPLEMENTATION PLAN

### Week 1 — Data + Training Setup
- [ ] Roboflow free account banao
- [ ] Helmet Detection India dataset download karo (5k+ images)
- [ ] Google Colab notebook setup karo (Runtime → T4 GPU)
- [ ] YOLOv8n train karo (50 epochs, 2-3 hrs)
- [ ] mAP50 check karo — target > 0.85
- [ ] ONNX export karo

### Week 2 — OCR + Edge Device
- [ ] PaddleOCR test karo — apni gaadi ki plate photo se
- [ ] `detection_engine.py` banao (Section 8 ka code)
- [ ] Webcam se live test karo laptop pe
- [ ] False positives check karo, threshold tune karo

### Week 3 — Backend
- [ ] Supabase account + PostgreSQL schema create karo (Section 2.2 ka SQL)
- [ ] Upstash Redis free instance create karo
- [ ] FastAPI backend deploy karo Railway.app pe
- [ ] Violation API test karo (Postman / curl)
- [ ] Dedup logic test karo (same plate twice within 1 hour)

### Week 4 — Storage + Notifications
- [ ] Cloudinary account + image upload integrate karo
- [ ] Twilio SMS setup karo (free $15 credit)
- [ ] End-to-end test: Camera → Detection → API → DB → SMS

### Week 5 — Dashboard
- [ ] React project create karo (Vite + Tailwind)
- [ ] Challan list page banao
- [ ] Manual review queue banao (approve/reject buttons)
- [ ] Vercel pe deploy karo

### Week 6 — Real Camera Testing
- [ ] Raspberry Pi 4 setup karo (Raspbian OS)
- [ ] Models Pi pe copy karo + inference test karo
- [ ] IP camera ya USB webcam lagao
- [ ] Real traffic pe test karo

### Week 7-8 — Optimization
- [ ] False positive rate measure karo
- [ ] Training data badhao (apne collected footage se)
- [ ] Model fine-tune karo (accuracy 88% → 93%+)
- [ ] Multi-camera support add karo

---

## 13. KNOWN CHALLENGES & SOLUTIONS

| Challenge | Cause | Solution |
|---|---|---|
| Indian plate variability (MH12 AB 1234 vs MH12AB1234) | No standard format | Regex normalize: remove spaces, uppercase |
| Night detection fails | No IR lighting | Add ₹200 IR LED spotlight, train model on dark images |
| Plate covered / dirty | Physical occlusion | Flag as manual_review if OCR confidence < 0.7 |
| Triple riding hard to detect from side angle | Occlusion of back rider | Camera placement at slight angle from front-rear direction |
| High CPU on Raspberry Pi | Model too heavy | Use YOLOv8n (not s/m/l), process every 3rd frame |
| Same vehicle gets multiple challans | No dedup | Redis TTL-based dedup (already in design) |
| RTO database not available | Government data locked | Manual CSV import of local vehicle data, OR build lookup for owner manually |

---

## 14. COST SUMMARY

### Hardware (One-time)
| Item | Cost |
|---|---|
| Raspberry Pi 4 (4GB) | ₹5,000 |
| USB 1080p Camera | ₹800 |
| 32GB SD Card | ₹400 |
| IR LED Spotlight (night) | ₹200 |
| Weatherproof enclosure | ₹500 |
| **Total per camera point** | **~₹7,000** |

### Software / Cloud (Monthly)
| Service | Cost |
|---|---|
| Supabase (500MB free) | ₹0 |
| Upstash Redis (10k req/day) | ₹0 |
| Cloudinary (25GB) | ₹0 |
| Railway.app (500 hrs) | ₹0 |
| Vercel (frontend) | ₹0 |
| Twilio SMS (after free credit) | ~₹1.5/SMS |
| **Total/month (MVP)** | **₹0** |

---

*Document generated: Traffic Violation Detection System — Complete Research & Implementation Plan*
*Version 1.0*
