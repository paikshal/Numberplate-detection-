# Software Requirements Specification (SRS)
## Number Plate Detection & Recognition System

**Document Version:** 1.0
**Date:** 2026-02-06
**Project Root:** c:\Users\paiks\OneDrive\Desktop\colab\number_platedection

---

## 1. Purpose
Build a system that detects vehicle number plates from either an uploaded image or a live camera feed, recognizes the plate text, and stores outputs in a local `cs` folder. The system must support both Indian plate formats and general international alphanumeric plates, and run on CPU and GPU.

## 2. Scope
**In Scope**
- Plate detection from images and live camera frames.
- Plate text recognition (OCR) for Indian + general plate formats.
- Results storage in `cs` folder with images and logs.
- CPU and GPU inference support.

**Out of Scope**
- Vehicle make/model classification.
- Owner database lookup.
- Challan or enforcement workflows.
- Cloud storage (local only).

## 3. Definitions
- **Detection:** Locating the plate region in an image/frame.
- **Recognition (OCR):** Extracting alphanumeric text from the detected plate region.
- **Frame:** A single image from a video/camera stream.
- **Confidence:** Model-provided score for detection/recognition.

## 4. Assumptions
- Input images are at least 720p for reliable OCR.
- Plates are reasonably visible (not fully occluded).
- Camera access is available on the target machine.

## 5. System Overview
Pipeline: **Input → Pre-process → Detect Plate → Crop/Rectify → OCR → Post-process → Save**

The same pipeline is used for image mode and camera mode. Camera mode adds temporal voting to stabilize OCR results across frames.

## 6. Functional Requirements
FR-1: System shall accept image files in `.jpg`, `.jpeg`, `.png`.
FR-2: System shall capture frames from a live camera stream.
FR-3: System shall detect one or multiple number plates in each image/frame.
FR-4: System shall crop detected plate regions for OCR.
FR-5: System shall recognize plate text from cropped plate images.
FR-6: System shall support Indian plate formats and general alphanumeric formats.
FR-7: System shall store outputs in `cs` folder:
- Original image/frame
- Cropped plate image(s)
- Logs with recognized text and metadata
FR-8: System shall log a result even when no plate is detected (status = `NO_PLATE`).
FR-9: System shall allow selection of CPU or GPU inference mode.

## 7. Non-Functional Requirements
NFR-1: Camera mode target throughput ≥ 10 FPS on GPU and ≥ 5 FPS on CPU (typical laptop).
NFR-2: OCR accuracy ≥ 90% on clear, front-facing plates.
NFR-3: Detection accuracy (mAP) ≥ 0.85 on validation set.
NFR-4: System shall operate offline without external API calls.
NFR-5: Each frame should be processed in < 400 ms on CPU or < 150 ms on GPU.

## 8. Data Storage & Folder Structure
All outputs are stored under `cs`.

- `cs/frames/`   : original frames or input images
- `cs/plates/`   : cropped plate images
- `cs/logs/`     : logs in CSV/JSON

**Log schema (CSV/JSON):**
- `timestamp`
- `source` (image|camera)
- `frame_id`
- `plate_text`
- `confidence`
- `status` (OK|NO_PLATE|LOW_CONF)
- `frame_path`
- `plate_path`

## 9. Model Requirements
**Detection Model**
- Recommended: YOLOv5/YOLOv8 fine-tuned on number plate dataset.
- Input: Full image/frame
- Output: Bounding boxes + confidence

**OCR Model**
- Recommended: CRNN/LPRNet or OCR library (e.g., EasyOCR/Tesseract).
- Input: Cropped plate image
- Output: Plate text + confidence

## 10. Plate Format Handling
**Indian Plates**
- Regex template examples: `[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}`

**General Plates**
- Alphanumeric patterns, length 4–12

Post-processing shall clean OCR output (remove spaces, normalize case) and validate against format rules.

## 11. Camera Mode Specific Logic
- Continuous capture loop
- Per-frame detection + OCR
- Temporal voting over last N frames (N=5–10) to stabilize text
- Save only if confidence passes threshold or stabilized consensus is achieved

## 12. Error Handling
- If no plate detected: log `NO_PLATE`
- If OCR confidence below threshold: log `LOW_CONF`
- If camera unavailable: retry and log error

## 13. Security & Privacy
- All data stored locally
- No external transmission of plate data

## 14. Acceptance Criteria
AC-1: Image input produces correct plate detection + OCR and stores output in `cs`.
AC-2: Camera input processes frames continuously and stores results.
AC-3: Logs contain timestamp, text, confidence, and file paths.
AC-4: System works on both CPU and GPU modes.
AC-5: Handles missing plates without crashing.

## 15. Suggested Enhancements (Optional)
- Multi-plate tracking to avoid duplicate logs
- Perspective correction for angled plates
- UI to select input source and view results
- Batch processing for folders of images

---

**End of Document**
