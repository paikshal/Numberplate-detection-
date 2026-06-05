import cv2
from ultralytics import YOLO
import easyocr
import numpy as np
import os
import time
from datetime import datetime

# --- Setup ---
print("Initializing YOLOv8 & OCR System...")

# 1. Load YOLOv8 Model (Nano for speed, Pre-trained on COCO)
# It will auto-download 'yolov8n.pt' on first run
model = YOLO('yolov8n.pt') 

# 2. Check Device
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using Device: {device}")
model.to(device)

# 3. Load Haarcascade for localized plate detection
cascade_filename = "haarcascade_russian_plate_number.xml"
if not os.path.exists(cascade_filename):
    cascade_filename = os.path.join(cv2.data.haarcascades, "haarcascade_russian_plate_number.xml")
plate_cascade = cv2.CascadeClassifier(cascade_filename)

# 4. Initialize OCR
print("Initializing OCR Engine...")
reader = easyocr.Reader(['en'], gpu=(device == 'cuda'))

# 5. Output Directory
OUTPUT_DIR = 'cs'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# COCO Class IDs for vehicles
VEHICLE_CLASSES = [2, 3, 5, 7] # 2=car, 3=motorcycle, 5=bus, 7=truck

def save_result(img, plate_text):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    img_name  = f"plate_{timestamp}.jpg"
    img_path = os.path.join(OUTPUT_DIR, img_name)
    cv2.imwrite(img_path, img)

    txt_name = f"plate_{timestamp}.txt"
    txt_path = os.path.join(OUTPUT_DIR, txt_name)
    with open(txt_path, 'w') as f:
        f.write(plate_text)
        
    print(f"[SAVED] Image: {img_path}")
    print(f"[SAVED] Number: {plate_text}")

def preprocess_plate(roi):
    # Enhance image for better OCR
    scale_percent = 150 
    w = int(roi.shape[1] * scale_percent / 100)
    h = int(roi.shape[0] * scale_percent / 100)
    resized = cv2.resize(roi, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # Sharpen
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(resized, -1, kernel)
    
    # Grayscale
    gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
    return gray

def ocr_process(roi, full_img, x_offset, y_offset, enhance=True):
    ocr_input = preprocess_plate(roi) if enhance else roi
    
    results = reader.readtext(ocr_input, detail=1)
    
    if results:
        # Sort top->bottom, left->right for correct reading order
        results.sort(key=lambda r: (r[0][0][1], r[0][0][0]))
        text_list = [res[1] for res in results]
        text = " ".join(text_list)
        detected_text = text.upper()
        
        # Filter noise (length > 3 and alphanumeric check)
        # removing spaces for alnum check to allow "MH 12"
        clean_text = detected_text.replace(" ", "")
        if len(clean_text) > 3 and clean_text.isalnum():
            print(f"FOUND: {detected_text}")
            cv2.putText(full_img, detected_text, (x_offset, y_offset-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            save_result(full_img, detected_text)
            return True
    return False

def detect_pipeline(img, display=True):
    # Step 1: Detect Vehicles using YOLOv8
    results = model(img, verbose=False) # Run inference
    
    vehicle_found = False
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            if cls_id in VEHICLE_CLASSES and conf > 0.5:
                vehicle_found = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Draw Vehicle Box
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img, model.names[cls_id], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
                
                # Crop Vehicle
                vehicle_roi = img[y1:y2, x1:x2]
                if vehicle_roi.size == 0: continue

                # Step 2: Detect Plate inside Vehicle ROI (Haarcascade)
                gray_roi = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2GRAY)
                plates = plate_cascade.detectMultiScale(gray_roi, 1.1, 4)
                
                plate_found_in_vehicle = False
                for (px, py, pw, ph) in plates:
                    # Draw Plate Box
                    cv2.rectangle(img, (x1+px, y1+py), (x1+px+pw, y1+py+ph), (0, 255, 0), 2)
                    plate_crop = vehicle_roi[py:py+ph, px:px+pw]
                    
                    if ocr_process(plate_crop, img, x1+px, y1+py):
                        plate_found_in_vehicle = True

                # Step 3: Fallback - Scan whole vehicle ROI if haarcascade fails
                if not plate_found_in_vehicle:
                    ocr_process(vehicle_roi, img, x1, y1)

    if not vehicle_found:
        print("No vehicle detected. Scanning full image...")
        ocr_process(img, img, 0, 0)

    if display:
        disp_img = img.copy()
        if disp_img.shape[1] > 1000:
            disp_img = cv2.resize(disp_img, (1000, int(1000*disp_img.shape[0]/disp_img.shape[1])))
        cv2.imshow("YOLOv8 Plate Detection", disp_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def mode_image():
    while True:
        path = input("\nEnter image path (or 'q'): ").strip()
        if path.lower() == 'q': break
        if path.startswith('"') and path.endswith('"'): path = path[1:-1]
        
        if not os.path.exists(path):
            print("File not found.")
            continue
            
        img = cv2.imread(path)
        if img is None:
            print("Cannot read image.")
            continue
            
        detect_pipeline(img, display=True)

def mode_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Camera Error.")
        return
        
    print("Camera Mode. Press 's' to scan, 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Real-time YOLO detection on resize for speed in display loop? 
        # For now, let's keep it simple: Show feed, detect on press 's'
        cv2.imshow("Feed", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            print("Capturing frame...")
            detect_pipeline(frame.copy(), display=True)
            print("Done.")
        elif key == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("--- YOLOv8 Number Plate System ---")
    print("1. Image Mode")
    print("2. Camera Mode")
    choice = input("Choice: ").strip()
    if choice == '1': mode_image()
    elif choice == '2': mode_camera()
