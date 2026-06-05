import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
import os
import cv2
import numpy as np
from PIL import Image

# --- paths ---
# Adjust these if your folder structure is slightly different
IMAGE_DIR = r"Indian_Number_Plates\Sample_Images"
ANNOT_DIR = r"Annotations\Annotations"

def get_model_instance_segmentation(num_classes):
    # Load model structure
    model = fasterrcnn_resnet50_fpn(weights=None)
    
    # Load weights from local file to avoid download errors
    weights_path = "fasterrcnn_resnet50_fpn_coco.pth"
    if os.path.exists(weights_path):
        print(f"Loading weights from local file: {weights_path}")
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)
    else:
        print("Local weights not found! Trying download (might fail if offline)...")
        model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    
    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

class PlateDataset(Dataset):
    def __init__(self, output_dir, xml_dir):
        self.output_dir = output_dir
        self.xml_dir = xml_dir
        
        # Sort to ensure alignment
        self.imgs = list(sorted(os.listdir(output_dir)))
        self.xmls = list(sorted(os.listdir(xml_dir)))
        
        # Simple check to filter out non-matching files could be added here
        # For now assuming 1:1 match based on filenames check previously

    def __getitem__(self, idx):
        # Load Image
        img_name = self.imgs[idx]
        img_path = os.path.join(self.output_dir, img_name)
        
        # Read with OpenCV (BGR) then convert to RGB
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to Tensor (0-1 float) using ToTensor manually or torchvision
        # Moving (H, W, C) -> (C, H, W) and dividing by 255
        img = torch.as_tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        # Load XML Annotation
        # Assuming xml name matches image name (excluding extension)
        base_name = os.path.splitext(img_name)[0]
        xml_path = os.path.join(self.xml_dir, base_name + ".xml")
        
        boxes = []
        labels = []
        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        for object in root.findall('object'):
            label = object.find('name').text
            if label == 'number_plate':  # We only care about this class
                bndbox = object.find('bndbox')
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(1) # Class 1 = Plate, 0 = Background
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        
        return img, target

    def __len__(self):
        return len(self.imgs)

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    print("--- Starting Training Setup ---")
    
    # Check paths
    if not os.path.exists(IMAGE_DIR) or not os.path.exists(ANNOT_DIR):
        print("Error: Dataset paths not found!")
        print(f"Checked: {IMAGE_DIR} and {ANNOT_DIR}")
        return

    # Use CUDA if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Device: {device}")

    # Dataset & DataLoader
    dataset = PlateDataset(IMAGE_DIR, ANNOT_DIR)
    
    # Split into train/test (optional, using all for train on small set for demo)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    
    print(f"Total Images: {len(dataset)}")

    # Model Setup (2 classes: Background, Plate)
    num_classes = 2
    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    # Training Loop
    num_epochs = 5 # 27 images is small, so 5-10 epochs is quick
    
    print("Training started...")
    for epoch in range(num_epochs):
        model.train()
        i = 0
        epoch_loss = 0
        
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass (Calculate Loss)
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass (Update Weights)
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
            i += 1
            if i % 2 == 0:
                print(f"Epoch: {epoch+1}, Step: {i}, Loss: {losses.item():.4f}")
                
        print(f"End of Epoch {epoch+1}, Avg Loss: {epoch_loss/i:.4f}")

    # Save Model
    save_path = "custom_plate_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
