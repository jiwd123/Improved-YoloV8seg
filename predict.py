from ultralytics import YOLO
import glob
import os
# Load a model
model = YOLO(r'best.pt')  # load an official model
 
# Predict with the model
imgpath = r'datasets\yolo\images\test'
imgs = glob.glob(os.path.join(imgpath,'*.jpg'))
for img in imgs:
    model.predict(img, save=True,imgsz = 512, conf=0.4, iou=0.01)
 