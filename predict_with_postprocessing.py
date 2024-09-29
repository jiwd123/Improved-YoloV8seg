from ultralytics import YOLO
import glob
import os
import cv2
import sys
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
np.set_printoptions(threshold=sys.maxsize)
def find_max_region(mask_sel):
    contours,hierarchy = cv2.findContours(mask_sel,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
 
    #找到最大区域并填充 
    area = []
 
    for j in range(len(contours)):
        area.append(cv2.contourArea(contours[j]))
 
    max_idx = np.argmax(area)
 
    max_area = cv2.contourArea(contours[max_idx])
 
    for k in range(len(contours)):
    
        if k != max_idx:
            cv2.fillPoly(mask_sel, [contours[k]], 0)
    return mask_sel
# Load a model
model = YOLO(r'best.pt')  # load an official model
 
# Predict with the model
imgpath = r'/hy-tmp/test'
imgs = glob.glob(os.path.join(imgpath,'*.jpg'))
for img in imgs:
    results = model.predict(img, imgsz = 2048, conf=0.05,iou=0.07,show_boxes=False,show_conf=False,show_labels=False)
    for result in results:
        masks = result.masks.cpu()
        #print(masks.data.numpy())
        i=0
        for i in range(len(masks.data)):
            masknew = find_max_region(masks.data[i].numpy().astype(np.uint8))
            masks.data[i] =torch.tensor(masknew)
            i=i+1
            
        result.masks = masks
        result.save(boxes = False)



 