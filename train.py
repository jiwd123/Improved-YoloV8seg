from ultralytics import YOLO
 
if __name__ == '__main__':
    
    model = YOLO('improved_yolov8n-seg.yaml')  # load a pretrained model (recommended for training)
    # Train the model
    model.train(epochs=100,data='mysegdata.yaml',imgsz = 2048)