# Description:
#     A script that performs semantic segmentation
# Build:
#     docker compose build
# Run:
#     docker compose run app python3 src/segmentation/ros/object_detection/src/object_detection/segmentation.py

from ultralytics import YOLO
import cv2 as cv

# Load model
# https://docs.ultralytics.com/quickstart/
# https://docs.ultralytics.com/tasks/segment/
# YOLOv8 segmentation models use the -seg suffix, i.e. yolov8n-seg.pt and are pretrained on COCO.
model = YOLO('yolov8n-seg.pt')

# Predict with the model
# bus.jpg is a picture of two men crossing a small street with a blue bus behind them in europe.
image = "https://ultralytics.com/images/bus.jpg"
results = model(image)  # predict on an image

print("Results:", results)