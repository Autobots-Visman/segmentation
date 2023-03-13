# Description:
#     A script that performs semantic segmentation
# Build:
#     docker compose build
# Run:
#     docker compose run app python3 src/segmentation/ros/object_detection/src/object_detection/segmentation.py

from ultralytics import YOLO
import cv2

# Configure
debug = True

# Load and initialize model and video
# https://docs.ultralytics.com/quickstart/
# https://docs.ultralytics.com/tasks/segment/
# YOLOv8 segmentation models use the -seg suffix , i.e. yolov8n-seg.pt and are pretrained on COCO.
model = YOLO('yolov8n-seg.pt')
# 0 = open default camera
vid_stream = cv2.VideoCapture(0)    

# Data for tests
# bus.jpg is a picture of two men crossing a small street with a blue bus behind them in europe.
image = "https://ultralytics.com/images/bus.jpg"

def label(image):
    """Use a computer vision model to semantically label an image. """
    results = model(image)  # predict on an image

    # Display results
    if debug:
        res_plotted = results[0].plot()
        cv2.imshow("result.png", res_plotted)
        cv2.waitKey(delay=1)

    return results

def get_frame(vid_stream):
    """Get the next image frame from a video stream. """
    captured, frame = vid_stream.read()

    if not vid_stream.isOpened():
        print("Error: Could not capture video.")
  
    if captured:
        return frame
    return None

try:
    count = 0
    while count < 200:
        frame   = get_frame(vid_stream)
        labels  = label(frame)

        count   += 1
finally:
    # After the loop release the cap object
    vid_stream.release()
    # Destroy all the windows
    cv2.destroyAllWindows()