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

# Use CAMERA_ID if it's open. Otherwise find another camera.
CAMERA_ID = 2
for i in range(5):
    vid_stream = cv2.VideoCapture(CAMERA_ID)
    if vid_stream.isOpened():
        break
    CAMERA_ID = i

# Tracker
# Kalman filters were chosen as the tracker for these reasons:
#     1. They can handle occlusion
#     2. Most trackers function by searching for similar image patches. That's not
#         needed here because YOLO already does that.
#     3. Kalman filters compensate for camera motion (can be estimated by opencv)


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
        exit(1)
  
    if captured:
        return frame
    return None

try:
    while True:
        frame   = get_frame(vid_stream)
        labels  = label(frame)

finally:
    # After the loop release the cap object
    vid_stream.release()
    # Destroy all the windows
    cv2.destroyAllWindows()