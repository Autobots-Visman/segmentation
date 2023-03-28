# Description:
#     A script that performs semantic segmentation

from ultralytics import YOLO
import numpy as np
import cv2

# Configure
debug = True

# YOLOv8 segmentation models use the -seg suffix , i.e. yolov8n-seg.pt and are pretrained on COCO.
model = YOLO('yolov8n-seg.pt')

# Use CAMERA_ID if it's open. Otherwise find another camera.
CAMERA_ID = 2
for i in range(5):
    vid_stream = cv2.VideoCapture(CAMERA_ID)
    if vid_stream.isOpened():
        break
    CAMERA_ID = i

# Data for tests
# bus.jpg is a picture of two men crossing a small street with a blue bus behind them in europe.
image = "https://ultralytics.com/images/bus.jpg"

def label(image):
    """Use a computer vision model to semantically label an image. """
    results = model(image)  # predict on an image

    return results[0]

def get_frame(vid_stream):
    """Get the next image frame from a video stream. """
    captured, frame = vid_stream.read()

    if not vid_stream.isOpened():
        print("Error: Could not capture video.")
        exit(1)
  
    if captured:
        return frame
    return None

def track():
    """Create a new kalman filter tracker for tracking one object. """
    # Assume the state and measurement paramters have a dimension of 2. There isn't a control vector. 
    tracker = cv2.KalmanFilter(4, 2)
    tracker.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    tracker.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    tracker.processNoiseCov = np.array([[1e-3, 0, 0, 0], [0, 1e-3, 0, 0], [0, 0, 5e-3, 0], [0, 0, 0, 5e-3]], np.float32)
    return tracker

def update_tracker(tracker, center_x, center_y):
    """Update the tracker with a new measurement and return the estimated object location."""
    measurement = np.array([[np.float32(center_x)], [np.float32(center_y)]])
    tracker.correct(measurement)
    prediction = tracker.predict()
    return prediction

def get_center(xyxy):
    bbox_center_x = (xyxy[2] + xyxy[0]) / 2
    bbox_center_y = (xyxy[3] + xyxy[1]) / 2
    return bbox_center_x, bbox_center_y

def get_tracker(trackers, object_type, center_x, center_y):
    """Return the coordinates of the nearest object of the same object type."""
    if object_type in trackers:
        min_distance = float('inf')
        nearest_object = None
        for oid, item in trackers[object_type].items():
            distance = np.sqrt((center_x - item[1].statePost[0][0])**2 + (center_y - item[1].statePost[1][0])**2)
            if distance < min_distance:
                min_distance = distance
                nearest_object = item
        return nearest_object
    else:
        oid = str(object_type) + "1"
        item = (oid, track())
        trackers[object_type] = {oid : item}
        return item

try:
    trackers = {}
    while True:
        frame   = get_frame(vid_stream)
        labels  = label(frame)
        boxes = labels.boxes

        # Use a Kalman Filter to predict the location of each objectz
        for i in range(len(boxes)):
            object_type = model.names[int(boxes[i].cls)]
            center_x, center_y = get_center(boxes.xyxy[i])
            id, tracker = get_tracker(trackers, object_type, center_x, center_y)
            location    = update_tracker(tracker, center_x, center_y)
            print("Object:", id, "at:", location)

            # Display results
            if debug:
                # Display Predictions inside bounding boxes
                cv2.rectangle(frame, (int(boxes.xyxy[i][0]), int(boxes.xyxy[i][1])), (int(boxes.xyxy[i][2]), int(boxes.xyxy[i][3])), (0, 255, 0), 2)
                cv2.putText(frame, id, (int(boxes.xyxy[i][0]), int(boxes.xyxy[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.drawMarker(frame, (int(location[0]), int(location[1])), (0, 0, 255), cv2.MARKER_CROSS, 20)
                cv2.imshow("result.png", frame)
                cv2.waitKey(delay=1)

finally:
    # After the loop release the cap object
    vid_stream.release()
    # Destroy all the windows
    cv2.destroyAllWindows()