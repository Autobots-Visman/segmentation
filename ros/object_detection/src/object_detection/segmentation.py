# Description:
#     A script that performs semantic segmentation

from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
import numpy as np
import cv2

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

def get_center(xyxy):
    """Convert a bounding box corner from the array boxes.xyxy to a center xy coordinate of the box. """
    bbox_center_x = (xyxy[2] + xyxy[0]) / 2
    bbox_center_y = (xyxy[3] + xyxy[1]) / 2
    return bbox_center_x, bbox_center_y

def assign_trackers(trackers, boxes):
    """Assign all detected bounding box centers to existing trackers with the least amount of error. """

    # Calculate the center of all boxes of a known class
    unused_boxes = set(range(len(boxes)))
    all_assignments = set()
    for object_class in trackers.keys():
            known_centers = []
            for i in range(len(boxes)):
                if model.names[int(boxes[i].cls)] == object_class:
                    unused_boxes.discard(i)
                    bbox_center_x, bbox_center_y = get_center(boxes.xyxy[i])
                    known_centers.append((bbox_center_x, bbox_center_y))

            # Create a cost matrix with all trackers and centers of the object_class
            cost_matrix = np.zeros((len(trackers[object_class]), len(known_centers)))
            for i, (id, kalman_filter) in enumerate(trackers[object_class].items()):
                for j, center in enumerate(known_centers):
                    location = kalman_filter.statePost[:2].reshape(-1)
                    cost_matrix[i][j] = np.linalg.norm(location - center)

            # Use the hungarian algorithm to assign each bounding box center measurement to an existing kalman filter
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Convert the output to a list of tuples called assignments with the format (tracker_id, (center_x, center_y))
            assignments = []
            for i in range(len(row_ind)):
                tracker_id = list(trackers[object_class].keys())[row_ind[i]]
                center = known_centers[col_ind[i]]
                assignments.append((tracker_id, center))
                all_assignments.add(tracker_id)

            # Update the trackers with the current center coordinate measurements
            for tracker_id, center in assignments:
                current_kf = trackers[object_class][tracker_id]
                update_tracker(current_kf, center[0], center[1])

    # If any tracker ids in trackers are unused, then update their kalman filter to lower the probability of their last known location. 
    for object_class in trackers.keys():
        for id in list(trackers[object_class].keys()):
            if id not in all_assignments:
                current_kf = trackers[object_class][id]
                current_kf.errorCovPost *= 1.1

    return unused_boxes

try:
    # Example: {"person": {"person1": kalman_filter, "person2": kalman_filter}, "chair": {"chair1": kalman_filter} }
    trackers = {}
    while True:
        frame   = get_frame(vid_stream)
        labels  = label(frame)
        boxes = labels.boxes

        # Use a Kalman Filter to predict the location of each object
        unused_boxes = assign_trackers(trackers, boxes)

        # Create trackers for each boxes index in unused_boxes
        for unused in list(unused_boxes):
            object_class = model.names[int(boxes[unused].cls)]
            if not object_class in trackers:
                trackers[object_class] = {}
            count = len(trackers[object_class])
            id = object_class + str(count + 1)
            trackers[object_class][id] = track()
            center_x, center_y = get_center(boxes.xyxy[unused])
            update_tracker(trackers[object_class][id], center_x, center_y)

        # Use the estimated locations of all visible objects
        for object_class in trackers.keys():
            for id in trackers[object_class].keys():
                kalman_filter = trackers[object_class][id]
                location = np.round(np.array(kalman_filter.statePost[:2].reshape(-1)))

                # Display results
                print("Object:", id, "\tat:", (location))
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