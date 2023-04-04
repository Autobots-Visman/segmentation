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

def track(center_x, center_y):
    """Create a new kalman filter tracker for tracking one object. """
    # Assume the state and measurement paramters have a dimension of 2. There isn't a control vector. 
    tracker = cv2.KalmanFilter(4, 2)
    # Which dimensions should be measured?
    tracker.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    # What dimensions need to be multiplied together to represent change?
    tracker.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    # How much should each measurement be trusted (bigger = more trust)?
    tracker.processNoiseCov = np.array([[5e-3, 0, 0, 0], [0, 5e-3, 0, 0], [0, 0, 5e-3, 0], [0, 0, 0, 5e-3]], np.float32)
    # What is the current state?
    tracker.statePre = np.array(
        [[center_x], [center_y], [0], [0]], np.float32
    )
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

            # Update the trackers for this class
            for i in range(len(row_ind)):
                tracker_id = list(trackers[object_class].keys())[row_ind[i]]
                center = known_centers[col_ind[i]]
                all_assignments.add(tracker_id)
                update_tracker(trackers[object_class][tracker_id], center[0], center[1])

    # If any tracker ids in trackers are unused, then update their kalman filter to lower the probability of their last known location. 
    for object_class in trackers.keys():
        for id in list(trackers[object_class].keys()):
            current_kf = trackers[object_class][id]
            if id not in all_assignments:
                current_kf.errorCovPost *= 1.1
            
            # Never be 100% certain
            current_kf.errorCovPost += 0.1

    return unused_boxes

try:
    # Example: {"person": {"person1": kalman_filter, "person2": kalman_filter}, "chair": {"chair1": kalman_filter} }
    trackers = {}
    # Example: {cup:{cup1:32}}
    age = {}
    # Example: [(cup, cup1)]
    forgetables = []
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
            center_x, center_y = get_center(boxes.xyxy[unused])
            trackers[object_class][id] = track(center_x, center_y)

        # Forget about items that haven't been seen in a few whiles
        for i in range(len(forgetables)):
            object_class, id = forgetables.pop()
            if object_class in trackers and id in trackers[object_class]:
                trackers[object_class].pop(id)
            if object_class in age and id in age[object_class]:
                age[object_class].pop(id)
                if len(trackers[object_class]) == 0:
                    trackers.pop(object_class)
                    age.pop(object_class)

        # Use the estimated locations of all visible objects
        for object_class in trackers.keys():
            for id in trackers[object_class].keys():
                # Get the average 2D location and 2D variance
                kalman_filter = trackers[object_class][id]
                row, col = np.round(np.array(kalman_filter.statePost[:2].reshape(-1)))
                covariance = kalman_filter.errorCovPost[:2,:2]
                uncertainty = np.sqrt(covariance[0][0]**2+covariance[1][1]**2)

                # Increment counter if location uncertainty is less than 5 standard deviations
                if uncertainty > 0.95:
                    if object_class not in age:
                        age[object_class] = {}
                    if id not in age[object_class]:
                        age[object_class][id] = 1
                    else:
                        age[object_class][id] += 1
                
                    # Remove id from trackers if count is over 60
                    # Forgettng items in 2 seconds depends on your computer's processing speed
                    threshold = 100
                    if age[object_class][id] > threshold:
                        print("Forgetting about", id, " ------------------------------------------")
                        forgetables.append((object_class, id))

                # Use/Display results with high confidence
                confidence = round(100 - (min(max(uncertainty, 0), 1) * 100), 3)
                print("I'm", confidence, "%\t sure that there's a", id, "\t\t\tat:", (row, col))
                
                # WARNING: The model sometimes hallucenates with 100% confidence for a single frame, so we wait until more data is gathered
                if uncertainty < 0.95 and uncertainty > 0.05:
                    cv2.drawMarker(frame, (int(row), int(col)), (0, 0, 255), cv2.MARKER_CROSS, 20)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    bottomLeftCornerOfText = (int(row)+20,int(col)+20)
                    fontScale = 1
                    fontColor = (255,255,255)
                    lineType = 2
                    cv2.putText(frame,str(id), bottomLeftCornerOfText, font, fontScale,fontColor,lineType)

                    # Draw a circle around the area where the object is predicted to exist
                    cv2.circle(frame,(int(row),int(col)),int(20*max(uncertainty, 0)),(255,255,255),thickness=1,lineType=8)

        cv2.imshow("result.png", frame)
        cv2.waitKey(delay=1)

finally:
    # After the loop release the cap object
    vid_stream.release()
    # Destroy all the windows
    cv2.destroyAllWindows()