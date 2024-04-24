# Description:
#     A script that performs semantic segmentation

from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from assignment import Assignment
import numpy as np
import cv2

# YOLOv8 segmentation models use the -seg suffix , i.e. yolov8n-seg.pt and are pretrained on COCO.
model = YOLO('yolov8n-seg.pt')

# Use CAMERA_ID if it's open. Otherwise find another camera.
# Handy: CAMERA_ID 4 is RGB, CAMERA_ID 2 is BW with dots.
CAMERA_ID = 4
vid_stream = cv2.VideoCapture(CAMERA_ID)
if not vid_stream.isOpened():
    raise ValueError("CAMERA_ID: ", CAMERA_ID, " is not valid.")

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

try:
    # create assignment object, passing in all class names from model
    assigner = Assignment(model.names)
    while True:
        # get current frame's bounding boxes
        frame   = get_frame(vid_stream)
        labels  = label(frame)
        boxes = labels.boxes
        # perform (re)assignment
        assigner.assign(boxes)
        estimators = assigner.estimators
        # display
        for cls in estimators.keys():
            print(cls)
            for id in estimators[cls].keys():
                print(id)
                # Get the average 2D location and 2D variance
                kf = estimators[cls][id]
                print(kf.statePost)
                row, col = np.round(np.array(kf.statePost[:2].reshape(-1)))
                covariance = kf.errorCovPost[:2,:2]
                uncertainty = np.sqrt(covariance[0][0]**2+covariance[1][1]**2)
                # Display results with high confidence
                confidence = round(100 - (min(max(uncertainty, 0), 1) * 100), 3)
                
                # WARNING: The model sometimes hallucenates with 100% confidence for a single frame, so we wait until more data is gathered
                if uncertainty < 0.95 and uncertainty > 0.05:
                    print("I'm", confidence, "%\t sure that there's a", id, " at:", (row, col))
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