#=============================== assignment ===============================
'''!
@brief  A class for the assignment of estimators during object detection.

The goal of this class is to be a general framework for assigning the estimators (a.k.a estimators) 
of detected objects. We are trying to blend previous methods from Handy and 
PuzzleSolver to create a base class usable for all implementations. 
We would also like to implement soft removal to provide support for 
a semantic scene graph that remembers past estimators. 
'''
#=============================== assignment ===============================
#
# @author   Indira Tatikola,        itatikola3@gatech.edu
# @date     2024/03/8              [created]
#
#=============================== assignment ===============================

from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
import numpy as np
import cv2

class Assignment():


    def __init__(self, classes):
        self.classes = classes
        self.estimators = {}
        self.boxes = None
        self.age = {}
        self.forgettables = []
        self.unused_boxes = None
        self.assignments = set()
        self.ageThreshold = 60
        self.errorCovMultiplier = 1.1


    def create_kf(self, x, y):
        """Create a new kalman filter tracker for tracking one object."""
        # Assume the state and measurement paramters have a dimension of 2. There isn't a control vector. 
        kf = cv2.KalmanFilter(4, 2)
        # Which dimensions should be measured?
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        # What dimensions need to be multiplied together to represent change?
        kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        # How much should each measurement be trusted (bigger = more trust)?
        kf.processNoiseCov = np.array([[5e-3, 0, 0, 0], [0, 5e-3, 0, 0], [0, 0, 5e-3, 0], [0, 0, 0, 5e-3]], np.float32)
        # What is the current state?
        kf.statePre = np.array(
            [[x], [y], [0], [0]], np.float32
        )
        return kf
    
    def update_kf(self, cls, id, measurement):
        """Update the kalman filter with a new measurement and return the estimated object location."""
        kf = self.estimators[cls][id]
        kf.correct(np.array([[np.float32(measurement[0])], [np.float32(measurement[1])]]))
        return kf.predict()

    def get_center(self, xyxy):
        """Convert a bounding box corner from the array boxes.xyxy to a center xy coordinate of the box. """
        bbox_center_x = (xyxy[2] + xyxy[0]) / 2
        bbox_center_y = (xyxy[3] + xyxy[1]) / 2
        return bbox_center_x, bbox_center_y
    
    def update_errorCov(self, multiplier, base=0.1):
        """Update the error covariance matrix for all kalman filters that have not been assigned"""
        for cls in self.estimators.keys():
            for id in list(self.estimators[cls].keys()):
                current_kf = self.estimators[cls][id]
                if id not in self.assignments:
                    current_kf.errorCovPost *= multiplier
                
                # Never be 100% certain
                current_kf.errorCovPost += base
    
    def hungarian_assignment(self):
        """Perform hungarian assignment of each object to each bounding box center such that the
        cost matrix is minimized"""
        print("inside hungarian assignment")
        self.unused_boxes = set(range(len(self.boxes)))
        for cls in self.estimators.keys():
            print(cls)
            centers = []
            for i in range(len(self.boxes)):
                if self.classes[int(self.boxes[i].cls)] == cls:
                    self.unused_boxes.discard(i)
                    center_x, center_y = self.get_center(self.boxes.xyxy[i])
                    centers.append((center_x, center_y))

            print(centers)
            cost_matrix = np.zeros((len(self.estimators[cls]), len(centers)))
            for i, (id, kf) in enumerate(self.estimators[cls].items()):
                for j, center in enumerate(centers):
                    location = kf.statePost[:2].reshape(-1)
                    cost_matrix[i][j] = np.linalg.norm(location - center)

            rows, cols = linear_sum_assignment(cost_matrix)

            for i, row in enumerate(rows):
                col = cols[i]
                id = list(self.estimators[cls].keys())[row]
                center = centers[col]
                self.assignments.add(id)
                self.update_kf(cls, id, center)
    
    def create_estimator(self, cls, box=None, kf=None):
        if not cls in self.estimators:
            self.estimators[cls] = {}
        count = len(self.estimators[cls])
        id = cls + str(count + 1)
        if not box is None:
            x, y = self.get_center(self.boxes.xyxy[box])
            kf = self.create_kf(x, y)
        self.estimators[cls][id] = kf
        return kf
    
    ## IN FORGET & AGE IS WHERE A SOFT REMOVAL CAN BE IMPLEMENTED
    def forget(self):
        print("inside forget")
        for i in range(len(self.forgettables)):
            cls, id = self.forgettables.pop()
            if cls in self.estimators and id in self.estimators[cls]:
                self.estimators[cls].pop(id)
                self.age[cls].pop(id)
                if len(self.estimators[cls]) == 0:
                    self.estimators.pop(cls)
                    self.age.pop(cls)
    
    def ageBoxes(self, threshold):
        print("inside age")
        for cls in self.estimators.keys():
            for i, (id, kf) in enumerate(self.estimators[cls].items()):
                covariance = kf.errorCovPost[:2,:2]
                uncertainty = np.sqrt(covariance[0][0]**2+covariance[1][1]**2)

                # Increment counter if location uncertainty is greater than 5 standard deviations
                if uncertainty > 0.95:
                    if cls not in self.age:
                        self.age[cls] = {}
                    if id not in self.age[cls]:
                        self.age[cls][id] = 1
                    else:
                        self.age[cls][id] += 1
                
                    # Remove id from trackers if count is over 60
                    # Forgetting items in 2 seconds depends on your computer's processing speed
                    if self.age[cls][id] > threshold:
                        self.forgettables.append((cls, id))   

    def assign(self, boxes):
        self.boxes = boxes
        self.unused_boxes = []
        self.hungarian_assignment()
        self.update_errorCov(self.errorCovMultiplier)

        # Create estimators for each unused box - boxes that are completely new
        for unused_box in self.unused_boxes:
            cls = self.classes[int(self.boxes.cls[unused_box])]
            self.create_estimator(cls, box=unused_box)
        
        # officially forget items one frame after they age past the threshold
        self.forget()

        # age all other items
        self.ageBoxes(self.ageThreshold)
        