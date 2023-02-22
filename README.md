# segmentation
A ROS package that converts images into a semantic scene graph for other systems.

## Goals
The final deliverable should be a reusable modular ROS package that takes video feed as input and outputs labeled segmented scene semantics.
    
The goal for this repository is not a robot using this package. Integration will occur in another repository. 

## Timeline
Part 1: Convert video into a semantic scene
    1. Install YOLO, OpenCV, and ROS Noetic
    2. Implement YOLO on a test image
    3. Modify YOLO to work with a live camera feed
    4. Implement deduplication of objects between frames
    5. Implement data structure to store data compatible with ROS messages

Part 2: Enhance model to generate scene graphs
    1. Read about Relational Scene Graphs by Jenkins
    2. Implement scene graphs
    3. Integrate with previous ROS system

## Setup
Todo

## Usage
Todo
