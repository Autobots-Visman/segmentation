FROM ros:noetic-ros-base-focal

# Install build tools ======================================================
RUN apt-get update && \
    apt-get install -y \
        apt-utils \
        build-essential \
        git \
        python3-pip \
        python-is-python3 \
        python3-catkin-tools

# Install ROS ================================================================
# Use the following command in a running container to find dependencies that we
# can install ahead of time. We do this to make reinstalling dependencies faster
#
#   rosdep install --from-paths src --simulate --reinstall
RUN apt-get install -y \
        ros-noetic-catkin \
        ros-noetic-message-runtime \
        ros-noetic-message-generation \
        python3-numpy \
        ros-noetic-cv-bridge \
        ros-noetic-ros-pytest

WORKDIR /app
RUN mkdir src
RUN /ros_entrypoint.sh catkin build

# Install Computer Vision =====================================================
RUN pip install numpy
RUN pip install ultralytics opencv-python

# Download and cache yolo model
RUN yolo detect predict model=yolov8n-seg.pt source="https://ultralytics.com/images/bus.jpg" 

# copy in our own entrypoint
ADD docker/bin ./bin

ADD . src/segmentation
RUN rosdep install --from-paths src --ignore-src -r -y
RUN /app/bin/ros_entrypoint.sh catkin build

ENTRYPOINT ["/app/bin/ros_entrypoint.sh", "stdbuf", "-o", "L"]
CMD ["true"]
