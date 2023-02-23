cmake_minimum_required(VERSION 3.0.2)
project(object_detection)

find_package(catkin
    REQUIRED COMPONENTS
    message_generation
    std_msgs
)
catkin_python_setup()

add_message_files(
    FILES
    DetectedObjectList.msg
)

generate_messages(
    DEPENDENCIES
    std_msgs
)

catkin_package(
    CATKIN_DEPENDS
    message_runtime
)

include_directories(${catkin_INCLUDE_DIRS})

if(CATKIN_ENABLE_TESTING)
    find_package(rostest REQUIRED)
    add_rostest(tests/test_object_detection.test)
endif()
