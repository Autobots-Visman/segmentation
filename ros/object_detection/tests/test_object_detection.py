import numpy as np
import pytest
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


@pytest.fixture()
def node():
    rospy.init_node("test_detect", anonymous=True)


@pytest.fixture()
def bridge():
    return CvBridge()


def test_recieve_image_mock(node, bridge):
    """A simple test to show-case how to mock a message and check that it is received."""

    # send an image that is all red to the topic
    shape = (480, 640, 3)
    red = np.zeros(shape, dtype=np.uint8)
    red[:, :, 0] = 255
    msg = bridge.cv2_to_imgmsg(red, "bgr8")

    # http://wiki.ros.org/rospy/Overview/Publishers%20and%20Subscribers
    # we latch the message, since the subscribe is not ready yet
    pub = rospy.Publisher("/camera/color/image_raw", Image, queue_size=1, latch=True)
    pub.publish(msg)

    # check that the image is received
    msg = rospy.wait_for_message("/camera/color/image_raw", Image)
    assert msg, "no message received"
    img = bridge.imgmsg_to_cv2(msg, msg.encoding)

    red, green, blue = img.reshape(-1, 3).sum(axis=0)
    assert red > green and red > blue, "image is not red"
    assert green == 0 and blue == 0, "image is not red"
