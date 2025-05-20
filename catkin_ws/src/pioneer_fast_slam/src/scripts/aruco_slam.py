import rospy
import cv2
import numpy as np
import threading
import tf
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from visualization_msgs.msg import Marker, MarkerArray
from img_callback import image_callback
from callbacks import odom_callback
import FastSlam  # You'll need to implement or adapt this

class ArucoSlam:
    def __init__(self):
        rospy.init_node('aruco_detector_node')
        rospy.loginfo('ArUco Detector Node Started')
        
        self.marker_size = rospy.get_param('~marker_size', 0.16)
        self.bridge = CvBridge()
        self.tf_broadcaster = tf.TransformBroadcaster()
        
        # Publishers
        self.landmark_pub = rospy.Publisher('/aruco_landmarks', MarkerArray, queue_size=10)
        self.aruco_info_pub = rospy.Publisher('/aruco_info', Float64MultiArray, queue_size=10)
        
        # Camera calibration
        self.calibrate_camera()
        
        # ArUco dictionary setup
        try:
            self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_250)
        except:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250) 

        # ArUco parameters setup
        try:
            self.parameters = cv2.aruco.DetectorParameters_create()
        except:
            self.parameters = cv2.aruco.DetectorParameters()
            
        # Landmark angle cache
        self.dict = {}
        
        # Subscribe to camera topic
        self.image_sub = rospy.Subscriber("/camera/image/compressed", 
                                        CompressedImage, 
                                        self.image_callback)
    
    #camera calibration instrinsic parameters
    def calibrate_camera(self):
        #get camera calibration matrix (intrsic parameters, ex: focal length, principal point)

        #values from camera calibration
        dist = [0.1639958233797625, -0.271840030972792, 0.001055841660100477, -0.00166555973740089, 0.0]
        K = [322.0704122808738, 0.0, 199.2680620421962, 0.0, 320.8673986158544, 155.2533082600705, 0.0, 0.0, 1.0]
        
        # Adjust for standard image dimensions
        K[2] = 320  # Principal point x
        K[5] = 240  # Principal point y
        
        mtx = np.array(K).reshape(3, 3)
        dist = np.array(dist).reshape(1, 5)
        self.camera_matrix = mtx.astype(float)
        self.dist_coeffs = dist.astype(float)
    
    def image_callback_wrapper(self, data):
        image_callback(self, data)
    
    def odom_callback_wrapper(self, data):
        odom_callback(self, data)

    def publish_tf(self):
        """Publish TF frames for visualization"""
        # Broadcast map to odom (fixed transform)
        self.tf_broadcaster.sendTransform(
            (0.0, 0.0, 0.0),
            tf.transformations.quaternion_from_euler(0, 0, 0),
            rospy.Time.now(),
            "odom",
            "map"
        )
        
        # Get best particle and broadcast odom to base_link
        best_particle = self.my_slam.get_best_particle()
        x, y, theta = best_particle.pose
        quaternion = tf.transformations.quaternion_from_euler(0, 0, theta)
        self.tf_broadcaster.sendTransform(
            (x, y, 0),
            quaternion,
            rospy.Time.now(),
            "base_link",
            "odom"
        )

    def run(self):
        """Main execution loop"""
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            self.publish_tf()
            self.my_slam.publish_landmarks()
            rate.sleep()
        
        # Clean up
        cv2.destroyAllWindows()

if __name__=="__main__":
    try:
        # run ArucoSlam node
        node=ArucoSlam()
        node.run()
    except rospy.ROSInterruptException:
        #stop running in case of ctrl+c or node is shutdown
        pass
