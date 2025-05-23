#!/usr/bin/env python3
# Import necessary libraries
import rospy
import subprocess
import argparse
import os
import cv2
import cv2.aruco as aruco
import numpy as np
import threading
import tf
from geometry_msgs.msg import TransformStamped
# from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image   
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import re
import pygame
from visualization_msgs.msg import Marker, MarkerArray
from utils import get_rosbag_duration, cart2pol, transform_camera_to_robot
from fast_slam import FastSlam

class ArucoSLAM:
    def __init__(self, rosbag_time, slam_variables):
        # Initialize instance variables and set up ROS node
        self.k = rosbag_time + 5
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.calibrate_camera()  # Camera calibration
        self.lock = threading.Lock()
        
        rospy.loginfo('ArucoSLAM Node Started')
        rospy.init_node('aruco_slam')  # Initialize the ROS node
        
        # Create SLAM object
        window_size_pixel, size_m, number_particles, tunning_options = slam_variables
        self.create_slam(window_size_pixel, size_m, tunning_options, number_particles)

        # Subscribe to relevant ROS topics
        # self.image_sub = rospy.Subscriber("/camera/image/compressed", CompressedImage, self.image_callback)
        self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.image_callback)    
        self.odom_sub = rospy.Subscriber("/pose", Odometry, self.odom_callback)
        self.landmark_pub = rospy.Publisher('/landmarks', MarkerArray, queue_size=10)
        self.current_aruco = []  
        self.odom = [0,0,0]
        self.bridge = CvBridge()  # Initialize the CvBridge object

        # Load ArUco dictionary
        try:
            self.aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_100)
        except:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100) 

        # Create ArUco detector parameters
        try:
            self.parameters = aruco.DetectorParameters_create()
        except:  # For different versions of OpenCV
            self.parameters = cv2.aruco.DetectorParameters()
        self.dict = {}
        self.map = {}
        self.count = 0
        self.tara = None

    # Create SLAM object
    def create_slam(self, window_size_pixel, size_m, tunning_options, number_particles):
        pioneer_L = 0.33  # Pioneer P3-DX wheelbase (check your specific model)
        self.my_slam = FastSlam(tunning_options, window_size_pixel, size_m, pioneer_L, number_particles)

    # Callback function for odometry data
    def odom_callback(self, odom_data):
        with self.lock:  # Ensure thread-safe operation with a lock
            # Extract position and orientation from odometry data
            x = odom_data.pose.pose.position.x
            y = odom_data.pose.pose.position.y
            xq = odom_data.pose.pose.orientation.x
            yq = odom_data.pose.pose.orientation.y
            zq = odom_data.pose.pose.orientation.z
            wq = odom_data.pose.pose.orientation.w
            quater = [xq, yq, zq, wq]
            
            # Update the odometry information
            self.odom = [x, y, quater]
            
            # Initialize reference (tara) position on the first callback
            if self.count == 0:
                self.tara = [x, y, quater]
                self.count += 1

            # Adjust odometry based on the initial reference position
            self.odom[0] -= self.tara[0]
            self.odom[1] -= self.tara[1]
            
            # Update SLAM with the new odometry information
            self.my_slam.update_odometry(self.odom)

    # Callback function for image data
    def image_callback(self, data):
        with self.lock:  # Ensure thread-safe operation with a lock
            self.current_aruco = []

            try:
                # Convert compressed image message to OpenCV format
                # cv_image = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
                cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")  
            except Exception as e:
                rospy.logerr("CvBridge Error: {0}".format(e))
                return

            # Convert the image to grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            # Detect ArUco markers in the image
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
            if ids is not None and len(ids) > 0:
                for i in range(len(ids)):
                    marker_corners = corners[i][0]
                    # Draw bounding box around detected markers
                    cv2.polylines(cv_image, [np.int32(marker_corners)], True, (0, 255, 0), 2)
                    # Estimate pose of each marker
                    _, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.25, self.camera_matrix, self.dist_coeffs)

                    # Draw a circle at the center of the image (for reference)
                    cv2.circle(cv_image, (320, 240), radius=10, color=(255, 0, 0), thickness=-1)

                    # Transform the translation vector to robot coordinates
                    tvec = transform_camera_to_robot(tvec[0][0])
                    # Convert Cartesian coordinates to polar coordinates
                    dist, phi = cart2pol(tvec[0], tvec[2])

                    # Fill the dictionary if the marker is detected
                    if ids[i][0] not in self.dict:
                        self.dict[ids[i][0]] = []
                    self.dict[ids[i][0]].append(phi)
                    
                    # Display the distance to the marker
                    cv2.putText(cv_image, 'dist= ' + str(round(dist, 3)), 
                                (int(marker_corners[2][0] - 80), int(marker_corners[2][1]) + 45), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                    # Compute median of the last few measurements to reduce noise
                    if len(self.dict[ids[i][0]]) >= 3:                
                        phi5 = -np.median(np.sort(self.dict[ids[i][0]][-4:-1]))  # Compute median
                        self.dict[ids[i][0]].pop(0)  # Remove the oldest measurement
                        cv2.putText(cv_image, 'ang=' + str(round(phi5, 3)), 
                                    (int(marker_corners[1][0] - 70), int(marker_corners[1][1]) - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                        self.current_aruco.append((dist, phi5, ids[i][0]))
                    else:
                        self.current_aruco.append((dist, -phi, ids[i][0]))

            # Compute SLAM with the detected ArUco markers
            self.my_slam.compute_slam(self.current_aruco)
            # Show the image with detected markers
            cv2.imshow('Aruco Detection', cv_image)
            cv2.waitKey(3)

    # Calibrate the camera
    def calibrate_camera(self):
        try:
            import os
            print(f"Current working directory: {os.getcwd()}")
            print(f"Looking for calibration file at: {os.path.abspath('camera_calibration.npz')}")
            
            # Try multiple locations
            possible_paths = [
                "camera_calibration_outro_grupo.npz",
                os.path.expanduser("~/SA/catkin_ws/camera_calibration_outro_grupo.npz"),
                "/home/dinisas/SA/catkin_ws/camera_calibration_outro_grupo.npz"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    print(f"Found calibration at: {path}")
                    data = np.load(path)
                    self.camera_matrix = data['camera_matrix']
                    self.dist_coeffs = data['dist_coeffs']
                    print("Calibration loaded successfully!")
                    return
                    
            raise FileNotFoundError("Could not find calibration file in any expected location")
            
        except Exception as e:
            print(f"Error loading calibration: {e}")
            print("Please check if the calibration file exists at the specified path.")
            exit(1)

    # Publish transformation frames
    def publish_tf(self):
        # Broadcast map to odom
        br = tf.TransformBroadcaster()
        br.sendTransform(
            (0.0, 0.0, 0.0),
            tf.transformations.quaternion_from_euler(0, 0, 0),
            rospy.Time.now(),
            "odom",
            "map"
        )

        # Get best particle and broadcast odom to base_link
        best_particle = self.my_slam.get_best_particle()
        x, y, theta = best_particle.pose  # Access the pose property correctly
        quaternion = tf.transformations.quaternion_from_euler(0, 0, theta - (np.pi))
        br.sendTransform(
            (-x, y, 0),
            quaternion,
            rospy.Time.now(),
            "base_link",
            "odom"
        )

    # Main loop to run the node
    def run(self):
        start_time = rospy.Time.now()
        while not rospy.is_shutdown():  # Continue running until ROS node is shutdown
            self.publish_tf()  # Publish transformation frames
            self.my_slam.publish_landmarks()
            if self.rosbag_finished(start_time, self.k):  # Check if the rosbag playback has finished
                rospy.loginfo("Rosbag playback finished. Shutting down...")
                rospy.signal_shutdown("Rosbag playback finished")  # Shutdown ROS node
                break  # Exit the loop
            rospy.sleep(0.1)  # Process ROS callbacks once
        cv2.destroyAllWindows()  # Close OpenCV windows when node exits
        pygame.display.quit()
        pygame.quit()

    # Check if the rosbag playback has finished
    def rosbag_finished(self, start_time, duration):
        end_time = start_time + rospy.Duration.from_sec(duration)
        if rospy.Time.now() > end_time:
            return True  # Playback finished
        else:
            return False  # Playback ongoing

    # Get the trajectory from the SLAM object
    def get_trajectory(self):
        return self.my_slam.get_best_trajectory()