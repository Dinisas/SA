#!/usr/bin/env python3
# Import necessary libraries
import rospy
import subprocess
import argparse
import os
import time
# aruco and open cv libraries
import cv2
import cv2.aruco as aruco
import numpy as np
# multi threading libraries
import threading
# ROS transform library (creates object to publish coordinate frame transformationss)
import tf
from geometry_msgs.msg import TransformStamped
# image messages in ROS
from sensor_msgs.msg import CompressedImage
# from sensor_msgs.msg import Image 

# odometry messages in ROS
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
        # node will run for rosbag duration plus 5 seconds for processing (shutdown timer)
        self.k = rosbag_time + 5
        # creates TF Broadcaster used to publish coordinate frame transformations to ROS
        self.tf_broadcaster = tf.TransformBroadcaster()
         # Camera calibration (reads camera matrix and distortion coefficients)
        self.calibrate_camera()
        # threading lock for parallel odometry and camera callbacks
        self.lock = threading.Lock()
        
        rospy.loginfo('ArucoSLAM Node Started')
        # Note: ROS node is now initialized in main.py before this class is created
        
        # Create SLAM object
        # unpacks slam configuration parameters
        window_size_pixel, size_m, number_particles, tunning_options = slam_variables
        print(f"DEBUG: ArucoSLAM unpacked slam_variables: particles = {number_particles}")
        # calls create_slam that creates Fast Slam object
        self.create_slam(window_size_pixel, size_m, tunning_options, number_particles)

        # Subscribe to relevant ROS topics
        # subscribes to image topic and calls callback for each new image
        self.image_sub = rospy.Subscriber("/camera/image_raw/compressed", CompressedImage, self.image_callback)
        # self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.image_callback)

        # subscribes to odometry topic and calls callback for each update in robot position    
        self.odom_sub = rospy.Subscriber("/pose", Odometry, self.odom_callback)
        # landmark publisher for visualization
        self.landmark_pub = rospy.Publisher('/landmarks', MarkerArray, queue_size=10)
        # stores aruco markers detected in the current frame
        self.current_aruco = [] 
        # stores robots current pose (location + orientation (x,y,quaternion))
        self.odom = [0,0,0]
        # Initialize the CvBridge object (converts between Ros image messages and Open cv format)
        self.bridge = CvBridge() 

        # Load ArUco dictionary (tries for different openCV versions)
        try:
            self.aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_100)
        except:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100) 

        # Create ArUco detector parameters algorithm
        try:
            self.parameters = aruco.DetectorParameters_create()
        except:  # For different versions of OpenCV
            self.parameters = cv2.aruco.DetectorParameters()
        # angle measurements for each marker id
        self.dict = {}
        # counter for callback calls
        self.count = 0
        # stores intial robot pose, everything else is built based on that
        self.tara = None
        
        # Performance monitoring
        self.callback_count = 0
        self.start_time = time.time()
        self.last_metrics_save = time.time()
        self.metrics_save_interval = 30.0  # Save metrics every 30 seconds

    # Create SLAM object
    def create_slam(self, window_size_pixel, size_m, tunning_options, number_particles):
        print(f"DEBUG: create_slam called with number_particles = {number_particles}")
        #Pioneer P3-DX wheelbase (distance between two drive wheels, used to convert wheel velocities into motion)
        pioneer_L = 0.33
        print(f"DEBUG: About to create FastSlam with {number_particles} particles")
        self.my_slam = FastSlam(tunning_options, window_size_pixel, size_m, pioneer_L, number_particles)
        print(f"DEBUG: FastSlam created. Actual particles in SLAM: {self.my_slam.num_particles}")

    # Callback function for odometry data
    def odom_callback(self, odom_data):
        with self.lock:  # Ensure thread-safe operation with a lock
            #Extract position and orientation from odometry data
            # position (Ros message structure pose.pose.position)
            x = odom_data.pose.pose.position.x
            y = odom_data.pose.pose.position.y
            # quaternion for orientation 3D rotation (Ros message structure pose.pose.quartenion)
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

            # Adjust odometry based on the initial reference position (tara) for x and y, position relative to start
            # we dont need to do this for orientation because its already relative
            self.odom[0] -= self.tara[0]
            self.odom[1] -= self.tara[1]
            
            # Update FAST SLAM algorithm with the new odometry information
            self.my_slam.update_odometry(self.odom)

    # Callback function for image data
    def image_callback(self, data):
        # Ensure thread-safe operation with a lock
        with self.lock:
            # resets list of currently detected arucos for this frame
            self.current_aruco = []
            self.callback_count += 1

            try:
                # Convert compressed image message to OpenCV format
                cv_image = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
                # cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")  
            except Exception as e:
                rospy.logerr("CvBridge Error: {0}".format(e))
                return

            # Convert the image to grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            # Detect ArUco markers in the image (corners, list of marker ids, shapes that look like markers but arent)
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
            if ids is not None and len(ids) > 0:
                for i in range(len(ids)):
                    marker_corners = corners[i][0]
                    # Draw bounding box around detected markers
                    cv2.polylines(cv_image, [np.int32(marker_corners)], True, (255, 165, 0), 3)

                    # perspective projection, Estimate pose of each marker in relation to the camera (tvec=translation vector in camera frame)
                    _, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.16, self.camera_matrix, self.dist_coeffs)

                    # Draw a circle at the center of the image (for reference, center is 320x240 for 640x480 image)
                    cv2.circle(cv_image, (320, 240), radius=10, color=(255, 192, 203), thickness=-1)

                    # Transform the translation vector from camera coordinates to robot coordinates (coordinate transformation)
                    tvec = transform_camera_to_robot(tvec[0][0])

                    # Convert 3D position Cartesian coordinates to 2D polar coordinates (returns distance and bearing angle)
                    dist, phi = cart2pol(tvec[0], tvec[2])

                    # checks if the marker as been seen before, fill the dictionary with angles if the marker is detected
                    if ids[i][0] not in self.dict:
                        self.dict[ids[i][0]] = []
                    # adds bearing angle to the dictionary for the marker id
                    self.dict[ids[i][0]].append(phi)
                    
                    # Display the distance to the marker (bottom-left corner of the marker)
                    cv2.putText(cv_image, 'dist= ' + str(round(dist, 3)), 
                                (int(marker_corners[2][0] - 80), int(marker_corners[2][1]) + 45), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 2)

                    # Compute median of the last few measurements to reduce noise
                    if len(self.dict[ids[i][0]]) >= 3:
                        # might be need to + or - (Compute median), negates for coordinate consistency                
                        phi5 = -np.median(np.sort(self.dict[ids[i][0]][-4:-1]))
                        # Remove the oldest measurement
                        self.dict[ids[i][0]].pop(0)
                        # displays measured angle (top-right corner of the marker)
                        cv2.putText(cv_image, 'ang=' + str(round(phi5, 3)), 
                                    (int(marker_corners[1][0] - 70), int(marker_corners[1][1]) - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        # creates tuple with distance, filrered angle and marker id
                        self.current_aruco.append((dist, phi5, ids[i][0]))
                    else:
                        # first two measurements not enough for median, uses raw angle measurement
                        self.current_aruco.append((dist, -phi, ids[i][0]))

            # Compute SLAM with the all the detected ArUco markers (contains lists of tuples (distance, angle, marker id))
            self.my_slam.compute_slam(self.current_aruco)
            # Show the image window with detected markers throughout time
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
                os.path.expanduser("~/SA/catkin_ws/camera_calibration.npz"),
                "/home/dinisas/SA/catkin_ws/camera_calibration.npz"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    print(f"Found calibration at: {path}")
                    # loads calibration data
                    data = np.load(path)
                    # extracts camera intrisinc matrix and distortion coefficients
                    self.camera_matrix = data['camera_matrix']
                    self.dist_coeffs = data['dist_coeffs']
                    print("Calibration loaded successfully!")
                    return
                    
            raise FileNotFoundError("Could not find calibration file in any expected location")
            
        except Exception as e:
            print(f"Error loading calibration: {e}")
            print("Please check if the calibration file exists at the specified path.")
            exit(1)

    # Publish coordinate frame transformations (chain of transformations)
    def publish_tf(self):
        # creates a ROS transform broadcaster object
        br = tf.TransformBroadcaster()

        # Publishes a transformation between two coordinate frames (between map and odometry frames)
        br.sendTransform(
            (0.0, 0.0, 0.0),
            # submodule of tf package, converts euler angles to quarternion representation
            tf.transformations.quaternion_from_euler(0, 0, 0),
            rospy.Time.now(),
            # child frame
            "odom",
            # parent frame
            "map"
        )
        # map -> odom -> base_link
        # Get best particle (robot pose) and broadcast odom to base_link (robots base frame)
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

    def log_performance_summary(self):
        """Log a performance summary to the ROS console."""
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            callback_rate = self.callback_count / elapsed_time
            
            # Get current metrics from SLAM
            metrics = self.my_slam.get_current_metrics()
            
            rospy.loginfo("=== FastSLAM Performance Summary ===")
            rospy.loginfo(f"Runtime: {elapsed_time:.1f}s, Callbacks: {self.callback_count}")
            rospy.loginfo(f"Callback Rate: {callback_rate:.2f} Hz")
            
            if 'effective_particle_count' in metrics:
                diversity = metrics['effective_particle_count']
                rospy.loginfo(f"Particle Diversity: {diversity['n_eff_ratio']:.2%} ({diversity['current_n_eff']:.1f}/{self.my_slam.num_particles})")
            
            if 'timing_performance' in metrics:
                timing = metrics['timing_performance']
                if 'real_time_performance' in timing:
                    rt = timing['real_time_performance']
                    rospy.loginfo(f"Real-time Factor: {rt['real_time_factor']:.2f}x")
            
            rospy.loginfo(f"Detection Rate: {metrics['detection_rate']:.1f}%")
            rospy.loginfo(f"Current RMSE: {metrics['current_rmse']:.3f}m")

    # Main loop to run the node
    def run(self):
        start_time = rospy.Time.now()
        last_performance_log = time.time()
        performance_log_interval = 60.0  # Log performance every 60 seconds
        
        while not rospy.is_shutdown():  # Continue running until ROS node is shutdown
            current_time = time.time()
            
            # Publish transformation frames
            self.publish_tf()
            # Publish detected landmarks
            self.my_slam.publish_landmarks()
            
            # Periodic performance logging
            if (current_time - last_performance_log) > performance_log_interval:
                self.log_performance_summary()
                last_performance_log = current_time
            
            # Periodic metrics saving
            if (current_time - self.last_metrics_save) > self.metrics_save_interval:
                timestamp = int(current_time)
                metrics_filename = f"slam_metrics_{timestamp}.txt"
                self.my_slam.save_metrics_to_file(metrics_filename)
                rospy.loginfo(f"Periodic metrics saved to {metrics_filename}")
                self.last_metrics_save = current_time
            
            if self.rosbag_finished(start_time, self.k):  # Check if the rosbag playback has finished
                rospy.loginfo("Rosbag playback finished. Performing final analysis...")
                
                # Save final comprehensive metrics
                final_metrics_filename = f"final_slam_metrics_{int(time.time())}.txt"
                self.my_slam.save_metrics_to_file(final_metrics_filename)
                
                # Log final performance summary
                self.log_performance_summary()
                
                # Log some final statistics
                final_metrics = self.my_slam.get_current_metrics()
                rospy.loginfo("=== Final SLAM Results ===")
                rospy.loginfo(f"Final ATE: {final_metrics['current_ate']:.4f}m")
                rospy.loginfo(f"Final RMSE: {final_metrics['current_rmse']:.4f}m")
                rospy.loginfo(f"Total Landmarks Mapped: {final_metrics['landmarks_detected']}")
                rospy.loginfo(f"Detection Success Rate: {final_metrics['detection_rate']:.1f}%")
                
                if 'effective_particle_count' in final_metrics:
                    diversity = final_metrics['effective_particle_count']
                    rospy.loginfo(f"Final Particle Diversity: {diversity['n_eff_ratio']:.2%}")
                
                if 'landmark_stability' in final_metrics:
                    stability = final_metrics['landmark_stability']
                    if stability['total_landmarks'] > 0:
                        stability_rate = (stability['stable_landmarks'] / stability['total_landmarks']) * 100
                        rospy.loginfo(f"Landmark Stability Rate: {stability_rate:.1f}%")
                
                rospy.loginfo(f"Comprehensive metrics saved to {final_metrics_filename}")
                rospy.signal_shutdown("Rosbag playback finished")  # Shutdown ROS node
                break  # Exit the loop
            rospy.sleep(0.1)  # Process ROS callbacks once
            
        cv2.destroyAllWindows()  # Close OpenCV windows when node exits
        pygame.display.quit()
        pygame.quit()

    # Check if the rosbag playback has finished (adds rosbag duartion to start time and checks idf it as finished)
    def rosbag_finished(self, start_time, duration):
        end_time = start_time + rospy.Duration.from_sec(duration)
        if rospy.Time.now() > end_time:
            return True  # Playback finished
        else:
            return False  # Playbook ongoing

    # Get the best trajectory from the SLAM object
    def get_trajectory(self):
        return self.my_slam.get_best_trajectory()