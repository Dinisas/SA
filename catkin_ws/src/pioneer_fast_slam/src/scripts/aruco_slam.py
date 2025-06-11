#!/usr/bin/env python3
# Import necessary libraries
import rospy
import subprocess
import argparse
import os
import time
import cv2
import cv2.aruco as aruco
import numpy as np
import threading
import tf
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import re
import pygame
from collections import deque

from visualization_msgs.msg import Marker, MarkerArray
from utils import get_rosbag_duration, cart2pol, transform_camera_to_robot
from fast_slam import FastSlam

class ArucoSLAM:

    def __init__(self, rosbag_time, slam_variables):
        # Initialize instance variables and set up ROS node
        self.k = rosbag_time + 5
        self.tf_broadcaster = tf.TransformBroadcaster()
        
        # Camera calibration
        rospy.loginfo('Loading camera calibration...')
        self.calibrate_camera()
        
        # Threading lock for parallel callbacks
        self.lock = threading.Lock()
        self.use_data_association = False
        
        # Create SLAM object
        window_size_pixel, size_m, number_particles, tunning_options, groundtruth_file = slam_variables
        self.create_slam(window_size_pixel, size_m, tunning_options, number_particles, groundtruth_file)

        # Subscribe to relevant ROS topics
        self.image_sub = rospy.Subscriber("/camera/image_raw/compressed", CompressedImage, self.image_callback)
        self.odom_sub = rospy.Subscriber("/pose", Odometry, self.odom_callback)
        self.landmark_pub = rospy.Publisher('/landmarks', MarkerArray, queue_size=10)
        
        # Initialize variables
        self.current_aruco = []
        self.odom = [0,0,0]
        self.bridge = CvBridge()
        
        # ArUco setup
        try:
            self.aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_100)
        except:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        
        try:
            self.parameters = aruco.DetectorParameters_create()
        except:
            self.parameters = cv2.aruco.DetectorParameters()
        
        self.dict = {}
        self.count = 0
        self.tara = None
        
        # SYNCHRONIZED UPDATE VARIABLES
        self.update_rate = 30.0  # Hz - synchronized update rate
        self.last_update_time = time.time()
        self.min_update_interval = 1.0 / self.update_rate
        
        # Motion detection thresholds
        self.min_linear_motion = 0.001  # 1mm
        self.min_angular_motion = 0.001  # ~0.057 degrees
        
        # Odometry buffering
        self.odometry_buffer = deque(maxlen=10)
        self.last_processed_odom = None
        self.accumulated_motion = {'linear': 0.0, 'angular': 0.0}
        
        # Camera measurement buffering
        self.measurement_buffer = deque(maxlen=5)
        self.new_measurements_available = False
        
        # State tracking
        self.robot_moving = False
        self.last_significant_motion_time = time.time()
        
        # Performance monitoring
        self.callback_count = 0
        self.start_time = time.time()
        self.last_metrics_save = time.time()
        self.metrics_save_interval = 30.0
        
        # Enhanced metrics logging
        self.last_msp_eta_log = time.time()
        self.msp_eta_log_interval = 10.0  # Log MSP and ETA every 10 seconds

    def create_slam(self, window_size_pixel, size_m, tunning_options, number_particles, groundtruth_file):
        pioneer_L = 0.33
        self.my_slam = FastSlam(tunning_options, window_size_pixel, size_m, pioneer_L, number_particles, groundtruth_file)

    def odom_callback(self, odom_data):
        """Buffer odometry data instead of immediately processing it"""
        with self.lock:
            # Extract position and orientation
            x = odom_data.pose.pose.position.x
            y = odom_data.pose.pose.position.y
            xq = odom_data.pose.pose.orientation.x
            yq = odom_data.pose.pose.orientation.y
            zq = odom_data.pose.pose.orientation.z
            wq = odom_data.pose.pose.orientation.w
            quater = [xq, yq, zq, wq]
            
            # Update odometry
            self.odom = [x, y, quater]
            
            # Initialize reference position on first callback
            if self.count == 0:
                self.tara = [x, y, quater]
                self.count += 1
                self.last_processed_odom = [x, y, quater]

            # Adjust odometry relative to initial position
            self.odom[0] -= self.tara[0]
            self.odom[1] -= self.tara[1]
            
            # Buffer the odometry reading with timestamp
            self.odometry_buffer.append({
                'time': odom_data.header.stamp.to_sec(),
                'odom': self.odom.copy()
            })
            
            # Check if robot is moving
            if self.last_processed_odom:
                dx = self.odom[0] - self.last_processed_odom[0]
                dy = self.odom[1] - self.last_processed_odom[1]
                linear_motion = np.sqrt(dx**2 + dy**2)
                
                # Simple angular motion check (could be improved with proper quaternion diff)
                angular_motion = abs(np.arccos(np.clip(
                    quater[3] * self.last_processed_odom[2][3] + 
                    quater[0] * self.last_processed_odom[2][0] + 
                    quater[1] * self.last_processed_odom[2][1] + 
                    quater[2] * self.last_processed_odom[2][2], -1.0, 1.0))) * 2
                
                # Accumulate motion
                self.accumulated_motion['linear'] += linear_motion
                self.accumulated_motion['angular'] += angular_motion
                
                # Update motion status
                if linear_motion > self.min_linear_motion or angular_motion > self.min_angular_motion:
                    self.robot_moving = True
                    self.last_significant_motion_time = time.time()
                elif time.time() - self.last_significant_motion_time > 0.5:  # 500ms without motion
                    self.robot_moving = False

    def image_callback(self, data):
        """Process camera data and buffer measurements"""
        with self.lock:
            self.current_aruco = []
            self.callback_count += 1

            try:
                cv_image = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
            except Exception as e:
                rospy.logerr("CvBridge Error: {0}".format(e))
                return

            # Detect ArUco markers
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
            
            if ids is not None and len(ids) > 0:
                for i in range(len(ids)):
                    marker_corners = corners[i][0]
                    cv2.polylines(cv_image, [np.int32(marker_corners)], True, (255, 165, 0), 3)

                    # Calculate marker pixel size
                    side1 = np.linalg.norm(marker_corners[1] - marker_corners[0])
                    side2 = np.linalg.norm(marker_corners[2] - marker_corners[1]) 
                    side3 = np.linalg.norm(marker_corners[3] - marker_corners[2])
                    side4 = np.linalg.norm(marker_corners[0] - marker_corners[3])
                    marker_pixel_size = (side1 + side2 + side3 + side4) / 4.0

                    # Estimate pose
                    _, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.16, self.camera_matrix, self.dist_coeffs)
                    cv2.circle(cv_image, (320, 240), radius=10, color=(255, 192, 203), thickness=-1)

                    # Transform to robot coordinates
                    tvec = transform_camera_to_robot(tvec[0][0])
                    dist, phi = cart2pol(tvec[0], tvec[2])

                    # Process angle measurements
                    if ids[i][0] not in self.dict:
                        self.dict[ids[i][0]] = []
                    self.dict[ids[i][0]].append(phi)
                    
                    # Compute median angle if enough measurements
                    if len(self.dict[ids[i][0]]) >= 3:
                        phi5 = -np.median(np.sort(self.dict[ids[i][0]][-4:-1]))
                        self.dict[ids[i][0]].pop(0)
                        cv2.putText(cv_image, 'ang=' + str(round(phi5, 3)), 
                                    (int(marker_corners[1][0] - 70), int(marker_corners[1][1]) - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        if self.use_data_association:
                            self.current_aruco.append((dist, phi5, -1, marker_pixel_size))
                        else:
                            self.current_aruco.append((dist, phi5, ids[i][0], marker_pixel_size))
                    else:
                        if self.use_data_association:
                            self.current_aruco.append((dist, -phi, -1, marker_pixel_size))
                        else:
                            self.current_aruco.append((dist, -phi, ids[i][0], marker_pixel_size))

                    cv2.putText(cv_image, 'dist= ' + str(round(dist, 3)), 
                                (int(marker_corners[2][0] - 80), int(marker_corners[2][1]) + 45), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 2)

            # Buffer measurements if any detected
            if self.current_aruco:
                self.measurement_buffer.append({
                    'time': data.header.stamp.to_sec(),
                    'measurements': self.current_aruco.copy()
                })
                self.new_measurements_available = True

            cv2.imshow('Aruco Detection', cv_image)
            cv2.waitKey(3)

    def synchronized_update(self):
        """Perform synchronized SLAM update at fixed rate"""
        current_time = time.time()
        
        # Check if it's time for an update
        if current_time - self.last_update_time < self.min_update_interval:
            return
        
        with self.lock:
            # Check if we should update
            should_update = False
            
            # Case 1: New measurements available
            if self.new_measurements_available and self.measurement_buffer:
                should_update = True
            
            # Case 2: Significant motion accumulated (even without measurements)
            elif (self.accumulated_motion['linear'] > 0.05 or  # 5cm
                  self.accumulated_motion['angular'] > 0.1):    # ~5.7 degrees
                should_update = True
            
            # Skip update if robot is stationary and no new measurements
            if not self.robot_moving and not self.new_measurements_available:
                return
            
            if should_update:
                # Process accumulated odometry
                if self.last_processed_odom is not None:
                    self.my_slam.update_odometry(self.odom)
                    self.last_processed_odom = self.odom.copy()
                
                # Process buffered measurements
                if self.measurement_buffer:
                    # Use the most recent measurement set
                    latest_measurements = self.measurement_buffer[-1]['measurements']
                    self.my_slam.compute_slam(latest_measurements)
                    self.measurement_buffer.clear()
                
                # Reset flags and counters
                self.new_measurements_available = False
                self.accumulated_motion = {'linear': 0.0, 'angular': 0.0}
                self.last_update_time = current_time
                
                # Log enhanced metrics periodically
                if current_time - self.last_msp_eta_log > self.msp_eta_log_interval:
                    self.log_enhanced_metrics()
                    self.last_msp_eta_log = current_time
                
                # Log update info
                if self.callback_count % 100 == 0:
                    rospy.loginfo(f"Update #{self.callback_count}: Moving={self.robot_moving}, "
                                  f"Measurements={len(self.measurement_buffer)}")

    def log_enhanced_metrics(self):
        """Log enhanced metrics including MSP and ETA."""
        metrics = self.my_slam.get_current_metrics()
        
        rospy.loginfo("=== Enhanced SLAM Metrics ===")
        rospy.loginfo(f"MSP (Mean Squared Position): {metrics.get('current_msp', 0.0):.6f} m²")
        rospy.loginfo(f"ETA (Est. Trajectory Accuracy): {metrics.get('current_eta', 0.0):.2f}%")
        rospy.loginfo(f"ATE (Absolute Trajectory Error): {metrics.get('current_mpd', 0.0):.6f} m")
        rospy.loginfo(f"Ground Truth Trajectory Points: {metrics.get('ground_truth_trajectory_length', 0)}")
        rospy.loginfo(f"Estimated Trajectory Points: {metrics.get('trajectory_length', 0)}")
        rospy.loginfo("============================")

    def calibrate_camera(self):
        try:
            possible_paths = [
                "camera_calibration_outro_grupo.npz",
                os.path.expanduser("~/SA/catkin_ws/camera_calibration.npz"),
                "/home/dinisas/SA/catkin_ws/camera_calibration.npz"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    data = np.load(path)
                    self.camera_matrix = data['camera_matrix']
                    self.dist_coeffs = data['dist_coeffs']
                    rospy.loginfo("Calibration loaded successfully!")
                    return
                    
            raise FileNotFoundError("Could not find calibration file")
            
        except Exception as e:
            rospy.logerr(f"Error loading calibration: {e}")
            exit(1)

    def publish_tf(self):
        br = tf.TransformBroadcaster()
        br.sendTransform(
            (0.0, 0.0, 0.0),
            tf.transformations.quaternion_from_euler(0, 0, 0),
            rospy.Time.now(),
            "odom",
            "map"
        )
        
        best_particle = self.my_slam.get_best_particle()
        x, y, theta = best_particle.pose
        quaternion = tf.transformations.quaternion_from_euler(0, 0, theta - (np.pi))
        br.sendTransform(
            (-x, y, 0),
            quaternion,
            rospy.Time.now(),
            "base_link",
            "odom"
        )

    def log_performance_summary(self):
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            callback_rate = self.callback_count / elapsed_time
            metrics = self.my_slam.get_current_metrics()
            
            rospy.loginfo("=== Enhanced FastSLAM Performance Summary ===")
            rospy.loginfo(f"Runtime: {elapsed_time:.1f}s, Updates: {self.callback_count}")
            rospy.loginfo(f"Effective Update Rate: {callback_rate:.2f} Hz")
            rospy.loginfo(f"Target Update Rate: {self.update_rate:.2f} Hz")
            
            # Enhanced trajectory metrics
            rospy.loginfo(f"Current MSP: {metrics.get('current_msp', 0.0):.6f} m²")
            rospy.loginfo(f"Average MSP: {metrics.get('average_msp', 0.0):.6f} m²")
            rospy.loginfo(f"Current ETA: {metrics.get('current_eta', 0.0):.2f}%")
            rospy.loginfo(f"Average ETA: {metrics.get('average_eta', 0.0):.2f}%")
            
            if 'effective_particle_count' in metrics:
                diversity = metrics['effective_particle_count']
                rospy.loginfo(f"Particle Diversity: {diversity['n_eff_ratio']:.2%}")
            
            rospy.loginfo(f"Detection Rate: {metrics['detection_rate']:.1f}%")
            rospy.loginfo(f"Current RMSE: {metrics['current_rmse']:.3f}m")
            
            # Ground truth information
            rospy.loginfo(f"Ground Truth Trajectory Length: {metrics.get('ground_truth_trajectory_length', 0)}")
            rospy.loginfo(f"Estimated Trajectory Length: {metrics.get('trajectory_length', 0)}")

    def run(self):
        start_time = rospy.Time.now()
        last_performance_log = time.time()
        performance_log_interval = 60.0
        
        # Log initial ground truth information
        metrics = self.my_slam.get_current_metrics()
        rospy.loginfo("=== Ground Truth Information ===")
        rospy.loginfo(f"Ground Truth Markers: {metrics.get('total_ground_truth', 0)}")
        rospy.loginfo(f"Ground Truth Trajectory Points: {metrics.get('ground_truth_trajectory_length', 0)}")
        rospy.loginfo("================================")
        
        # Create a rate object for the main loop
        rate = rospy.Rate(self.update_rate * 2)  # Run at 2x update rate to ensure we don't miss updates
        
        while not rospy.is_shutdown():
            current_time = time.time()
            
            # Perform synchronized update
            self.synchronized_update()
            
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
                metrics_filename = f"slam_enhanced_metrics_{timestamp}.txt"
                self.my_slam.save_metrics_to_file(metrics_filename)
                rospy.loginfo(f"Enhanced metrics saved to {metrics_filename}")
                self.last_metrics_save = current_time
            
            # Check if rosbag finished
            if self.rosbag_finished(start_time, self.k):
                rospy.loginfo("Rosbag playback finished. Performing final enhanced analysis...")
                
                # Save final enhanced metrics
                final_metrics_filename = f"final_enhanced_slam_metrics_{int(time.time())}.txt"
                self.my_slam.save_metrics_to_file(final_metrics_filename)
                
                # Log final enhanced performance
                self.log_performance_summary()
                
                final_metrics = self.my_slam.get_current_metrics()
                rospy.loginfo("=== Final Enhanced SLAM Results ===")
                rospy.loginfo(f"Final ATE: {final_metrics.get('current_mpd', 0.0):.4f}m")
                rospy.loginfo(f"Final MSP: {final_metrics.get('current_msp', 0.0):.6f}m²")
                rospy.loginfo(f"Final ETA: {final_metrics.get('current_eta', 0.0):.2f}%")
                rospy.loginfo(f"Final RMSE: {final_metrics['current_rmse']:.4f}m")
                rospy.loginfo(f"Total Landmarks Mapped: {final_metrics['landmarks_detected']}")
                rospy.loginfo(f"Detection Success Rate: {final_metrics['detection_rate']:.1f}%")
                rospy.loginfo(f"Ground Truth vs Estimated Trajectory Points: {final_metrics.get('ground_truth_trajectory_length', 0)}/{final_metrics.get('trajectory_length', 0)}")
                
                rospy.signal_shutdown("Rosbag playback finished")
                break
            
            rate.sleep()
            
        cv2.destroyAllWindows()
        pygame.display.quit()
        pygame.quit()

    def rosbag_finished(self, start_time, duration):
        end_time = start_time + rospy.Duration.from_sec(duration)
        return rospy.Time.now() > end_time

    def get_trajectory(self):
        return self.my_slam.get_best_trajectory()