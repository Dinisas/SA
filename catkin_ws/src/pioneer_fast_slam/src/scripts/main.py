#!/usr/bin/env python3
from aruco_slam import ArucoSLAM
import rospy
import rospkg
import os
import re
import numpy as np
from utils import get_rosbag_duration
import traceback
import sys

# Function to set SLAM variables
def set_slam_variables(particles, groundtruth_file, update_rate):
    print(f"DEBUG: set_slam_variables called with particles = {particles}, update_rate = {update_rate}Hz")
    window_size_pixel = 1500    # pixel size of window
    size_window = 30  # in meters
    number_particles = particles
    groundtruth_file = groundtruth_file
    
    # Adaptive tuning parameters based on update rate
    # Lower update rates need less aggressive motion noise
    noise_scale = 30.0 / update_rate  # Scale noise based on update rate
    
    # Tuning parameters - adjusted for better performance
    alphas = [
        0.00004 * noise_scale,  # Rotation noise from rotation
        0.00004 * noise_scale,  # Rotation noise from translation  
        0.000005 * noise_scale,  # Translation noise from translation
        0.000005 * noise_scale   # Translation noise from rotation
    ]
    
    tuning_option = [alphas]
    return (window_size_pixel, size_window, number_particles, tuning_option, groundtruth_file)

# Function to run the SLAM process
def run_slam(rosbag_file, particles, groundtruth_file, update_rate):
    print(f"DEBUG: run_slam called with particles = {particles}, update_rate = {update_rate}Hz")
    rosbag_process = None
    try:
        if not os.path.isfile(rosbag_file):
            rospy.logerr(f"ERROR: The file {rosbag_file} does not exist.")
            rospy.logerr(f"Looking in: {os.path.dirname(rosbag_file)}")
            rospy.logerr(f"Available files: {os.listdir(os.path.dirname(rosbag_file)) if os.path.exists(os.path.dirname(rosbag_file)) else 'Directory does not exist'}")
            return
        
        rospy.loginfo(f"Found rosbag file: {rosbag_file}")
        
        # get rosbag duration
        rosbag_time = get_rosbag_duration(rosbag_file)
        rospy.loginfo(f"Rosbag duration: {rosbag_time} seconds")
        
        # Display SLAM configuration
        rospy.loginfo("=== FastSLAM Configuration ===")
        rospy.loginfo(f"Number of particles: {particles}")
        rospy.loginfo(f"Update rate: {update_rate} Hz")
        rospy.loginfo(f"Ground truth file: {groundtruth_file}")
        rospy.loginfo("==============================")
        
        # get defined parameters
        slam_variables = set_slam_variables(particles, groundtruth_file, update_rate)
        
        # Modify ArucoSLAM to accept update_rate
        # Since we can't modify the ArucoSLAM constructor directly, 
        # we'll set it as a ROS parameter that ArucoSLAM can read
        rospy.set_param('~slam_update_rate', update_rate)
        
        # call ArucoSlam class
        rospy.loginfo("Creating ArucoSLAM instance...")
        slam = ArucoSLAM(rosbag_time, slam_variables)
        
        # Override the update rate if the class supports it
        if hasattr(slam, 'update_rate'):
            slam.update_rate = update_rate
            slam.min_update_interval = 1.0 / update_rate
            rospy.loginfo(f"Set SLAM update rate to {update_rate} Hz")
        
        # play run function from ArucoSlam class
        rospy.loginfo("Starting SLAM run...")
        slam.run()
        rospy.loginfo(f"SLAM completed for {rosbag_file}")
        
    except FileNotFoundError as e:
        rospy.logerr(f"File not found error: {e}")
        traceback.print_exc()
    except Exception as e:
        rospy.logerr(f"Error during SLAM execution: {e}")
        rospy.logerr(f"Error type: {type(e).__name__}")
        traceback.print_exc()
    finally:
        if rosbag_process:
            rosbag_process.terminate()

# Main execution block
if __name__ == '__main__':
    try:
        # Initialize ROS node first (before any ROS operations)
        rospy.init_node('aruco_slam_main', anonymous=True)
        
        # Get the rosbag file parameter from the launch file
        file = rospy.get_param('~rosbag', 'simulation1.bag')
        rospy.loginfo(f'Rosbag file parameter: {file}')

        # Get groundtruth file parameter from the launch file
        yaml = rospy.get_param("~groundtruth", "simulation3.yaml")
        rospy.loginfo(f'Groundtruth file parameter: {yaml}')
        
        # Get number of particles from launch file parameter
        particles = int(rospy.get_param('~particles', 50))
        rospy.loginfo(f'Number of Particles: {particles}')
        
        # Get update rate from launch file parameter
        update_rate = float(rospy.get_param('~update_rate', 30.0))
        rospy.loginfo(f'Target Update Rate: {update_rate} Hz')
        
        # Find the rosbag file path
        rospack = rospkg.RosPack()
        path = rospack.get_path('pioneer_fast_slam')
        rospy.loginfo(f'Package path: {path}')
        
        # Construct the full paths
        rosbag_file = f"{path}/src/rosbag/{file}"
        rospy.loginfo(f'Full rosbag path: {rosbag_file}')
        
        # Get groundtruth file path
        groundtruth_file = f"{path}/src/groundtruth/{yaml}"
        rospy.loginfo(f'Full groundtruth path: {groundtruth_file}')
        
        print(f"DEBUG: About to call run_slam with particles = {particles}, update_rate = {update_rate}")
        
        # Run the SLAM process
        run_slam(rosbag_file, particles, groundtruth_file, update_rate)
        
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS was interrupted")
    except Exception as e:
        rospy.logerr(f"Unexpected error in main: {e}")
        traceback.print_exc()
        sys.exit(1)