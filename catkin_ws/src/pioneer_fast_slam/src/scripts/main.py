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
def set_slam_variables(particles):
    window_size_pixel = 1500    # pixel size of window
    size_window = 30  # in meters
    number_particles = particles
    # tuning parameters
    # Q_init=np.diag([0.1,0.1])
    # Q_update=np.diag([0.7,0.7])
    alphas=[0.00008,0.00008,0.00001,0.00001]
    # can add to tuning options to use static instead of dynamic Q_init, Q_update,
    tuning_option = [ alphas]
    return (window_size_pixel, size_window, number_particles, tuning_option)

# Function to run the SLAM process
def run_slam(rosbag_file, particles):
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
        
        # get defined parameters
        slam_variables = set_slam_variables(particles)
        
        # call ArucoSlam class
        rospy.loginfo("Creating ArucoSLAM instance...")
        slam = ArucoSLAM(rosbag_time, slam_variables)
        
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
        
        # Find the rosbag file path
        rospack = rospkg.RosPack()
        path = rospack.get_path('pioneer_fast_slam')
        rospy.loginfo(f'Package path: {path}')
        
        rosbag_file = f"{path}/src/rosbag/{file}"
        rospy.loginfo(f'Full rosbag path: {rosbag_file}')

        # Get number of particles from launch file parameter
        particles = int(rospy.get_param('~particles', 25))
        rospy.loginfo(f'Number of Particles: {particles}')
        
        # Run the SLAM process
        run_slam(rosbag_file, particles)
        
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS was interrupted")
    except Exception as e:
        rospy.logerr(f"Unexpected error in main: {e}")
        traceback.print_exc()
        sys.exit(1)