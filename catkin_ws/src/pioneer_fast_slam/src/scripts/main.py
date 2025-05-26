#!/usr/bin/env python3
from aruco_slam import ArucoSLAM
import rospy
import rospkg
import os
import re
import numpy as np
from utils import get_rosbag_duration

# Function to set SLAM variables
def set_slam_variables(particles):
    window_size_pixel = 1500    # pixel size of window
    size_window = 30  # in meters
    number_particles = particles
    # tuning parameters
    Q_init=np.diag([0.1,0.1])
    Q_update=np.diag([0.7,0.7])
    alphas=[0.00008,0.00008,0.00001,0.00001]
    tuning_option = [Q_init, Q_update, alphas]
    return (window_size_pixel, size_window, number_particles, tuning_option)

# Function to run the SLAM process
def run_slam(rosbag_file, particles):
    rosbag_process = None
    try:
        if not os.path.isfile(rosbag_file):
            print(f"ERROR: The file {rosbag_file} does not exist.")
            exit(1)
        
        # get rosbag duration
        rosbag_time = get_rosbag_duration(rosbag_file)
        # get defined parameters
        slam_variables = set_slam_variables(particles)
        # call ArucoSlam class
        slam = ArucoSLAM(rosbag_time, slam_variables)
        # play run function from ArucoSlam class
        slam.run()
        print(f"SLAM completed for {rosbag_file}")
        
    finally:
        if rosbag_process:
            rosbag_process.terminate()

# Main execution block
if __name__ == '__main__':
    try:
        # Get the rosbag file parameter from the launch file
        file = rospy.get_param('~rosbag', 'simulation1.bag')
        rospy.loginfo(f'Rosbag file: {file}')
        
        # Find the rosbag file path
        rospack = rospkg.RosPack()
        path = rospack.get_path('pioneer_fast_slam')
        rosbag_file = f"{path}/src/rosbag/{file}"

        # Get number of particles from launch file parameter
        particles=int(rospy.get_param('~particles',25))
        rospy.loginfo(f'Number of Particles: {particles}')
        
        # Run the SLAM process
        run_slam(rosbag_file, particles)
        
    except rospy.ROSInterruptException:
        print("ROS was interrupted")