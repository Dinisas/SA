#!/usr/bin/env python3
from aruco_slam import ArucoSLAM
import rospy
import rospkg
import os
import re
import numpy as np
from utils import get_rosbag_duration

# Function to set SLAM variables
def set_slam_variables():
    window_size_pixel = 1500    # pixel size of window
    size_window = 30  # in meters
    number_particles = 25
    Q_init = np.diag([0.1, 0.1])
    Q_update = np.diag([0.7, 0.7])
    alphas = [0.00008, 0.00008, 0.00001, 0.00001]
    tuning_option = [Q_init, Q_update, alphas]
    return (window_size_pixel, size_window, number_particles, tuning_option)

# Function to run the SLAM process
def run_slam(rosbag_file):
    rosbag_process = None
    try:
        if not os.path.isfile(rosbag_file):
            print(f"ERROR: The file {rosbag_file} does not exist.")
            exit(1)
            
        rosbag_time = get_rosbag_duration(rosbag_file)
        slam_variables = set_slam_variables()
        slam = ArucoSLAM(rosbag_time, slam_variables)
        slam.run()
        print(f"SLAM completed for {rosbag_file}")
        
    finally:
        if rosbag_process:
            rosbag_process.terminate()

# Main execution block
if __name__ == '__main__':
    try:
        # Get the rosbag file parameter
        file = rospy.get_param('~rosbag', 'odometry1.bag')
        rospy.loginfo(f'Rosbag file: {file}')
        
        # Find the rosbag file path
        rospack = rospkg.RosPack()
        path = rospack.get_path('pioneer_fast_slam')
        rosbag_file = f"{path}/src/rosbag/{file}"
        
        # Extract map number from filename if present
        match = re.search(r"\d", file)
        if match:
            map_nr = int(match.group(0))
            print(f"Map found: {map_nr}")
        else:
            print("No map number found in filename")
        
        # Run the SLAM process
        run_slam(rosbag_file)
        
    except rospy.ROSInterruptException:
        print("ROS was interrupted")