#!/usr/bin/env python3
import os
import rospkg
import rospy

def list_groundtruth_files():
    """List all available groundtruth YAML files"""
    rospack = rospkg.RosPack()
    path = rospack.get_path('pioneer_fast_slam')
    groundtruth_dir = os.path.join(path, 'src', 'groundtruth')
    
    rospy.loginfo("=" * 60)
    rospy.loginfo("Available groundtruth files:")
    rospy.loginfo("=" * 60)
    
    if os.path.exists(groundtruth_dir):
        yaml_files = [f for f in os.listdir(groundtruth_dir) if f.endswith('.yaml')]
        for i, yaml_file in enumerate(sorted(yaml_files)):
            rospy.loginfo(f"  {i+1}. {yaml_file}")
    
    rospy.loginfo("=" * 60)
    rospy.loginfo("To use a specific groundtruth file, run:")
    rospy.loginfo("  roslaunch pioneer_fast_slam aruco_slam.launch groundtruth:=<filename>")
    rospy.loginfo("Example:")
    rospy.loginfo("  roslaunch pioneer_fast_slam aruco_slam.launch groundtruth:=trajectory_x.yaml")
    rospy.loginfo("=" * 60)

if __name__ == '__main__':
    rospy.init_node('list_groundtruth', anonymous=True)
    list_groundtruth_files()