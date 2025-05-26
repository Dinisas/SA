cmake_minimum_required(VERSION 3.0.2)
project(pioneer_fast_slam)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  nav_msgs
  visualization_msgs
  tf
)

## Declare a catkin package
catkin_package(
  CATKIN_DEPENDS 
    rospy 
    std_msgs 
    geometry_msgs 
    sensor_msgs 
    nav_msgs 
    visualization_msgs 
    tf
)

## Include directories
include_directories(
  ${catkin_INCLUDE_DIRS}
)

## DEVELOPMENT: Create symlinks instead of wrapper scripts
# This preserves Python imports between modules
set(SCRIPTS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/scripts)
set(DEVEL_SCRIPTS_DIR ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_BIN_DESTINATION})

# Create the directory
file(MAKE_DIRECTORY ${DEVEL_SCRIPTS_DIR})

# Create symlinks for each Python script
foreach(script 
    main.py 
    aruco_slam.py 
    fast_slam.py 
    particle.py 
    landmark.py 
    utils.py)
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink
    ${SCRIPTS_DIR}/${script}
    ${DEVEL_SCRIPTS_DIR}/${script}
  )
endforeach()

## DEPLOYMENT ONLY: Uncomment this when you need to install
# catkin_install_python(PROGRAMS
#   src/scripts/main.py
#   src/scripts/aruco_slam.py
#   src/scripts/fast_slam.py
#   src/scripts/particle.py
#   src/scripts/landmark.py
#   src/scripts/utils.py
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Install launch files
install(DIRECTORY src/launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  FILES_MATCHING PATTERN "*.launch"
)

## Install rviz configuration files
install(DIRECTORY src/rviz/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/rviz
  FILES_MATCHING PATTERN "*.rviz"
)

## Install rosbag directory (optional, for file structure)
install(DIRECTORY src/rosbag/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/rosbag
  PATTERN "*.bag*"
)