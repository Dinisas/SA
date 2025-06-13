# FastSLAM on Pioneer 3DX Robot

A comprehensive implementation of the FastSLAM algorithm on a Pioneer 3DX robot using ArUco markers as landmarks. This project features real-time SLAM with enhanced metrics tracking, visualization, and performance monitoring.

![FastSLAM Demo](docs/images/fastslam_demo.gif)

## 🎯 Project Overview

This project implements the FastSLAM algorithm with the following key features:

- **Real-time SLAM** using particle filtering and Extended Kalman Filters
- **ArUco marker detection** for landmark-based localization
- **Comprehensive metrics tracking** including ATE, MSP, RMSE, and particle diversity
- **Kabsch alignment** for accurate map evaluation
- **Dynamic measurement covariance** adapting to marker visibility conditions
- **Enhanced visualization** with RViz integration and real-time performance monitoring
- **Micro-simulation environment** for algorithm testing and validation

## 🛠 Technical Architecture

### Core Components

- **Particle Filter**: Tracks multiple robot pose hypotheses with configurable particle count
- **Extended Kalman Filters**: Individual EKFs for each landmark position estimation
- **Motion Model**: Probabilistic differential drive model with adaptive noise
- **Measurement Model**: Camera-based ArUco detection with dynamic covariance
- **Resampling Strategy**: Low-variance resampling with particle diversity preservation

### Performance Features

- **Real-Time Factor (RTF)**: Monitoring for deployment readiness
- **Effective Particle Count**: Automatic diversity tracking and resampling triggers
- **Computational Profiling**: Detailed timing analysis for optimization
- **Ground Truth Comparison**: Multiple trajectory pattern evaluation

## 📁 Repository Structure

```
catkin_ws/src/pioneer_fast_slam/
├── launch/
│   ├── aruco_slam.launch          # Main SLAM launch file
│   └── record_data.launch         # Data recording configuration
├── src/
│   ├── scripts/
│   │   ├── main.py               # Entry point and configuration
│   │   ├── aruco_slam.py         # Main SLAM node implementation
│   │   ├── fast_slam.py          # Core FastSLAM algorithm
│   │   ├── particle.py           # Particle class with EKF updates
│   │   ├── landmark.py           # Landmark representation
│   │   ├── metrics.py            # Comprehensive metrics tracking
│   │   └── utils.py              # Utility functions and transformations
│   ├── groundtruth/
│   │   ├── trajectory_x.yaml     # X-pattern trajectory ground truth
│   │   ├── trajectory_o.yaml     # O-pattern trajectory ground truth
│   │   └── simulation3.yaml      # Marker positions ground truth
│   ├── rviz/
│   │   └── slam.rviz             # RViz configuration
│   └── rosbag/                   # Recorded datasets (ignored)
├── microSimulation/
│   └── simulation.py             # Pygame-based micro-simulator
├── CMakeLists.txt
├── package.xml
└── README.md
```

## 🚀 Quick Start

### Prerequisites

```bash
# Ubuntu 20.04 LTS with ROS Noetic
sudo apt update
sudo apt install ros-noetic-desktop-full

# Required Python packages
pip install pygame numpy scipy opencv-python
sudo apt install ros-noetic-cv-bridge ros-noetic-image-transport
```

### Installation

1. **Clone the repository**:
```bash
cd ~/catkin_ws/src
git clone https://github.com/yourusername/pioneer-fastslam.git
cd ..
catkin_make
source devel/setup.bash
```

2. **Camera calibration**:
```bash
# Place your camera calibration file as:
# ~/catkin_ws/camera_calibration.npz
```

3. **Test with micro-simulation**:
```bash
cd microSimulation
python simulation.py
```

### Usage

#### Real Robot Operation

1. **Connect to Pioneer 3DX**:
```bash
# Terminal 1: Connect to robot
ssh pi@192.168.28.XX
roscore

# Terminal 2: Start robot drivers
ssh pi@192.168.28.XX
rosrun p2os_driver p2os_driver _port:="/dev/ttyUSB0"

# Terminal 3: Set environment and launch SLAM
export ROS_MASTER_URI=http://192.168.28.XX:11311
export ROS_IP=192.168.28.YY
roslaunch pioneer_fast_slam aruco_slam.launch
```

2. **Configure parameters**:
```bash
# Launch with custom settings
roslaunch pioneer_fast_slam aruco_slam.launch \
    particles:=50 \
    groundtruth:=trajectory_x.yaml \
    update_rate:=30
```

#### Data Recording

```bash
# Record new datasets
roslaunch pioneer_fast_slam record_data.launch

# Process recorded data
roslaunch pioneer_fast_slam aruco_slam.launch \
    rosbag:=your_dataset.bag \
    particles:=25
```

## 🙏 Acknowledgments

- **Autonomous Systems Course** - Instituto Superior Técnico
- **FastSLAM Algorithm** - Original work by Montemerlo et al.
- **OpenCV Community** - ArUco marker detection
- **ROS Community** - Robot Operating System framework

---

**Authors**: Dinis Alves da Silva, Henrique Rodrigues, Joana Teixeira, Mafalda Lopes  
**Institution**: Instituto Superior Técnico, Universidade de Lisboa  
**Course**: Autonomous Systems 2024/2025
