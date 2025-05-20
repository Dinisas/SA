import rospy
import numpy as np
import tf.transformations
import copy
import cv2
from visualization_msgs.msg import Marker, MarkerArray

    def odom_callback(self, data):
    """Process odometry data for SLAM"""
    with self.lock:  # Thread-safe operation
        # Extract position from odometry message
        x = data.pose.pose.position.x
        y = data.pose.pose.position.y
        
        # Extract orientation as quaternion
        xq = data.pose.pose.orientation.x
        yq = data.pose.pose.orientation.y
        zq = data.pose.pose.orientation.z
        wq = data.pose.pose.orientation.w
        quaternion = [xq, yq, zq, wq]
        
        # Update current odometry
        self.odom = [x, y, quaternion]
        
        # Initialize reference position on first callback
        if self.count == 0:
            self.tara = copy.deepcopy(self.odom)
            self.count += 1
        
        # Adjust odometry relative to starting position
        self.odom[0] -= self.tara[0]
        self.odom[1] -= self.tara[1]
        
        # Update SLAM with odometry
        self.my_slam.update_odometry(self.odom)

    #camera calibration extrinsic parameters
    def transform_camera_to_robot(self, tvec):
        """Transform camera coordinates to robot coordinates"""
        #extrinsic parametres of a camera mount in relation to the robot = rotation + translation

        #orientation of the camera relative to the robot
        # Define rotation matrix (3x3 matrix) based on Pioneer robot's camera mount
        R_cam_to_robot = np.array([
            [1, 0, 0],  
            [0, 1, 0],
            [0, 0, 1]
        ])

        #position of the camera relative to the robot
        #Define translation vector (3x1 matrix) based on Pionneer robot's camera mount
        T_cam_to_robot = np.array([0.076, 0, 0.103])  

        # Convert translation_vector to homogeneous coordinates
        tvec_hom = np.append(tvec, [1])
        
        # Create a extrinsic matrix from R and T (4x4 matrix)
        transform_matrix = np.eye(4)

        #rotation matrix component
        transform_matrix[:3, :3] = R_cam_to_robot

        #translation matrix component
        transform_matrix[:3, 3] = T_cam_to_robot
        
        # Apply the transformation
        tvec_robot_hom = np.dot(transform_matrix, tvec_hom)
        
        # Convert back to 3D coordinates
        tvec_robot = tvec_robot_hom[:3]
        
        return tvec_robot   
    
    def cart2pol(x, y):
    """Convert Cartesian coordinates to polar coordinates"""
    rho = np.sqrt(x**2 + y**2)  # Distance
    phi = np.arctan2(x, y)      # Angle in radians
    return rho, np.degrees(phi)  # Return distance and angle in degrees

    def image_callback(self, data):
    """Process image data to detect ArUco markers"""
    with self.lock:  # Thread-safe operation (image and odometry processing are happening in parallel, avoid interferences)
        self.current_aruco = []  # Reset current detections
        
        try:
            # Convert compressed image to OpenCV format
            cv_image = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
        except Exception as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return
        
        # Convert to grayscale for ArUco detection (grayscale is better for aruco detection)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Detect ArUco markers
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
        
        # Create visualization markers
        marker_array = MarkerArray()

        # Process detected markers
        if ids is not None and len(ids) > 0:
            # Draw detected markers on image
            cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
            
            # Process each marker
            for i in range(len(ids)):
                marker_id = ids[i][0]
                marker_corners = corners[i][0]
                
                # Draw bounding box around marker
                cv2.polylines(cv_image, [np.int32(marker_corners)], True, (0, 255, 0), 2)
                
                # Estimate pose of marker (returns rotation and translation vectors)
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners[i], 0.16, self.camera_matrix, self.dist_coeffs)
                
                # Transform position from camera to robot coordinate frame
                tvec = transform_camera_to_robot(tvecs[0][0])

                 # Calculate distance and bearing angle
                dist, phi = cart2pol(tvec[0], tvec[2])
                
                # Initialize/update angle history for this marker
                if marker_id not in self.dict:
                    self.dict[marker_id] = []
                self.dict[marker_id].append(phi)
                
                # Draw reference point at image center
                cv2.circle(cv_image, (320, 240), radius=10, color=(255, 0, 0), thickness=-1)
                
                # Display distance on the image
                cv2.putText(cv_image, f'dist={dist:.3f}m', 
                           (int(marker_corners[2][0] - 80), int(marker_corners[2][1]) + 45), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Smooth angle using median filter if enough history
                if len(self.dict[marker_id]) >= 3:
                    phi_smooth = -np.median(np.sort(self.dict[marker_id][-4:-1]))
                    self.dict[marker_id].pop(0)  # Remove oldest measurement
                    
                    # Display smoothed angle
                    cv2.putText(cv_image, f'ang={phi_smooth:.3f}°', 
                               (int(marker_corners[1][0] - 70), int(marker_corners[1][1]) - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    # Add to current detections list
                    self.current_aruco.append((dist, phi_smooth, marker_id))
                else:
                    # Use raw angle if not enough history
                    self.current_aruco.append((dist, -phi, marker_id))
                
                # Draw axes to show marker orientation
                cv2.drawFrameAxes(cv_image, self.camera_matrix, self.dist_coeffs, 
                                 rvecs[0], tvecs[0], 0.1)
                
                # Create visualization marker
                viz_marker = Marker()
                viz_marker.header.frame_id = "base_link"
                viz_marker.header.stamp = rospy.Time.now()
                viz_marker.ns = "aruco_markers"
                viz_marker.id = marker_id
                viz_marker.type = Marker.CUBE
                viz_marker.action = Marker.ADD
                viz_marker.pose.position.x = tvec[0]
                viz_marker.pose.position.y = tvec[1]
                viz_marker.pose.position.z = tvec[2]
                
                # Convert rotation to quaternion
                rot_matrix, _ = cv2.Rodrigues(rvecs[0])
                rot_matrix_3x3 = np.eye(4)
                rot_matrix_3x3[:3, :3] = rot_matrix
                quat = tf.transformations.quaternion_from_matrix(rot_matrix_3x3)
                
                viz_marker.pose.orientation.x = quat[0]
                viz_marker.pose.orientation.y = quat[1]
                viz_marker.pose.orientation.z = quat[2]
                viz_marker.pose.orientation.w = quat[3]
                
                viz_marker.scale.x = 0.16  # Marker size
                viz_marker.scale.y = 0.16
                viz_marker.scale.z = 0.01
                viz_marker.color.r = 0.0
                viz_marker.color.g = 1.0
                viz_marker.color.b = 0.0
                viz_marker.color.a = 0.7
                
                marker_array.markers.append(viz_marker)
                
                # Broadcast TF for this marker
                self.tf_broadcaster.sendTransform(
                    (tvec[0], tvec[1], tvec[2]),
                    quat,
                    rospy.Time.now(),
                    f"aruco_marker_{marker_id}",
                    "base_link"
                )
                
        # Publish visualization markers
        self.landmark_pub.publish(marker_array)
        
        # Run SLAM update with detected markers
        if self.current_aruco:
            self.my_slam.compute_slam(self.current_aruco)
        
        # Display the processed image
        cv2.imshow('ArUco Detection', cv_image)
        cv2.waitKey(1)