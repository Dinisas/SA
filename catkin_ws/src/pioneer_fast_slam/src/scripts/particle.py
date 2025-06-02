import numpy as np
import math
from landmark import Landmark
from utils import normalize_angle

class Particle:
    """ 
    Particle class representing a single particle in the particle filter.
    Attributes:
        pose: The current pose (x, y, theta) of the particle.
        landmarks: A dictionary of landmarks observed by this particle.
        weight: The weight of the particle.
    """
        
    def __init__(self, pose, nr_particles, pioneer_L, tuning_option):
        # initialize particle pose, landmarks, weight, tuning options
        self.pose = pose
        self.landmarks = {}
        self.weight = 1.0
        self.pioneer_L = pioneer_L
        self.observation_vector = np.zeros((2, 1))
        self.default_weight = 1 / nr_particles
        # jacobian matrix placeholder
        self.J_matrix = np.zeros((2, 2))
        # covariance matrix placeholder
        self.adjusted_covariance = np.zeros((2, 2))
        self.trajectory = []
        self.next_virtual_id = 0
        self.virtual_id_assignments = {}
        # tuning options
        # self.Q_init = tuning_option[0]
        # self.Q_update = tuning_option[1]
        #change index to 2, in case of texting with static Q
        self.alphas = tuning_option[0]


    def assign_virtual_id(self):
        """Generate a new virtual ID for a newly detected marker"""
        virtual_id = f"virtual_{self.next_virtual_id}"
        self.next_virtual_id += 1
        return virtual_id

    ## MOTION MODEL ## (particle prediction step based on motion model)
    def motion_model(self, odometry_delta):
        """ 
        This function updates the particle's pose based on odometry (motion model).
        Args:
            odometry_delta: Tuple containing (delta_dist, delta_rot1, delta_rot2).
        """
        # get current pose
        x, y, theta = self.get_pose()
        # add current pose to trajectory
        self.trajectory.append((x, y, theta))
        # unpack motion deltas
        delta_dist, delta_rot1, delta_rot2 = odometry_delta

        # Parameters for motion noise
        alpha1 = self.alphas[0]
        alpha2 = self.alphas[1]
        alpha3 = self.alphas[2]
        alpha4 = self.alphas[3]

        # Calculate deviations with noise from alpha tuning parameters
        deviation_dist = math.sqrt(alpha1 * delta_rot1**2 + alpha2 * delta_dist**2)
        deviation_rot1 = math.sqrt(alpha3 * delta_dist**2 + alpha4 * delta_rot1**2 + alpha4 * delta_rot2**2)
        deviation_rot2 = math.sqrt(alpha1 * delta_rot2**2 + alpha2 * delta_dist**2)

        # add gaussian noise to distance and first/second rotations
        delta_dist -= np.random.normal(0, deviation_dist)
        delta_rot1 -= np.random.normal(0, deviation_rot1)
        delta_rot2 -= np.random.normal(0, deviation_rot2)
        
        # Update the pose (with new orientation and position)
        new_x = x + delta_dist * math.cos(theta + delta_rot1)
        new_y = y - delta_dist * math.sin(theta + delta_rot1)
        new_theta = normalize_angle(theta + delta_rot1 + delta_rot2)
        self.pose = np.array([new_x, new_y, new_theta])

    def get_dynamic_measurement_covariance(self, distance_meters, marker_pixel_size):
        """
        Calculate realistic measurement covariance for camera-based SLAM.
        
        Args:
            distance_meters: Distance to the landmark
            marker_pixel_size: Size of marker in pixels
        
        Returns:
            2x2 covariance matrix
        """
        # Realistic camera uncertainty: 3-5% of distance
        sigma_r = 0.03 * distance_meters + 0.02  # 3% of distance + 2cm base
        
        # Bearing uncertainty: 2-5 degrees is typical for cameras
        sigma_theta = np.radians(3.0)  # 3 degrees base uncertainty
        
        # Ensure minimum values
        sigma_r = max(sigma_r, 0.05)  # At least 5cm
        sigma_theta = max(sigma_theta, np.radians(2.0))  # At least 2 degrees
        
        # SLAM scaling factors (moderate, not too aggressive)
        slam_range_factor = 2.0
        slam_bearing_factor = 2.0
        
        sigma_r *= slam_range_factor
        sigma_theta *= slam_bearing_factor
        
        # Final minimums after scaling
        sigma_r = max(sigma_r, 0.10)  # At least 10cm
        sigma_theta = max(sigma_theta, np.radians(3.0))  # At least 3 degrees
        
        # Create covariance matrix
        Q = np.array([[sigma_r**2, 0.0],
                    [0.0, sigma_theta**2]])
        return Q

    def predict_measurement_for_landmark(self, landmark):
        """
        Predict the expected range and bearing measurement for a landmark.
        
        Args:
            landmark: Landmark object
            
        Returns:
            tuple: (predicted_range, predicted_bearing)
        """
        x, y, theta = self.pose
        dx = landmark.x - x
        dy = landmark.y - y
        
        # Predicted range
        predicted_range = math.sqrt(dx**2 + dy**2)
        
        # Predicted bearing (consistent with your measurement convention)
        predicted_bearing = -math.atan2(dy, dx) - theta
        predicted_bearing = normalize_angle(predicted_bearing)
        
        return predicted_range, predicted_bearing

    def calculate_mahalanobis_distance(self, measured_range, measured_bearing, landmark, marker_pixel_size):
        """
        Calculate Mahalanobis distance for data association.
        
        Args:
            measured_range: Measured distance to landmark
            measured_bearing: Measured bearing to landmark
            landmark: Landmark object to compare against
            marker_pixel_size: Size of marker in pixels
            
        Returns:
            float: Mahalanobis distance
        """
        # Get predicted measurement
        predicted_range, predicted_bearing = self.predict_measurement_for_landmark(landmark)
        
        # Calculate innovation
        innovation_range = measured_range - predicted_range
        innovation_bearing = normalize_angle(measured_bearing - predicted_bearing)
        innovation = np.array([innovation_range, innovation_bearing])
        
        # Calculate Jacobian
        x, y, theta = self.pose
        dx = landmark.x - x
        dy = landmark.y - y
        q = dx**2 + dy**2
        sqrt_q = math.sqrt(q)
        
        if sqrt_q < 0.001:
            return float('inf')
        
        J = np.array([[dx / sqrt_q, dy / sqrt_q, 0],
                    [dy / q, -dx / q, -1]])
        
        # Get measurement covariance
        Q = self.get_dynamic_measurement_covariance(measured_range, marker_pixel_size)
        
        # Innovation covariance: S = J*P*J' + Q
        S = J @ landmark.sigma @ J.T + Q
        
        # Add small regularization for numerical stability
        S += np.eye(2) * 1e-6
        
        try:
            # Use Cholesky decomposition for stability
            L = np.linalg.cholesky(S)
            y = np.linalg.solve(L, innovation)
            mahalanobis_dist = math.sqrt(np.dot(y, y))
            return mahalanobis_dist
        except np.linalg.LinAlgError:
            # Fallback to standard inverse
            try:
                S_inv = np.linalg.inv(S)
                mahalanobis_dist = math.sqrt(innovation.T @ S_inv @ innovation)
                return mahalanobis_dist
            except:
                return float('inf')

    def find_best_landmark_association(self, measured_range, measured_bearing, marker_pixel_size):
        """
        Find the best landmark to associate with a measurement.
        
        Args:
            measured_range: Measured distance
            measured_bearing: Measured bearing angle
            marker_pixel_size: Size of marker in pixels
            
        Returns:
            tuple: (best_landmark_id, min_distance)
        """
        if not self.landmarks:
            return None, float('inf')
        
        min_distance = float('inf')
        best_landmark_id = None
        
        for landmark_id, landmark in self.landmarks.items():
            # Quick distance check for optimization
            rough_dist = landmark.distance_to(self.pose[0], self.pose[1])
            if rough_dist > measured_range + 3.0:  # 3m tolerance
                continue
            
            # Calculate Mahalanobis distance
            distance = self.calculate_mahalanobis_distance(
                measured_range, measured_bearing, landmark, marker_pixel_size
            )
            
            if distance < min_distance:
                min_distance = distance
                best_landmark_id = landmark_id
        
        return best_landmark_id, min_distance

    def handle_landmark_with_id(self, landmark_dist, landmark_bearing_angle, landmark_id, marker_pixel_size):
        """
        Handle landmark with known ID (correspondence problem).
        
        Args:
            landmark_dist: Distance to landmark
            landmark_bearing_angle: Bearing angle to landmark (radians)
            landmark_id: Known landmark ID
            marker_pixel_size: Size of marker in pixels
        """
        if landmark_id not in self.landmarks:
            self.create_landmark(landmark_dist, landmark_bearing_angle, landmark_id, marker_pixel_size)
        else:
            self.update_landmark(landmark_dist, landmark_bearing_angle, landmark_id, marker_pixel_size)

    def handle_landmark(self, landmark_dist, landmark_bearing_angle, landmark_id, marker_pixel_size):
        """
        Main landmark handling method - routes to appropriate handler.
        
        This method is called by the old compute_slam for backwards compatibility.
        """
        # For data association mode (landmark_id == -1), this shouldn't be called
        # as the particle filter handles association at a higher level
        if landmark_id == -1:
            raise ValueError("Data association should be handled at particle filter level")
        
        # For correspondence mode, use the ID-based handler
        self.handle_landmark_with_id(landmark_dist, landmark_bearing_angle, str(landmark_id), marker_pixel_size)

    def create_landmark(self, distance, angle, landmark_id, marker_pixel_size):
        """
        Create a new landmark in the particle's map.
        Args:
            distance: Distance to the landmark.
            angle: Bearing angle to the landmark.
            landmark_id: Identifier of the landmark.
        """
        # get particle pose and update landmark position accordingly
        x, y, theta = self.get_pose()
        #bearing angle (angle): how much the robot is rotated from the robot's forward direction
        #particle orientation (theta): how much the robot is rotated from global frame
        # create marker from observation
        landmark_x = x + distance * math.cos(theta + angle)
        landmark_y = y - distance * math.sin(theta + angle)
        self.landmarks[landmark_id] = Landmark(landmark_x, landmark_y)
        
        # Landmark prediction of the measurement based on the motion model(predicted distance and bearing angle to the landmark)
        #Rk (process noise covariance matrix) is zero, landmarks are static
        dx = landmark_x - x
        dy = landmark_y - y
        predicted_distance = math.sqrt(dx**2 + dy**2)
        predicted_angle = math.atan2(dy, dx) - theta
        
        # Normalize the angle between -pi and pi
        predicted_angle = (predicted_angle + np.pi) % (2 * np.pi) - np.pi
        
        # Measurement noise covariance matrix (tunes itself dynamically based on distance and pixel error) -> Landmark correction step based on measurement model

        # Calculate Jacobian matrix H of the measurement function for the EKF
        q = dx**2 + dy**2
        sqrt_q = math.sqrt(q)

        J = np.array([[dx / sqrt_q, dy / sqrt_q, 0],
                      [-dy / q, dx / q, -1]])

        # Q = self.Q_init  # Example values, Q is Q_t in the book
        Q = self.get_dynamic_measurement_covariance(distance, marker_pixel_size)
        # innovation covariance
        S = J @ self.landmarks[landmark_id].sigma @ J.T + Q
        # kalman gain
        K = self.landmarks[landmark_id].sigma @ J.T @ np.linalg.inv(S)
        # update covariance
        self.landmarks[landmark_id].sigma = (np.eye(3) - K @ J) @ self.landmarks[landmark_id].sigma
                
        # Set a default importance weight
        self.weight = self.default_weight  # p0 in the book

    def update_landmark(self, distance, angle, landmark_id, marker_pixel_size):
        """
        Updates an existing landmark using the EKF update step.
        Args:
            distance: Measured distance to the landmark.
            angle: Measured bearing angle to the landmark.
            landmark_id: Identifier of the landmark.
        """
        landmark = self.landmarks[str(landmark_id)]
        x, y, theta = self.pose
 
        # Landmark prediction of the measurement based on motion model
        #Rk (process noise covariance matrix) is zero, landmarks are static
        dx = landmark.x - x
        dy = landmark.y - y
        predicted_distance = math.sqrt(dx**2 + dy**2)
        predicted_angle = -math.atan2(dy, dx) - theta
            
        # Normalize the angle between -pi and pi
        predicted_angle = (predicted_angle + np.pi) % (2 * np.pi) - np.pi
            
        # Measurement noise covariance matrix (tunes itself dynamically based on distance and pixel error) -> correction step based on measuremet model

        # 1)Calculate Jacobian matrix H of the measurement function
        q = dx**2 + dy**2
        sqrt_q = math.sqrt(q)
            
        J = np.array([[dx / sqrt_q, dy / sqrt_q, 0],
                      [dy / q, -dx / q, -1]])

        # Q = self.Q_update  # Example values
        Q = self.get_dynamic_measurement_covariance(distance, marker_pixel_size)

        # 2)Calculate the Kalman Gain
        S = J @ landmark.sigma @ J.T + Q  # Measurement prediction covariance
        K = landmark.sigma @ J.T @ np.linalg.inv(S)  # S is Q in the book

        # 3)Innovation (measurement residual)
        innovation = np.array([distance - predicted_distance, angle - predicted_angle])
    
        # 4)Update landmark state (x and y coordinate)
        landmark.x += K[0, 0] * innovation[0] + K[0, 1] * innovation[1]
        landmark.y += K[1, 0] * innovation[0] + K[1, 1] * innovation[1]
            
        # 5)Update the covariance
        I = np.eye(3)  # Identity matrix
        landmark.sigma = (I - K @ J) @ landmark.sigma

        #6) Update the weight using the measurement likelihood
        det_S = np.linalg.det(S)
        if det_S > 0:
            weight_factor = 1 / np.sqrt(2 * np.pi * det_S)
            exponent = -0.5 * innovation.T @ np.linalg.inv(S) @ innovation
            self.weight *= weight_factor * np.exp(exponent)

    ## POSE ##
    def get_pose(self):
        """Return the current pose of the particle."""
        return self.pose