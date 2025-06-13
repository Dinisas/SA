#!/usr/bin/env python3
import numpy as np
import math
from landmark import Landmark
from utils import normalize_angle

class Particle:
    """ 
    Particle class representing a single particle in the particle filter.
    Enhanced with robust numerical stability and overflow protection.
    """
        
    def __init__(self, pose, nr_particles, pioneer_L, tuning_option):
        # Initialize particle pose, landmarks, weight, tuning options
        self.pose = pose
        self.landmarks = {}
        self.weight = 1.0
        self.pioneer_L = pioneer_L
        self.observation_vector = np.zeros((2, 1))
        self.default_weight = 1.0 / nr_particles
        
        # Jacobian matrix placeholder
        self.J_matrix = np.zeros((2, 2))
        # Covariance matrix placeholder
        self.adjusted_covariance = np.zeros((2, 2))
        self.trajectory = []
        self.next_virtual_id = 0
        self.virtual_id_assignments = {}
        
        # Tuning options
        self.alphas = tuning_option[0]
        
        # Motion thresholds
        self.min_motion_for_update = {
            'distance': 0.001,  # 1mm
            'rotation': 0.001   # ~0.057 degrees
        }
        
        # Numerical stability constants
        self.MIN_WEIGHT = 1e-30  # Minimum allowed weight
        self.MAX_WEIGHT = 1e10   # Maximum allowed weight
        self.MAX_EXPONENT = 50   # Maximum exponent for exp() function
        self.MIN_DETERMINANT = 1e-12  # Minimum determinant for matrix inversion
        self.REGULARIZATION = 1e-6   # Regularization term for numerical stability

    def assign_virtual_id(self):
        """Generate a new virtual ID for a newly detected marker"""
        virtual_id = f"virtual_{self.next_virtual_id}"
        self.next_virtual_id += 1
        return virtual_id

    def clamp_weight(self, weight):
        """Clamp weight to prevent numerical overflow/underflow."""
        if np.isnan(weight) or np.isinf(weight) or weight <= 0:
            return self.MIN_WEIGHT
        return np.clip(weight, self.MIN_WEIGHT, self.MAX_WEIGHT)

    def safe_exp(self, exponent):
        """Safe exponential function that prevents overflow."""
        clamped_exp = np.clip(exponent, -self.MAX_EXPONENT, self.MAX_EXPONENT)
        return np.exp(clamped_exp)

    def safe_matrix_inverse(self, matrix):
        """Safe matrix inversion with regularization."""
        try:
            # Add regularization to diagonal
            regularized_matrix = matrix + np.eye(matrix.shape[0]) * self.REGULARIZATION
            
            # Check condition number
            cond_num = np.linalg.cond(regularized_matrix)
            if cond_num > 1e12:
                # Use pseudo-inverse for ill-conditioned matrices
                return np.linalg.pinv(regularized_matrix)
            
            # Use Cholesky decomposition if matrix is positive definite
            try:
                L = np.linalg.cholesky(regularized_matrix)
                return np.linalg.inv(regularized_matrix)
            except np.linalg.LinAlgError:
                # Fallback to standard inverse
                return np.linalg.inv(regularized_matrix)
                
        except np.linalg.LinAlgError:
            # Final fallback to pseudo-inverse
            return np.linalg.pinv(matrix + np.eye(matrix.shape[0]) * self.REGULARIZATION)

    ## MOTION MODEL ## (particle prediction step based on motion model)
    def motion_model(self, odometry_delta):
        """ 
        Enhanced motion model that handles stationary robots better.
        Args:
            odometry_delta: Tuple containing (delta_dist, delta_rot1, delta_rot2).
        """
        # Get current pose
        x, y, theta = self.get_pose()
        
        # Unpack motion deltas
        delta_dist, delta_rot1, delta_rot2 = odometry_delta
        
        # Check if motion is significant enough to warrant an update
        if (abs(delta_dist) < self.min_motion_for_update['distance'] and 
            abs(delta_rot1) < self.min_motion_for_update['rotation'] and 
            abs(delta_rot2) < self.min_motion_for_update['rotation']):
            # Robot is essentially stationary - minimal noise
            # Add very small noise to prevent particle degeneracy
            new_x = x + np.random.normal(0, 0.0001)  # 0.1mm std dev
            new_y = y + np.random.normal(0, 0.0001)
            new_theta = normalize_angle(theta + np.random.normal(0, 0.0001))  # ~0.0057 degrees
        else:
            # Robot is moving - use full motion model
            # Add current pose to trajectory only when moving
            self.trajectory.append((x, y, theta))
            
            # Parameters for motion noise - adaptive based on motion magnitude
            alpha1 = self.alphas[0]
            alpha2 = self.alphas[1]
            alpha3 = self.alphas[2]
            alpha4 = self.alphas[3]
            
            # Scale noise based on motion magnitude
            motion_scale = min(1.0, abs(delta_dist) / 0.1)  # Normalized by 10cm
            rotation_scale = min(1.0, (abs(delta_rot1) + abs(delta_rot2)) / 0.2)  # Normalized by ~11.5 degrees
            
            # Adaptive noise calculation with bounds
            deviation_dist = motion_scale * math.sqrt(
                max(0, alpha1 * delta_rot1**2 + alpha2 * delta_dist**2)
            )
            deviation_rot1 = rotation_scale * math.sqrt(
                max(0, alpha3 * delta_dist**2 + alpha4 * delta_rot1**2 + alpha4 * delta_rot2**2)
            )
            deviation_rot2 = rotation_scale * math.sqrt(
                max(0, alpha1 * delta_rot2**2 + alpha2 * delta_dist**2)
            )
            
            # Ensure minimum noise to prevent degeneracy, maximum to prevent chaos
            deviation_dist = np.clip(deviation_dist, 0.001, 0.1)  # 1mm to 10cm
            deviation_rot1 = np.clip(deviation_rot1, 0.001, 0.1)  # 0.057° to 5.7°
            deviation_rot2 = np.clip(deviation_rot2, 0.001, 0.1)
            
            # Add gaussian noise to motion components
            noisy_dist = delta_dist + np.random.normal(0, deviation_dist)
            noisy_rot1 = delta_rot1 + np.random.normal(0, deviation_rot1)
            noisy_rot2 = delta_rot2 + np.random.normal(0, deviation_rot2)
            
            # Update the pose with noisy motion
            new_x = x + noisy_dist * math.cos(theta + noisy_rot1)
            new_y = y - noisy_dist * math.sin(theta + noisy_rot1)
            new_theta = normalize_angle(theta + noisy_rot1 + noisy_rot2)
        
        self.pose = np.array([new_x, new_y, new_theta])

    def get_dynamic_measurement_covariance(self, distance_meters, marker_pixel_size):
        """
        Calculate realistic measurement covariance for camera-based SLAM.
        Enhanced to consider robot motion state and numerical stability.
        
        Args:
            distance_meters: Distance to the landmark
            marker_pixel_size: Size of marker in pixels
        
        Returns:
            2x2 covariance matrix
        """
        # Ensure distance is positive and reasonable
        distance_meters = max(0.1, min(distance_meters, 50.0))  # Clamp between 10cm and 50m
        
        # Base camera uncertainty
        base_range_uncertainty = 0.03 * distance_meters + 0.02  # 3% of distance + 2cm
        base_bearing_uncertainty = np.radians(3.0)  # 3 degrees base
        
        # Pixel size factor - smaller markers have higher uncertainty
        if marker_pixel_size > 0:
            pixel_factor = max(1.0, min(100.0 / marker_pixel_size, 10.0))  # Clamp factor
        else:
            pixel_factor = 5.0  # Default for unknown pixel size
        
        # Apply pixel factor with bounds
        sigma_r = np.clip(base_range_uncertainty * pixel_factor, 0.05, 2.0)  # 5cm to 2m
        sigma_theta = np.clip(base_bearing_uncertainty * pixel_factor, 
                             np.radians(1.0), np.radians(30.0))  # 1° to 30°
        
        # SLAM scaling factors (more conservative)
        slam_range_factor = 1.2
        slam_bearing_factor = 1.2
        
        sigma_r *= slam_range_factor
        sigma_theta *= slam_bearing_factor
        
        # Final bounds after scaling
        sigma_r = np.clip(sigma_r, 0.08, 3.0)  # 8cm to 3m
        sigma_theta = np.clip(sigma_theta, np.radians(2.0), np.radians(45.0))  # 2° to 45°
        
        # Create covariance matrix with regularization
        Q = np.array([[sigma_r**2, 0.0],
                     [0.0, sigma_theta**2]])
        
        # Add small regularization to ensure positive definiteness
        Q += np.eye(2) * self.REGULARIZATION
        
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
        
        # Predicted range with minimum threshold
        predicted_range = max(0.01, math.sqrt(dx**2 + dy**2))  # Minimum 1cm
        
        # Predicted bearing (consistent with your measurement convention)
        predicted_bearing = -math.atan2(dy, dx) - theta
        predicted_bearing = normalize_angle(predicted_bearing)
        
        return predicted_range, predicted_bearing

    def calculate_mahalanobis_distance(self, measured_range, measured_bearing, landmark, marker_pixel_size):
        """
        Calculate Mahalanobis distance for data association with robust numerics.
        
        Args:
            measured_range: Measured distance to landmark
            measured_bearing: Measured bearing to landmark
            landmark: Landmark object to compare against
            marker_pixel_size: Size of marker in pixels
            
        Returns:
            float: Mahalanobis distance
        """
        # Input validation
        if measured_range <= 0:
            return float('inf')
        
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
        sqrt_q = math.sqrt(max(q, 0.01))  # Minimum 1cm distance
        
        if sqrt_q < 0.01:  # Too close
            return float('inf')
        
        J = np.array([[dx / sqrt_q, dy / sqrt_q, 0],
                     [dy / q, -dx / q, -1]])
        
        # Get measurement covariance
        Q = self.get_dynamic_measurement_covariance(measured_range, marker_pixel_size)
        
        # Innovation covariance: S = J*P*J' + Q
        try:
            S = J @ landmark.sigma @ J.T + Q
            
            # Add regularization for numerical stability
            S += np.eye(2) * self.REGULARIZATION
            
            # Check condition number
            cond_num = np.linalg.cond(S)
            if cond_num > 1e10:
                return float('inf')
            
            # Use safe matrix inverse
            S_inv = self.safe_matrix_inverse(S)
            
            # Calculate Mahalanobis distance
            mahalanobis_dist_sq = innovation.T @ S_inv @ innovation
            
            if mahalanobis_dist_sq < 0:
                return float('inf')
            
            mahalanobis_dist = math.sqrt(max(0, mahalanobis_dist_sq))
            return mahalanobis_dist
            
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return float('inf')

    def find_best_landmark_association(self, measured_range, measured_bearing, marker_pixel_size):
        """
        Find the best landmark to associate with a measurement.
        Enhanced with better search strategy and numerical stability.
        
        Args:
            measured_range: Measured distance
            measured_bearing: Measured bearing angle
            marker_pixel_size: Size of marker in pixels
            
        Returns:
            tuple: (best_landmark_id, min_distance)
        """
        if not self.landmarks or measured_range <= 0:
            return None, float('inf')
        
        min_distance = float('inf')
        best_landmark_id = None
        
        # Pre-filter landmarks based on rough distance
        max_range_error = max(measured_range * 0.5 + 1.0, 2.0)  # At least 2m tolerance
        
        candidates = []
        for landmark_id, landmark in self.landmarks.items():
            # Quick distance check
            rough_dist = landmark.distance_to(self.pose[0], self.pose[1])
            if abs(rough_dist - measured_range) <= max_range_error:
                candidates.append((landmark_id, landmark))
        
        # If no candidates within range, consider all landmarks
        if not candidates:
            candidates = list(self.landmarks.items())
        
        # Calculate Mahalanobis distance for candidates
        for landmark_id, landmark in candidates:
            try:
                distance = self.calculate_mahalanobis_distance(
                    measured_range, measured_bearing, landmark, marker_pixel_size
                )
                
                if distance < min_distance and not (np.isnan(distance) or np.isinf(distance)):
                    min_distance = distance
                    best_landmark_id = landmark_id
            except (ValueError, FloatingPointError):
                continue  # Skip this landmark if calculation fails
        
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
        # Input validation
        if landmark_dist <= 0:
            return
            
        if landmark_id not in self.landmarks:
            self.create_landmark(landmark_dist, landmark_bearing_angle, landmark_id, marker_pixel_size)
        else:
            self.update_landmark(landmark_dist, landmark_bearing_angle, landmark_id, marker_pixel_size)

    def handle_landmark(self, landmark_dist, landmark_bearing_angle, landmark_id, marker_pixel_size):
        """
        Main landmark handling method - routes to appropriate handler.
        """
        if landmark_id == -1:
            raise ValueError("Data association should be handled at particle filter level")
        
        self.handle_landmark_with_id(landmark_dist, landmark_bearing_angle, str(landmark_id), marker_pixel_size)

    def create_landmark(self, distance, angle, landmark_id, marker_pixel_size):
        """
        Create a new landmark in the particle's map with robust initialization.
        """
        # Input validation
        if distance <= 0:
            return
            
        # Get particle pose and update landmark position accordingly
        x, y, theta = self.get_pose()
        
        # Create marker from observation
        landmark_x = x + distance * math.cos(theta + angle)
        landmark_y = y - distance * math.sin(theta + angle)
        self.landmarks[landmark_id] = Landmark(landmark_x, landmark_y)
        
        # Calculate Jacobian matrix for the measurement function
        dx = landmark_x - x
        dy = landmark_y - y
        q = max(dx**2 + dy**2, 0.01)  # Minimum distance squared
        sqrt_q = math.sqrt(q)

        J = np.array([[dx / sqrt_q, dy / sqrt_q, 0],
                      [-dy / q, dx / q, -1]])

        # Get dynamic measurement covariance
        Q = self.get_dynamic_measurement_covariance(distance, marker_pixel_size)
        
        # Initialize landmark covariance with moderate uncertainty
        initial_sigma = np.eye(3) * 5.0  # Reasonable initial uncertainty
        
        try:
            # Innovation covariance
            S = J @ initial_sigma @ J.T + Q
            
            # Safe matrix inversion
            S_inv = self.safe_matrix_inverse(S)
            
            # Kalman gain
            K = initial_sigma @ J.T @ S_inv
            
            # Update covariance
            I = np.eye(3)
            self.landmarks[landmark_id].sigma = (I - K @ J) @ initial_sigma
            
            # Ensure covariance remains positive definite
            self.landmarks[landmark_id].sigma += np.eye(3) * self.REGULARIZATION
            
        except (np.linalg.LinAlgError, ValueError):
            # Fallback to default covariance
            self.landmarks[landmark_id].sigma = np.eye(3) * 10.0
                
        # Set a safe default importance weight
        self.weight = self.clamp_weight(self.default_weight)

    def update_landmark(self, distance, angle, landmark_id, marker_pixel_size):
        """
        Updates an existing landmark using the EKF update step with robust numerics.
        """
        # Input validation
        if distance <= 0:
            return
            
        landmark = self.landmarks[str(landmark_id)]
        x, y, theta = self.pose
 
        # Landmark prediction of the measurement based on motion model
        dx = landmark.x - x
        dy = landmark.y - y
        
        # Prevent numerical issues with very close landmarks
        q = dx**2 + dy**2
        if q < 0.01:  # Less than 1cm distance
            return
        
        sqrt_q = math.sqrt(q)
        
        predicted_distance = sqrt_q
        predicted_angle = -math.atan2(dy, dx) - theta
        predicted_angle = normalize_angle(predicted_angle)

        # Calculate Jacobian matrix
        J = np.array([[dx / sqrt_q, dy / sqrt_q, 0],
                      [dy / q, -dx / q, -1]])

        # Get dynamic measurement covariance
        Q = self.get_dynamic_measurement_covariance(distance, marker_pixel_size)

        try:
            # Calculate the Kalman Gain with regularization
            S = J @ landmark.sigma @ J.T + Q
            S += np.eye(2) * self.REGULARIZATION  # Regularization
            
            # Safe matrix inversion
            S_inv = self.safe_matrix_inverse(S)
            K = landmark.sigma @ J.T @ S_inv

            # Innovation (measurement residual)
            innovation = np.array([distance - predicted_distance, 
                                  normalize_angle(angle - predicted_angle)])
            
            # Limit innovation magnitude to prevent jumps
            max_position_innovation = min(0.5, distance * 0.3)  # Max 50cm or 30% of distance
            max_angle_innovation = np.radians(30)  # 30 degrees max
            
            innovation[0] = np.clip(innovation[0], -max_position_innovation, max_position_innovation)
            innovation[1] = np.clip(innovation[1], -max_angle_innovation, max_angle_innovation)
        
            # Update landmark state
            update = K @ innovation
            
            # Limit position updates to prevent wild jumps
            max_position_update = min(1.0, distance * 0.5)  # Max 1m or 50% of distance
            update[0] = np.clip(update[0], -max_position_update, max_position_update)
            update[1] = np.clip(update[1], -max_position_update, max_position_update)
            
            landmark.x += update[0]
            landmark.y += update[1]
                
            # Update the covariance
            I = np.eye(3)
            landmark.sigma = (I - K @ J) @ landmark.sigma
            
            # Ensure covariance remains positive definite
            landmark.sigma += np.eye(3) * self.REGULARIZATION

            # Update the weight using the measurement likelihood (robust version)
            det_S = np.linalg.det(S)
            
            if det_S > self.MIN_DETERMINANT:
                # Calculate weight factor with overflow protection
                log_weight_factor = -0.5 * math.log(2 * math.pi * det_S)
                
                # Calculate exponent with clipping
                exponent = -0.5 * innovation.T @ S_inv @ innovation
                exponent = np.clip(exponent, -self.MAX_EXPONENT, self.MAX_EXPONENT)
                
                # Calculate total log likelihood
                log_likelihood = log_weight_factor + exponent
                log_likelihood = np.clip(log_likelihood, -self.MAX_EXPONENT, self.MAX_EXPONENT)
                
                # Update weight in log space then convert back
                likelihood = self.safe_exp(log_likelihood)
                new_weight = self.weight * likelihood
                
                self.weight = self.clamp_weight(new_weight)
            else:
                # If determinant is too small, apply small weight penalty
                self.weight = self.clamp_weight(self.weight * 0.1)
                
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            # If numerical issues occur, apply weight penalty and skip update
            self.weight = self.clamp_weight(self.weight * 0.05)

    ## POSE ##
    def get_pose(self):
        """Return the current pose of the particle."""
        return self.pose