import numpy as np
import math
from landmark import Landmark
from utils import normalize_angle

class Particle:
    """ 
    Particle class representing a single particle in the particle filter.
    Enhanced with better motion model for stationary robot handling.
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
        self.alphas = tuning_option[0]
        
        # Motion thresholds
        self.min_motion_for_update = {
            'distance': 0.001,  # 1mm
            'rotation': 0.001   # ~0.057 degrees
        }

    def assign_virtual_id(self):
        """Generate a new virtual ID for a newly detected marker"""
        virtual_id = f"virtual_{self.next_virtual_id}"
        self.next_virtual_id += 1
        return virtual_id

    ## MOTION MODEL ## (particle prediction step based on motion model)
    def motion_model(self, odometry_delta):
        """ 
        Enhanced motion model that handles stationary robots better.
        Args:
            odometry_delta: Tuple containing (delta_dist, delta_rot1, delta_rot2).
        """
        # get current pose
        x, y, theta = self.get_pose()
        
        # unpack motion deltas
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
            
            # Adaptive noise calculation
            deviation_dist = motion_scale * math.sqrt(alpha1 * delta_rot1**2 + alpha2 * delta_dist**2)
            deviation_rot1 = rotation_scale * math.sqrt(alpha3 * delta_dist**2 + alpha4 * delta_rot1**2 + alpha4 * delta_rot2**2)
            deviation_rot2 = rotation_scale * math.sqrt(alpha1 * delta_rot2**2 + alpha2 * delta_dist**2)
            
            # Ensure minimum noise to prevent degeneracy
            deviation_dist = max(deviation_dist, 0.001)  # At least 1mm
            deviation_rot1 = max(deviation_rot1, 0.001)  # At least 0.057 degrees
            deviation_rot2 = max(deviation_rot2, 0.001)
            
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
        Enhanced to consider robot motion state.
        
        Args:
            distance_meters: Distance to the landmark
            marker_pixel_size: Size of marker in pixels
        
        Returns:
            2x2 covariance matrix
        """
        # Base camera uncertainty
        base_range_uncertainty = 0.03 * distance_meters + 0.02  # 3% of distance + 2cm
        base_bearing_uncertainty = np.radians(3.0)  # 3 degrees base
        
        # Pixel size factor - smaller markers have higher uncertainty
        pixel_factor = max(1.0, 100.0 / marker_pixel_size) if marker_pixel_size > 0 else 2.0
        
        # Apply pixel factor
        sigma_r = base_range_uncertainty * pixel_factor
        sigma_theta = base_bearing_uncertainty * pixel_factor
        
        # Ensure minimum values
        sigma_r = max(sigma_r, 0.05)  # At least 5cm
        sigma_theta = max(sigma_theta, np.radians(2.0))  # At least 2 degrees
        
        # SLAM scaling factors
        slam_range_factor = 1.5  # Less aggressive than before
        slam_bearing_factor = 1.5
        
        sigma_r *= slam_range_factor
        sigma_theta *= slam_bearing_factor
        
        # Final minimums after scaling
        sigma_r = max(sigma_r, 0.08)  # At least 8cm
        sigma_theta = max(sigma_theta, np.radians(2.5))  # At least 2.5 degrees
        
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
        Enhanced with better numerical stability.
        
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
        
        if sqrt_q < 0.01:  # 1cm minimum distance
            return float('inf')
        
        J = np.array([[dx / sqrt_q, dy / sqrt_q, 0],
                    [dy / q, -dx / q, -1]])
        
        # Get measurement covariance
        Q = self.get_dynamic_measurement_covariance(measured_range, marker_pixel_size)
        
        # Innovation covariance: S = J*P*J' + Q
        S = J @ landmark.sigma @ J.T + Q
        
        # Add regularization for numerical stability
        regularization = np.eye(2) * 1e-4
        S += regularization
        
        try:
            # Check condition number
            cond_num = np.linalg.cond(S)
            if cond_num > 1e10:
                # Matrix is poorly conditioned
                return float('inf')
            
            # Use Cholesky decomposition for stability
            L = np.linalg.cholesky(S)
            y = np.linalg.solve(L, innovation)
            mahalanobis_dist = math.sqrt(np.dot(y, y))
            return mahalanobis_dist
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse for singular matrices
            try:
                S_pinv = np.linalg.pinv(S)
                mahalanobis_dist = math.sqrt(innovation.T @ S_pinv @ innovation)
                return mahalanobis_dist
            except:
                return float('inf')

    def find_best_landmark_association(self, measured_range, measured_bearing, marker_pixel_size):
        """
        Find the best landmark to associate with a measurement.
        Enhanced with better search strategy.
        
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
        
        # Pre-filter landmarks based on rough distance
        max_range_error = measured_range * 0.5 + 1.0  # 50% + 1m tolerance
        
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
        """
        if landmark_id == -1:
            raise ValueError("Data association should be handled at particle filter level")
        
        self.handle_landmark_with_id(landmark_dist, landmark_bearing_angle, str(landmark_id), marker_pixel_size)

    def create_landmark(self, distance, angle, landmark_id, marker_pixel_size):
        """
        Create a new landmark in the particle's map.
        Enhanced with better initialization.
        """
        # get particle pose and update landmark position accordingly
        x, y, theta = self.get_pose()
        
        # create marker from observation
        landmark_x = x + distance * math.cos(theta + angle)
        landmark_y = y - distance * math.sin(theta + angle)
        self.landmarks[landmark_id] = Landmark(landmark_x, landmark_y)
        
        # Calculate Jacobian matrix for the measurement function
        dx = landmark_x - x
        dy = landmark_y - y
        q = dx**2 + dy**2
        sqrt_q = math.sqrt(max(q, 0.0001))  # Prevent division by zero

        J = np.array([[dx / sqrt_q, dy / sqrt_q, 0],
                      [-dy / q, dx / q, -1]])

        # Get dynamic measurement covariance
        Q = self.get_dynamic_measurement_covariance(distance, marker_pixel_size)
        
        # Initialize landmark covariance
        # Start with moderate uncertainty (not too high, not too low)
        initial_sigma = np.eye(3) * 10.0  # Reduced from 1000
        
        # innovation covariance
        S = J @ initial_sigma @ J.T + Q
        # kalman gain
        K = initial_sigma @ J.T @ np.linalg.inv(S)
        # update covariance
        self.landmarks[landmark_id].sigma = (np.eye(3) - K @ J) @ initial_sigma
                
        # Set a default importance weight
        self.weight = self.default_weight

    def update_landmark(self, distance, angle, landmark_id, marker_pixel_size):
        """
        Updates an existing landmark using the EKF update step.
        Enhanced with better numerical stability.
        """
        landmark = self.landmarks[str(landmark_id)]
        x, y, theta = self.pose
 
        # Landmark prediction of the measurement based on motion model
        dx = landmark.x - x
        dy = landmark.y - y
        
        # Prevent numerical issues with very close landmarks
        if abs(dx) < 0.01 and abs(dy) < 0.01:
            return
        
        predicted_distance = math.sqrt(dx**2 + dy**2)
        predicted_angle = -math.atan2(dy, dx) - theta
        predicted_angle = normalize_angle(predicted_angle)

        # Calculate Jacobian matrix
        q = dx**2 + dy**2
        sqrt_q = math.sqrt(max(q, 0.0001))
            
        J = np.array([[dx / sqrt_q, dy / sqrt_q, 0],
                      [dy / q, -dx / q, -1]])

        # Get dynamic measurement covariance
        Q = self.get_dynamic_measurement_covariance(distance, marker_pixel_size)

        # Calculate the Kalman Gain with regularization
        S = J @ landmark.sigma @ J.T + Q
        S += np.eye(2) * 1e-6  # Regularization
        
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            S_inv = np.linalg.pinv(S)
        
        K = landmark.sigma @ J.T @ S_inv

        # Innovation (measurement residual)
        innovation = np.array([distance - predicted_distance, 
                              normalize_angle(angle - predicted_angle)])
        
        # Limit innovation magnitude to prevent jumps
        max_position_innovation = 0.5  # 50cm max update
        max_angle_innovation = np.radians(30)  # 30 degrees max
        
        innovation[0] = np.clip(innovation[0], -max_position_innovation, max_position_innovation)
        innovation[1] = np.clip(innovation[1], -max_angle_innovation, max_angle_innovation)
    
        # Update landmark state
        update = K @ innovation
        landmark.x += update[0]
        landmark.y += update[1]
            
        # Update the covariance
        I = np.eye(3)
        landmark.sigma = (I - K @ J) @ landmark.sigma
        
        # Ensure covariance remains positive definite
        # Add small diagonal to maintain numerical stability
        landmark.sigma += np.eye(3) * 1e-6

        # Update the weight using the measurement likelihood
        det_S = np.linalg.det(S)
        if det_S > 1e-10:  # Check for numerical stability
            weight_factor = 1 / np.sqrt(2 * np.pi * det_S)
            exponent = -0.5 * innovation.T @ S_inv @ innovation
            
            # Limit exponent to prevent numerical overflow
            exponent = max(exponent, -50)
            
            self.weight *= weight_factor * np.exp(exponent)
        else:
            # If determinant too small, use a small weight update
            self.weight *= 0.1

    ## POSE ##
    def get_pose(self):
        """Return the current pose of the particle."""
        return self.pose