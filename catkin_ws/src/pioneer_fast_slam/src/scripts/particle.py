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
        # tuning options
        self.Q_init = tuning_option[0]
        self.Q_update = tuning_option[1]
        self.alphas = tuning_option[2]

    ## MOTION MODEL ##
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

    ## WEIGHT ##
    def handle_landmark(self, landmark_dist, landmark_bearing_angle, landmark_id):
        """
        Handle landmark observation for the particle.
        Args:
            landmark_dist: Distance to the landmark.
            landmark_bearing_angle: Bearing angle to the landmark.
            landmark_id: Identifier of the landmark.
        """
        landmark_id = str(landmark_id)
        if landmark_id not in self.landmarks:
            # create new landmark
            self.create_landmark(landmark_dist, landmark_bearing_angle, landmark_id)
        else:
            # Update Extended Kalman Filter (update existing landmark)
            self.update_landmark(landmark_dist, landmark_bearing_angle, landmark_id)

    def create_landmark(self, distance, angle, landmark_id):
        """
        Create a new landmark in the particle's map.
        Args:
            distance: Distance to the landmark.
            angle: Bearing angle to the landmark.
            landmark_id: Identifier of the landmark.
        """
        # get particle pose and update landmark position accordingly
        x, y, theta = self.get_pose()
        landmark_x = x + distance * math.cos(theta + angle)
        landmark_y = y - distance * math.sin(theta + angle)
        self.landmarks[landmark_id] = Landmark(landmark_x, landmark_y)
        
        # Prediction of the measurement (predicted distance and bearing angle to the landmark)
        dx = landmark_x - x
        dy = landmark_y - y
        predicted_distance = math.sqrt(dx**2 + dy**2)
        predicted_angle = math.atan2(dy, dx) - theta
        
        # Normalize the angle between -pi and pi
        predicted_angle = (predicted_angle + np.pi) % (2 * np.pi) - np.pi
        
        # Calculate Jacobian matrix H of the measurement function for the EKF
        q = dx**2 + dy**2
        sqrt_q = math.sqrt(q)

        J = np.array([[dx / sqrt_q, dy / sqrt_q, 0],
                      [-dy / q, dx / q, -1]])
        
        # Measurement noise covariance matrix (should be tuned)
        Q = self.Q_init  # Example values, Q is Q_t in the book
        # innovation covariance
        S = J @ self.landmarks[landmark_id].sigma @ J.T + Q
        # kalman gain
        K = self.landmarks[landmark_id].sigma @ J.T @ np.linalg.inv(S)
        # update covariance
        self.landmarks[landmark_id].sigma = (np.eye(3) - K @ J) @ self.landmarks[landmark_id].sigma
                
        # Set a default importance weight
        self.weight = self.default_weight  # p0 in the book

    def update_landmark(self, distance, angle, landmark_id):
        """
        Updates an existing landmark using the EKF update step.
        Args:
            distance: Measured distance to the landmark.
            angle: Measured bearing angle to the landmark.
            landmark_id: Identifier of the landmark.
        """
        landmark = self.landmarks[str(landmark_id)]
        x, y, theta = self.pose
 
        # Prediction of the measurement
        dx = landmark.x - x
        dy = landmark.y - y
        predicted_distance = math.sqrt(dx**2 + dy**2)
        predicted_angle = -math.atan2(dy, dx) - theta
            
        # Normalize the angle between -pi and pi
        predicted_angle = (predicted_angle + np.pi) % (2 * np.pi) - np.pi
            
        # Calculate Jacobian matrix H of the measurement function
        q = dx**2 + dy**2
        sqrt_q = math.sqrt(q)
            
        J = np.array([[dx / sqrt_q, dy / sqrt_q, 0],
                      [dy / q, -dx / q, -1]])
            
        # Measurement noise covariance matrix (should be tuned)
        Q = self.Q_update  # Example values

        # Calculate the Kalman Gain
        S = J @ landmark.sigma @ J.T + Q  # Measurement prediction covariance
        K = landmark.sigma @ J.T @ np.linalg.inv(S)  # S is Q in the book

        # Innovation (measurement residual)
        innovation = np.array([distance - predicted_distance, angle - predicted_angle])
    
        # Update landmark state (x and y coordinate)
        landmark.x += K[0, 0] * innovation[0] + K[0, 1] * innovation[1]
        landmark.y += K[1, 0] * innovation[0] + K[1, 1] * innovation[1]
            
        # Update the covariance
        I = np.eye(3)  # Identity matrix
        landmark.sigma = (I - K @ J) @ landmark.sigma

        # Update the weight using the measurement likelihood
        det_S = np.linalg.det(S)
        if det_S > 0:
            weight_factor = 1 / np.sqrt(2 * np.pi * det_S)
            exponent = -0.5 * innovation.T @ np.linalg.inv(S) @ innovation
            self.weight *= weight_factor * np.exp(exponent)

    ## POSE ##
    def get_pose(self):
        """Return the current pose of the particle."""
        return self.pose