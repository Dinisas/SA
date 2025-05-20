import numpy as np
import math

class Landmark:
    #initalization of landmark position, id and gaussian parameters (mean and covariance)
    def __init__(self,x,y):
        self.x=x 
        self.y=y 
        self.id=id

        self.mean=np.array([[self.x],[self.y]])
        self.covariance=np.eye(2)*1000
    
    #update the landmark's mean and covariance
    def update(self, mean, covariannce):
        self.mean=mean
        self.covariance=covariance

    # Calculate the Euclidean distance to another point (x, y)
    def distance_to(self, x, y):
        return math.sqrt((self.x - x) ** 2 + (self.y - y) ** 2)


class Particle:
    """ 
    Represents a single hypothesis in the FastSLAM algorithm.
    Each particle tracks:
    - Robot pose (x, y, theta)
    - Map of landmarks
    - Weight (importance factor)
    """
    def __init__(self, pose, num_particles, wheel_base, motion_model_type='differential_drive', tuning_options=None):
        self.pose = pose  # [x, y, theta]
        self.landmarks = {}  # Dictionary of landmarks {id: Landmark}
        self.weight = 1.0
        self.wheel_base = wheel_base  # Distance between wheels (Pioneer 3DX: ~0.38m)
        self.default_weight = 1/num_particles
        self.motion_model_type = motion_model_type
        self.trajectory = []  # Store trajectory for visualization/analysis
        
        # Default tuning parameters if none provided
        if tuning_options is None:
            self.Q_init = np.diag([0.1, 0.1])  # Initial measurement noise
            self.Q_update = np.diag([0.7, 0.7])  # Update measurement noise
            self.alphas = [0.00001, 0.00001, 0.00001, 0.00001]  # Motion model noise
        else:
            self.Q_init, self.Q_update, self.alphas = tuning_options
        
    ##MOTION MODEL##
    def motion_model(self, odometry_delta):
        """ 
        Updates particle pose based on odometry
        odometry_delta: [delta_dist, delta_rot1, delta_rot2]
        """
        # Save current pose to trajectory
        self.trajectory.append(np.copy(self.pose))
        
        # Apply motion model with noise
        self.pose = differential_drive_model(
            self.pose, 
            odometry_delta, 
            self.wheel_base, 
            self.alphas
        )

    ##LANDMARK HANDLING##
    def handle_landmark(self, landmark_dist, landmark_bearing_angle, landmark_id):
        """Process a landmark observation to update the map"""
        landmark_id = str(landmark_id)
        if landmark_id not in self.landmarks:
            self.create_landmark(landmark_dist, landmark_bearing_angle, landmark_id)
        else:
            self.update_landmark(landmark_dist, landmark_bearing_angle, landmark_id)

    def create_landmark(self, distance, angle, landmark_id):
        """Create a new landmark in the map based on the observation"""
        x, y, theta = self.pose
        
        # Calculate global position of landmark based on robot's position and measurement
        landmark_x = x + distance * math.cos(theta + angle)
        landmark_y = y - distance * math.sin(theta + angle)
        
        # Create new landmark
        self.landmarks[landmark_id] = Landmark(landmark_x, landmark_y)
        
        # Initial covariance based on measurement uncertainty
        # Higher uncertainty for distant landmarks
        distance_uncertainty = 0.1 + 0.05 * distance  # Increases with distance
        angle_uncertainty = 0.1  # Radians
        
        # Calculate Jacobian matrix for the measurement model
        dx = landmark_x - x
        dy = landmark_y - y
        q = dx**2 + dy**2
        sqrt_q = math.sqrt(q)
        
        # Jacobian relates how changes in landmark position affect the measurement
        J = np.array([
            [dx / sqrt_q, dy / sqrt_q, 0],  # Derivative of distance w.r.t. x, y, theta
            [-dy / q, dx / q, -1]           # Derivative of bearing w.r.t. x, y, theta
        ])
        
        # Initialize with high uncertainty
        # 3x3 matrix since we track x, y of landmark and robot's orientation
        initial_sigma = np.eye(3) * 1000.0
        
        # Measurement noise covariance
        Q = np.diag([distance_uncertainty**2, angle_uncertainty**2])
        
        # Update the covariance using EKF update equations
        S = J @ initial_sigma @ J.T + Q  # Innovation covariance
        K = initial_sigma @ J.T @ np.linalg.inv(S)  # Kalman gain
        
        # Update the landmark's covariance
        self.landmarks[landmark_id].sigma = (np.eye(3) - K @ J) @ initial_sigma
        
        # Set weight to default since this is a new landmark
        self.weight = self.default_weight
    
    def update_landmark(self, distance, angle, landmark_id):
        """Updates an existing landmark using the EKF update step"""
        landmark = self.landmarks[str(landmark_id)]
        x, y, theta = self.pose
        
        # Prediction of the measurement based on current estimates
        dx = landmark.x - x
        dy = landmark.y - y
        
        predicted_distance = math.sqrt(dx**2 + dy**2)
        predicted_angle = normalize_angle(math.atan2(-dy, dx) - theta)
        
        # Calculate innovation (measurement residual)
        distance_innovation = distance - predicted_distance
        angle_innovation = normalize_angle(angle - predicted_angle)
        innovation = np.array([distance_innovation, angle_innovation])
        
        # Calculate Jacobian of the measurement model
        q = dx**2 + dy**2
        sqrt_q = math.sqrt(q)
        
        # This Jacobian relates how changes in landmark position affect measurements
        J = np.array([
            [dx / sqrt_q, dy / sqrt_q, 0],  # Derivative of distance w.r.t. x, y, theta
            [-dy / q, dx / q, -1]           # Derivative of bearing w.r.t. x, y, theta
        ])
        
        # Measurement noise - increases with distance for realistic camera model
        distance_uncertainty = 0.1 + 0.05 * distance  # More uncertainty at greater distances
        angle_uncertainty = 0.1 + 0.02 * distance     # Angular precision decreases with distance
        Q = np.diag([distance_uncertainty**2, angle_uncertainty**2])
        
        # EKF update equations
        S = J @ landmark.sigma @ J.T + Q  # Innovation covariance
        K = landmark.sigma @ J.T @ np.linalg.inv(S)  # Kalman gain
        
        # Update landmark position
        update_vector = K @ innovation
        landmark.x += update_vector[0]
        landmark.y += update_vector[1]
        
        # Update the covariance (represents uncertainty in landmark position)
        landmark.sigma = (np.eye(3) - K @ J) @ landmark.sigma
        
        # Update the particle weight based on how well the observation matches prediction
        det_S = np.linalg.det(S)
        if det_S > 0:
            # Multivariate Gaussian probability
            weight_factor = 1.0 / math.sqrt(2 * math.pi * det_S)
            exponent = -0.5 * innovation.T @ np.linalg.inv(S) @ innovation
            measurement_likelihood = weight_factor * math.exp(exponent)
            
            # Adjust weight by this measurement's likelihood
            self.weight *= measurement_likelihood
        else:
            # If determinant is not positive (can happen due to numerical issues)
            # Use a small positive weight to avoid eliminating this particle
            self.weight *= 0.01

