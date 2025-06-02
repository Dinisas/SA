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

    #DYNAMIC MEASUREMENT COVARIANCE MATRIX
    # def get_dynamic_measurement_covariance(self, distance_meters, marker_pixel_size):
    #     """
    #     Calculate dynamic measurement covariance matrix based on calibrated error models
        
    #     Args: 
    #         distance_meters - estimated distance to target
    #         marker_pixel_size - size of marker in pixels (average side length)
    #     Returns:
    #         2x2 covariance matrix [[σ_r², 0], [0, σ_θ²]]
    #     """
    #     # Range error model: σ_r(d) = 0.002428·d² - 0.005080
    #     sigma_r = 0.002428 * distance_meters**2 - 0.005080
    #     # Ensure positive value (minimum 1mm error)
    #     sigma_r = max(sigma_r, 0.001)
        
    #     # Bearing error model: σ_θ = 0.011040 - 0.210128/s  
    #     sigma_theta = 0.011040 - 0.210128 / marker_pixel_size
    #     # Ensure positive value (minimum 0.001 rad ≈ 0.06°)
    #     sigma_theta = max(sigma_theta, 0.001)
        
    #     # Create covariance matrix (note: covariance = variance = σ²)
    #     Q_dynamic = np.array([[sigma_r**2, 0.0],
    #                         [0.0, sigma_theta**2]])
    #     return Q_dynamic
    # def get_dynamic_measurement_covariance(self, distance_meters, marker_pixel_size):
    #     """
    #     Calculate dynamic measurement covariance with moderate SLAM scaling
    #     """
    #     # Base calibrated models
    #     sigma_r = 0.002428 * distance_meters**2 - 0.005080
    #     sigma_theta = 0.011040 - 0.210128 / marker_pixel_size
        
    #     # Ensure minimum values
    #     sigma_r = max(sigma_r, 0.001)
    #     sigma_theta = max(sigma_theta, 0.001)
        
    #     # **MODERATE SLAM SCALING** - not too pessimistic
    #     slam_range_factor = 5.0      # Scale range uncertainty 10x (not 50x)
    #     slam_bearing_factor = 10.0    # Scale bearing uncertainty 20x (not 100x)
        
    #     sigma_r *= slam_range_factor     # ~0.01-0.5m uncertainty
    #     sigma_theta *= slam_bearing_factor  # ~1-10° uncertainty
        
    #     print(f"Scaled covariance: σ_r={sigma_r:.3f}m, σ_θ={np.degrees(sigma_theta):.1f}°")
        
    #     Q_dynamic = np.array([[sigma_r**2, 0.0],
    #                         [0.0, sigma_theta**2]])
    #     return Q_dynamic
    def get_dynamic_measurement_covariance(self, distance_meters, marker_pixel_size):
        """
        Calculate REALISTIC measurement covariance for camera-based SLAM
        """
        # Base calibrated models - but DON'T use these directly
        # sigma_r = 0.002428 * distance_meters**2 - 0.005080
        # sigma_theta = 0.011040 - 0.210128 / marker_pixel_size
        
        # REALISTIC camera uncertainty:
        # Range: 2-5% of distance is typical for cameras
        sigma_r = 0.03 * distance_meters + 0.02  # 3% of distance + 2cm base
        
        # Bearing: depends on pixel resolution, but 2-5 degrees is realistic
        sigma_theta = np.radians(3.0)  # 3 degrees base uncertainty
        
        # Ensure minimum values
        sigma_r = max(sigma_r, 0.05)      # At least 5cm (not 5mm!)
        sigma_theta = max(sigma_theta, np.radians(2.0))  # At least 2 degrees
        
        # MODERATE SLAM scaling (not too aggressive)
        slam_range_factor = 2.0      # Range uncertainty multiplier
        slam_bearing_factor = 2.0    # Bearing uncertainty multiplier
        
        sigma_r *= slam_range_factor
        sigma_theta *= slam_bearing_factor
        
        # Final minimums after scaling
        sigma_r = max(sigma_r, 0.10)      # At least 10cm uncertainty
        sigma_theta = max(sigma_theta, np.radians(3.0))  # At least 3 degrees
        
        # print(f"Scaled covariance: σ_r={sigma_r:.3f}m, σ_θ={np.degrees(sigma_theta):.1f}°")
        
        # Remove the hardcoded matrix!
        Q_dynamic = np.array([[0.1, 0.0],
                            [0.0, 0.1]])
        return Q_dynamic

    def predict_measurement_for_landmark(self, landmark):
        """Predict what range/bearing we should measure for this landmark"""
        x, y, theta = self.get_pose()
        dx = landmark.x - x
        dy = landmark.y - y
        predicted_range = math.sqrt(dx**2 + dy**2)
        
        # CRITICAL: Ensure consistent angle calculation
        # Bearing is the angle from robot's forward direction to landmark
        global_angle = math.atan2(dy, dx)  # Angle from x-axis to landmark
        predicted_bearing = global_angle - theta  # Convert to robot frame
        
        # Normalize to [-π, π]
        predicted_bearing = normalize_angle(predicted_bearing)
        
        # IMPORTANT: Match the sign convention used in your measurement
        # If your measurements use -atan2(dy, dx) - theta, then:
        predicted_bearing = -math.atan2(dy, dx) - theta
        predicted_bearing = normalize_angle(predicted_bearing)
        
        return predicted_range, predicted_bearing

    # def predict_measurement_for_landmark(self, landmark):
    #     """Predict what range/bearing we should measure for this landmark"""
    #     x, y, theta = self.get_pose()
    #     dx = landmark.x - x
    #     dy = landmark.y - y
    #     predicted_range = math.sqrt(dx**2 + dy**2)
    #     predicted_bearing = -math.atan2(dy, dx) - theta
    #     predicted_bearing = (predicted_bearing + np.pi) % (2 * np.pi) - np.pi
    #     return predicted_range, predicted_bearing

    # def measurement_based_mahalanobis_distance(self, measured_range, measured_bearing, landmark, marker_pixel_size):
    #     """
    #     Calculate Mahalanobis distance based on MEASUREMENT prediction, not position
    #     This is the theoretically correct approach!
    #     """
    #     # Predict what we should measure for this landmark
    #     predicted_range, predicted_bearing = self.predict_measurement_for_landmark(landmark)
        
    #     # Innovation = actual - predicted measurement
    #     innovation = np.array([measured_range - predicted_range, 
    #                         measured_bearing - predicted_bearing])
        
    #     print(f"  Predicted: r={predicted_range:.3f}m, θ={np.degrees(predicted_bearing):.1f}°")
    #     print(f"  Measured:  r={measured_range:.3f}m, θ={np.degrees(measured_bearing):.1f}°")
    #     print(f"  Innovation: Δr={innovation[0]:.3f}m, Δθ={np.degrees(innovation[1]):.1f}°")
        
    #     # Calculate Jacobian for this prediction
    #     x, y, theta = self.pose
    #     dx = landmark.x - x
    #     dy = landmark.y - y
    #     q = dx**2 + dy**2
    #     sqrt_q = math.sqrt(q)
        
    #     if sqrt_q < 0.001:
    #         return float('inf')
        
    #     J = np.array([[dx / sqrt_q, dy / sqrt_q, 0],
    #                 [dy / q, -dx / q, -1]])
        
    #     # Get measurement covariance based on actual measurement conditions
    #     Q = self.get_dynamic_measurement_covariance(measured_range, marker_pixel_size)
        
    #     # Innovation covariance: S = H*P*H' + R
    #     S = J @ landmark.sigma @ J.T + Q
        
    #     # Ensure numerical stability
    #     det_S = np.linalg.det(S)
    #     if abs(det_S) < 1e-8:
    #         S[0,0] += 1e-6
    #         S[1,1] += 1e-8
    #         det_S = np.linalg.det(S)
        
    #     if abs(det_S) < 1e-10:
    #         return float('inf')
        
    #     try:
    #         S_inv = np.linalg.inv(S)
    #         mahalanobis_dist = math.sqrt(innovation.T @ S_inv @ innovation)
    #          # **ADD THIS CRITICAL PRINT STATEMENT**
    #         print(f"  Mahalanobis distance: {mahalanobis_dist:.3f}")
    #         return mahalanobis_dist
    #     except np.linalg.LinAlgError:
    #         print("  Mahalanobis distance: inf (inversion failed)")
    #         return float('inf')

    def measurement_based_mahalanobis_distance(self, measured_range, measured_bearing, landmark, marker_pixel_size):
        """
        Calculate Mahalanobis distance with proper angle wrapping
        """
        # Predict what we should measure for this landmark
        predicted_range, predicted_bearing = self.predict_measurement_for_landmark(landmark)
        
        # Innovation = actual - predicted measurement
        innovation_range = measured_range - predicted_range
        innovation_bearing = measured_bearing - predicted_bearing
        
        # CRITICAL: Properly wrap bearing innovation to [-π, π]
        innovation_bearing = normalize_angle(innovation_bearing)
        
        innovation = np.array([innovation_range, innovation_bearing])
        
        print(f"  Predicted: r={predicted_range:.3f}m, θ={np.degrees(predicted_bearing):.1f}°")
        print(f"  Measured:  r={measured_range:.3f}m, θ={np.degrees(measured_bearing):.1f}°")
        print(f"  Innovation: Δr={innovation[0]:.3f}m, Δθ={np.degrees(innovation[1]):.1f}°")
        
        # Calculate Jacobian for this prediction
        x, y, theta = self.pose
        dx = landmark.x - x
        dy = landmark.y - y
        q = dx**2 + dy**2
        sqrt_q = math.sqrt(q)
        
        if sqrt_q < 0.001:
            return float('inf')
        
        J = np.array([[dx / sqrt_q, dy / sqrt_q, 0],
                    [dy / q, -dx / q, -1]])
        
        # Get REALISTIC measurement covariance
        Q = self.get_dynamic_measurement_covariance(measured_range, marker_pixel_size)
        
        # Innovation covariance: S = H*P*H' + R
        S = J @ landmark.sigma @ J.T + Q
        
        # Add small regularization for numerical stability
        S += np.eye(2) * 1e-6
        
        try:
            # Use Cholesky decomposition for better numerical stability
            L = np.linalg.cholesky(S)
            # Solve L*y = innovation for y
            y = np.linalg.solve(L, innovation)
            # Mahalanobis distance = sqrt(y'*y)
            mahalanobis_dist = math.sqrt(np.dot(y, y))
            
            print(f"  Mahalanobis distance: {mahalanobis_dist:.3f}")
            return mahalanobis_dist
            
        except np.linalg.LinAlgError:
            # Fallback to standard inverse
            try:
                S_inv = np.linalg.inv(S)
                mahalanobis_dist = math.sqrt(innovation.T @ S_inv @ innovation)
                print(f"  Mahalanobis distance: {mahalanobis_dist:.3f}")
                return mahalanobis_dist
            except:
                print("  Mahalanobis distance: inf (matrix inversion failed)")
                return float('inf')

    # def find_best_landmark_association_measurement_based(self, measured_range, measured_bearing, marker_pixel_size, 
    #                                            association_threshold=3.0):
    #     if not self.landmarks:
    #         print("Creating first landmark")
    #         return None, True
        
    #     min_distance = float('inf')
    #     best_landmark_id = None
        
    #     for landmark_id, landmark in self.landmarks.items():
    #         distance = self.measurement_based_mahalanobis_distance(
    #             measured_range, measured_bearing, landmark, marker_pixel_size)
            
    #         if distance < min_distance:
    #             min_distance = distance
    #             best_landmark_id = landmark_id
        
    #     # **ADD THESE DEBUG PRINTS**
    #     print(f"  BEST MATCH: {best_landmark_id} with distance {min_distance:.3f}")
    #     print(f"  THRESHOLD: {association_threshold}")
        
    #     if min_distance < association_threshold:
    #         print("  DECISION: ✅ ASSOCIATE")
    #         return best_landmark_id, False
    #     else:
    #         print("  DECISION: ❌ CREATE NEW")
    #         print(f"Creating new landmark: virtual_{len(self.landmarks)}")
    #         return None, True

    def find_best_landmark_association_measurement_based(self, measured_range, measured_bearing, marker_pixel_size, 
                                               association_threshold=4.0):
        """
        Improved association with better logic and debugging
        """
        if not self.landmarks:
            print("Creating first landmark")
            return None, True
        
        # First, do a sanity check on the measurement
        # print(f"\n  New measurement: r={measured_range:.3f}m, θ={np.degrees(measured_bearing):.1f}°")
        
        # Calculate Mahalanobis distance to all landmarks
        distances = {}
        innovations = {}
        
        for landmark_id, landmark in self.landmarks.items():
            # Skip if landmark is way too far (optimization)
            rough_dist = landmark.distance_to(self.pose[0], self.pose[1])
            if rough_dist > measured_range + 3.0:  # 3m tolerance
                continue
                
            distance = self.measurement_based_mahalanobis_distance(
                measured_range, measured_bearing, landmark, marker_pixel_size)
            
            # Store for analysis
            distances[landmark_id] = distance
            
            # Also store innovation for debugging
            predicted_range, predicted_bearing = self.predict_measurement_for_landmark(landmark)
            innovations[landmark_id] = {
                'range_inn': measured_range - predicted_range,
                'bearing_inn': np.degrees(measured_bearing - predicted_bearing)
            }
        
        if not distances:
            # print("  No landmarks within reasonable range")
            return None, True
        
        # Find best match
        best_landmark_id = min(distances, key=distances.get)
        min_distance = distances[best_landmark_id]
        
        # Show top 3 candidates for debugging
        sorted_landmarks = sorted(distances.items(), key=lambda x: x[1])[:3]
        # print(f"  Top candidates:")
        for lid, dist in sorted_landmarks:
            inn = innovations[lid]
            # print(f"    {lid}: Mahal={dist:.2f}, Δr={inn['range_inn']:.3f}m, Δθ={inn['bearing_inn']:.1f}°")
        
        # print(f"  BEST MATCH: {best_landmark_id} with distance {min_distance:.3f}")
        # print(f"  THRESHOLD: {association_threshold}")
        
        # Decision logic with multiple criteria
        should_associate = True
        
        # Criterion 1: Mahalanobis distance
        if min_distance >= association_threshold:
            should_associate = False
            print(f"  ❌ Failed Mahalanobis test: {min_distance:.2f} >= {association_threshold}")
        
        # Criterion 2: Sanity check on innovation magnitude
        best_innovation = innovations[best_landmark_id]
        range_innovation_abs = abs(best_innovation['range_inn'])
        
        if range_innovation_abs > 2.0:  # More than 2m error is suspicious
            print(f"  ⚠️  Large range innovation: {range_innovation_abs:.2f}m")
            if min_distance > association_threshold * 0.7:
                should_associate = False
                print(f"  ❌ Failed range innovation test")
        
        # Criterion 3: If we have many landmarks already, be more conservative
        if len(self.landmarks) >= 5 and min_distance > association_threshold * 0.8:
            should_associate = False
            print(f"  ❌ Being conservative with {len(self.landmarks)} landmarks")
        
        if should_associate:
            print("  DECISION: ✅ ASSOCIATE")
            return best_landmark_id, False
        else:
            print("  DECISION: ❌ CREATE NEW")
            return None, True

    ## WEIGHT ##
    # def handle_landmark(self, landmark_dist, landmark_bearing_angle, landmark_id, marker_pixel_size):
    #     """
    #     Handle landmark observation for the particle.
    #     Args:
    #         landmark_dist: Distance to the landmark.
    #         landmark_bearing_angle: Bearing angle to the landmark.
    #         landmark_id: Identifier of the landmark.
    #     """
    #     landmark_id = str(landmark_id)
    #     if landmark_id not in self.landmarks:
    #         # create new landmark
    #         self.create_landmark(landmark_dist, landmark_bearing_angle, landmark_id,marker_pixel_size)
    #     else:
    #         # Update Extended Kalman Filter (update existing landmark)
    #         self.update_landmark(landmark_dist, landmark_bearing_angle, landmark_id, marker_pixel_size)

    # def handle_landmark(self, landmark_dist, landmark_bearing_angle, landmark_id, marker_pixel_size):
    #     """
    #     Handle landmark observation - supports both unique IDs and data association.
    #     Args:
    #         landmark_dist: Distance to the landmark.
    #         landmark_bearing_angle: Bearing angle to the landmark.
    #         landmark_id: Identifier of the landmark (-1 means use data association).
    #         marker_pixel_size: Size of detected marker in pixels
    #     """
        
    #     if landmark_id == -1:  # Use data association for identical markers
    #         # Find best landmark match using Mahalanobis distance
    #         best_id, should_create = self.find_best_landmark_association(
    #             landmark_dist, landmark_bearing_angle, marker_pixel_size)
            
    #         if should_create:
    #             # Create new landmark with auto-generated ID
    #             new_id = str(len(self.landmarks))
    #             self.create_landmark(landmark_dist, landmark_bearing_angle, new_id, marker_pixel_size)
    #         else:
    #             # Update existing landmark
    #             self.update_landmark(landmark_dist, landmark_bearing_angle, best_id, marker_pixel_size)
        
    #     else:  # Use provided unique ID (your original approach)
    #         landmark_id = str(landmark_id)
    #         if landmark_id not in self.landmarks:
    #             self.create_landmark(landmark_dist, landmark_bearing_angle, landmark_id, marker_pixel_size)
    #         else:
    #             self.update_landmark(landmark_dist, landmark_bearing_angle, landmark_id, marker_pixel_size)

    def handle_landmark(self, landmark_dist, landmark_bearing_angle, landmark_id, marker_pixel_size):
        """
        Handle landmark using measurement-based association
        """
        # Use measurement-based association (the smart way!)
        best_id, should_create = self.find_best_landmark_association_measurement_based(
            landmark_dist, landmark_bearing_angle, marker_pixel_size)
        
        if should_create:
            # Create new landmark
            virtual_id = self.assign_virtual_id()
            print(f"🆕 Creating new landmark: {virtual_id}")
            self.create_landmark(landmark_dist, landmark_bearing_angle, virtual_id, marker_pixel_size)
        else:
            # Update existing landmark
            print(f"🔄 Updating existing landmark: {best_id}")
            self.update_landmark(landmark_dist, landmark_bearing_angle, best_id, marker_pixel_size)

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