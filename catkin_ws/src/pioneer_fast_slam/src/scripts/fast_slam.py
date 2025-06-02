import pygame
import math
import numpy as np
import copy
import tf.transformations
import rospy
import yaml
import os
import time
from scipy.spatial.distance import cdist
# ROS message types for landmark visualization
from visualization_msgs.msg import Marker, MarkerArray
from utils import resample, normalize_angle
# custom Particle class
from particle import Particle

class FastSlam:
    def __init__(self, tuning_option, window_size_pixel, size_m, pioneer_L, num_particles=50, screen=None, resample_method="low variance"):
        print(f"DEBUG: FastSlam.__init__ called with num_particles = {num_particles}")
        # Initialize various parameters and settings
        self.tuning_options = tuning_option
        self.SCREEN_WIDTH = window_size_pixel
        self.SCREEN_HEIGHT = window_size_pixel

        # Set up the pygame screen
        if screen is None:
            pygame.init()
            pygame.display.init()
            screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            pygame.display.set_caption("Pioneer FastSLAM")
            self.left_coordinate = 0
            self.right_coordinate = self.SCREEN_WIDTH
        else:
            self.left_coordinate = 0
            self.right_coordinate = self.SCREEN_WIDTH
        
        # choose resanpling method
        self.resample_method = resample_method
        
        # Define colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 160, 0)
        self.BLUE = (10, 10, 255)
        self.RED = (170, 0, 0)
        self.ORANGE = (255, 165, 0)  # For ground truth markers
        self.PURPLE = (128, 0, 128)  # For estimated markers
        self.CYAN = (0, 255, 255)    # For trajectory
        self.LIME = (50, 205, 50)    # For aligned landmarks
        self.YELLOW = (255, 255, 0)  # For computational performance metrics

        # Screen and map dimensions
        self.screen = screen      
        self.width_meters = size_m 
        self.height_meters = size_m
        # Pioneer P3-DX radius (approximately 44cm diameter)
        self.pioneer_radius = 0.22
        # robot wheel base
        self.pioneer_L = pioneer_L
        # convert radius to pixels
        self.pioneer_radius_pixel = int(self.pioneer_radius * self.SCREEN_WIDTH / self.width_meters)

        # Initialize SLAM-related variables
        self.old_odometry = [0.0, 0.0]
        self.old_yaw = 0
        self.num_particles = num_particles
        print(f"DEBUG: self.num_particles set to = {self.num_particles}")
        # index of best particle
        self.best_particle_ID = -1
        # create initial particles
        self.particles = self.initialize_particles()
        # Ros publisher for landmarks
        self.landmark_pub = rospy.Publisher('/landmarks', MarkerArray, queue_size=10)
        
        # ATE and evaluation variables
        self.ground_truth_markers = {}
        self.robot_trajectory = []  # Store robot trajectory for ATE calculation
        self.ground_truth_trajectory = []  # If available
        self.load_ground_truth_markers()
        
        # Performance metrics
        self.ate_values = []
        self.landmark_errors = {}
        
        # Kabsch alignment and SSE metrics
        self.sse_values = []
        self.rmse_values = []
        self.per_landmark_errors = {}
        self.alignment_transformation = None
        self.aligned_landmarks = {}
        self.detection_rate_history = []
        
        # NEW: Effective Particle Count tracking
        self.n_eff_history = []
        self.n_eff_threshold = num_particles / 2  # Common threshold for resampling
        
        # NEW: Computational Time Analysis
        self.timing_history = {
            'total_update': [],
            'motion_update': [],
            'landmark_update': [],
            'resampling': [],
            'weight_calculation': [],
            'measurement_processing': []
        }
        
        # Real-time performance tracking
        self.update_count = 0
        self.start_time = time.time()
        self.expected_update_rate = 10.0  # Expected Hz (adjust based on your rosbag rate)
        
        # Landmark tracking stability
        self.landmark_stability = {}  # Track how long each landmark has been consistently detected
        self.consecutive_detections = {}  # Count consecutive detections per landmark
        self.landmark_first_seen = {}  # When each landmark was first detected
        
        # Font for displaying metrics
        pygame.font.init()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        self.update_screen()  # Update screen with initial state
        return
    
    def load_ground_truth_markers(self):
        """Load ground truth marker positions from YAML file."""
        try:
            # Try to find the YAML file in the scripts directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            yaml_path = os.path.join(script_dir, 'ground_truth_markers.yaml')
            
            if not os.path.exists(yaml_path):
                rospy.logwarn(f"Ground truth file not found at {yaml_path}")
                rospy.logwarn("Creating example ground_truth_markers.yaml file...")
                self.create_example_yaml(yaml_path)
                return
            
            with open(yaml_path, 'r') as file:
                data = yaml.safe_load(file)
                
            if 'markers' in data:
                for marker_id, coords in data['markers'].items():
                    self.ground_truth_markers[str(marker_id)] = coords
                rospy.loginfo(f"Loaded {len(self.ground_truth_markers)} ground truth markers")
            else:
                rospy.logwarn("No 'markers' section found in ground truth YAML")
                
        except Exception as e:
            rospy.logerr(f"Error loading ground truth markers: {e}")
    
    def create_example_yaml(self, yaml_path):
        """Create an example YAML file for ground truth markers."""
        example_data = {
            'markers': {
                0: [1.5, 2.0],
                1: [-1.0, 1.5],
                2: [2.5, -1.0],
                3: [-2.0, -1.5],
                4: [0.5, -2.5]
            },
            'robot_start': [0.0, 0.0]
        }
        
        try:
            with open(yaml_path, 'w') as file:
                yaml.dump(example_data, file, default_flow_style=False)
            rospy.loginfo(f"Created example ground truth file at {yaml_path}")
            rospy.loginfo("Please edit this file with your actual marker coordinates!")
        except Exception as e:
            rospy.logerr(f"Error creating example YAML: {e}")
    
    def calculate_effective_particle_count(self, weights):
        """
        Calculate effective particle count (n_eff).
        
        Args:
            weights: Array of normalized particle weights
            
        Returns:
            n_eff: Effective particle count
        """
        # Ensure weights are normalized
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            normalized_weights = weights / weight_sum
        else:
            normalized_weights = np.ones(len(weights)) / len(weights)
        
        # Calculate n_eff = 1 / Σ(w_i²)
        n_eff = 1.0 / np.sum(normalized_weights ** 2)
        return n_eff
    
    def update_landmark_stability(self, landmarks_in_sight):
        """
        Update landmark stability tracking.
        
        Args:
            landmarks_in_sight: List of currently detected landmarks
        """
        current_time = time.time()
        detected_ids = set()
        
        # Process currently detected landmarks
        for landmark in landmarks_in_sight:
            _, _, landmark_id = landmark
            landmark_id = str(landmark_id)
            detected_ids.add(landmark_id)
            
            # Update consecutive detection count
            if landmark_id in self.consecutive_detections:
                self.consecutive_detections[landmark_id] += 1
            else:
                self.consecutive_detections[landmark_id] = 1
                self.landmark_first_seen[landmark_id] = current_time
            
            # Update stability score (how long it's been consistently detected)
            if landmark_id in self.landmark_first_seen:
                time_since_first = current_time - self.landmark_first_seen[landmark_id]
                self.landmark_stability[landmark_id] = {
                    'consecutive_detections': self.consecutive_detections[landmark_id],
                    'time_tracked': time_since_first,
                    'last_seen': current_time
                }
        
        # Reset consecutive count for landmarks not seen this update
        for landmark_id in list(self.consecutive_detections.keys()):
            if landmark_id not in detected_ids:
                self.consecutive_detections[landmark_id] = 0
    
    def kabsch_alignment(self, estimated_points, ground_truth_points):
        """
        Perform Kabsch alignment to find optimal rigid transformation.
        
        Args:
            estimated_points: Nx2 numpy array of estimated landmark positions
            ground_truth_points: Nx2 numpy array of ground truth landmark positions
            
        Returns:
            R: 2x2 rotation matrix
            t: 2x1 translation vector
            aligned_points: Transformed estimated points
        """
        if len(estimated_points) < 2 or len(ground_truth_points) < 2:
            return np.eye(2), np.zeros(2), estimated_points
        
        # Convert to numpy arrays
        P = np.array(estimated_points).T  # 2xN (estimated)
        Q = np.array(ground_truth_points).T  # 2xN (ground truth)
        
        # Center the point sets
        centroid_P = np.mean(P, axis=1, keepdims=True)
        centroid_Q = np.mean(Q, axis=1, keepdims=True)
        
        P_centered = P - centroid_P
        Q_centered = Q - centroid_Q
        
        # Compute cross-covariance matrix
        H = P_centered @ Q_centered.T
        
        # Use SVD to find optimal rotation
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Ensure proper rotation matrix (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Find optimal translation
        t = centroid_Q.flatten() - R @ centroid_P.flatten()
        
        # Apply transformation to estimated points
        aligned_points = (R @ P + t.reshape(-1, 1)).T
        
        return R, t, aligned_points
    
    def calculate_sse_metrics(self):
        """Calculate SSE and related metrics after Kabsch alignment."""
        best_particle = self.get_best_particle()
        
        if not best_particle.landmarks or not self.ground_truth_markers:
            return {
                'sse': 0.0,
                'rmse': 0.0,
                'mean_error': 0.0,
                'per_landmark_errors': {},
                'detection_rate': 0.0,
                'num_detected': 0,
                'num_ground_truth': len(self.ground_truth_markers)
            }
        
        # Find common landmarks (detected landmarks that have ground truth)
        common_landmarks = []
        estimated_positions = []
        ground_truth_positions = []
        
        for marker_id, landmark in best_particle.landmarks.items():
            if marker_id in self.ground_truth_markers:
                common_landmarks.append(marker_id)
                estimated_positions.append([landmark.x, landmark.y])
                ground_truth_positions.append(self.ground_truth_markers[marker_id])
        
        if len(common_landmarks) < 2:
            # Not enough landmarks for meaningful alignment
            return {
                'sse': 0.0,
                'rmse': 0.0,
                'mean_error': 0.0,
                'per_landmark_errors': {},
                'detection_rate': len(common_landmarks) / len(self.ground_truth_markers) * 100,
                'num_detected': len(common_landmarks),
                'num_ground_truth': len(self.ground_truth_markers)
            }
        
        # Perform Kabsch alignment
        R, t, aligned_positions = self.kabsch_alignment(estimated_positions, ground_truth_positions)
        
        # Store alignment transformation for visualization
        self.alignment_transformation = (R, t)
        
        # Calculate SSE and per-landmark errors
        sse = 0.0
        per_landmark_errors = {}
        aligned_landmarks_dict = {}
        
        for i, marker_id in enumerate(common_landmarks):
            # Aligned estimated position
            aligned_x, aligned_y = aligned_positions[i]
            aligned_landmarks_dict[marker_id] = [aligned_x, aligned_y]
            
            # Ground truth position
            gt_x, gt_y = ground_truth_positions[i]
            
            # Calculate squared error
            error_squared = (aligned_x - gt_x)**2 + (aligned_y - gt_y)**2
            sse += error_squared
            
            # Store per-landmark error
            per_landmark_errors[marker_id] = math.sqrt(error_squared)
        
        # Store aligned landmarks for visualization
        self.aligned_landmarks = aligned_landmarks_dict
        
        # Calculate derived metrics
        num_landmarks = len(common_landmarks)
        rmse = math.sqrt(sse / num_landmarks) if num_landmarks > 0 else 0.0
        mean_error = sum(per_landmark_errors.values()) / num_landmarks if num_landmarks > 0 else 0.0
        detection_rate = len(common_landmarks) / len(self.ground_truth_markers) * 100
        
        return {
            'sse': sse,
            'rmse': rmse,
            'mean_error': mean_error,
            'per_landmark_errors': per_landmark_errors,
            'detection_rate': detection_rate,
            'num_detected': len(common_landmarks),
            'num_ground_truth': len(self.ground_truth_markers),
            'transformation': (R, t)
        }
    
    def calculate_ate(self):
        """Calculate Absolute Trajectory Error."""
        if len(self.robot_trajectory) < 2:
            return 0.0
        
        # Get estimated trajectory from best particle
        best_particle = self.get_best_particle()
        if not hasattr(best_particle, 'trajectory') or len(best_particle.trajectory) == 0:
            return 0.0
        
        # Calculate ATE as RMSE between estimated and true positions
        total_error = 0.0
        count = 0
        
        # Use the stored robot trajectory (from odometry) as ground truth
        min_len = min(len(self.robot_trajectory), len(best_particle.trajectory))
        
        for i in range(min_len):
            true_x, true_y, _ = self.robot_trajectory[i]
            est_x, est_y, _ = best_particle.trajectory[i]
            
            error = math.sqrt((true_x - est_x)**2 + (true_y - est_y)**2)
            total_error += error**2
            count += 1
        
        if count > 0:
            ate = math.sqrt(total_error / count)
            return ate
        return 0.0
    
    # Publish landmark positions to a ROS topic
    def publish_landmarks(self):
        # create ROS marker array message
        marker_array = MarkerArray()
        for landmark_id, landmark in self.particles[self.best_particle_ID].landmarks.items():
            # extract landmark coordinates
            landmark_x, landmark_y = landmark.x, landmark.y

            # Create a marker for the landmark
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "landmarks"
            marker.id = int(landmark_id)
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = -landmark_x
            marker.pose.position.y = landmark_y
            marker.pose.position.z = 0  # Assuming 2D landmarks
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.a = 1.0  # Set the alpha
            marker.color.r = 0.0
            marker.color.g = 255.0
            marker.color.b = 0.0

            # Add the marker to the array message
            marker_array.markers.append(marker)

            # Create a marker for the text
            text_marker = Marker()
            text_marker.header.frame_id = "map"
            text_marker.header.stamp = rospy.Time.now()
            text_marker.ns = "landmark_text"
            text_marker.id = int(landmark_id) + 1000  # Ensure unique ID for text
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = -landmark_x
            text_marker.pose.position.y = landmark_y
            text_marker.pose.position.z = 0.5  # Slightly above the landmark
            text_marker.pose.orientation.x = 0.0
            text_marker.pose.orientation.y = 0.0
            text_marker.pose.orientation.z = 0.0
            text_marker.pose.orientation.w = 1.0
            text_marker.scale.z = 0.4  # Text height
            text_marker.color.a = 1.0  # Set the alpha
            text_marker.color.r = 1.0
            text_marker.color.g = 255.0
            text_marker.color.b = 1.0
            text_marker.text = f"id: {landmark_id}"

            # Add the text marker to the array
            marker_array.markers.append(text_marker)

        # Publish all markers (full marker array) to ROS
        self.landmark_pub.publish(marker_array)
    
    # Get the particle with the best (highest) weight
    def get_best_particle(self):
        return self.particles[self.best_particle_ID]
    
    # Initialize particles with random poses and empty landmarks (call Particle class)
    def initialize_particles(self, landmarks={}):
        print(f"DEBUG: initialize_particles called, creating {self.num_particles} particles")
        particles = []
        for _ in range(self.num_particles):
            x = 0
            y = 0
            theta = 0
            pose = np.array([x, y, theta])
            particles.append(Particle(pose, self.num_particles, self.pioneer_L, self.tuning_options))
        print(f"DEBUG: Created {len(particles)} particles")
        return particles
    
    # Update the particles based on odometry data
    def update_odometry(self, odometry):
        motion_start_time = time.time()
        
        # extract quarternion
        quaternion = [odometry[2][0], odometry[2][1], odometry[2][2], odometry[2][3]]
        # convert to euler angles
        (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(quaternion)
        # normalize yaw to -pi to pi
        yaw = normalize_angle(yaw)
    
        # Store robot trajectory for ATE calculation
        self.robot_trajectory.append((odometry[0], odometry[1], yaw))
    
        # calculate motion deltas (robot motion can be defined into three parameters)
        delta_dist = math.sqrt((odometry[0] - self.old_odometry[0])**2 + (odometry[1] - self.old_odometry[1])**2)
        delta_rot1 = normalize_angle(math.atan2(odometry[1] - self.old_odometry[1], odometry[0] - self.old_odometry[0]) - self.old_yaw)
        delta_rot2 = normalize_angle(yaw - self.old_yaw - delta_rot1)

        # Update each particle with motion and observation models
        for particle in self.particles:
            particle.motion_model([delta_dist, delta_rot1, delta_rot2])
        
        # Update old odometry and yaw with current values
        self.old_odometry = copy.deepcopy(odometry)
        self.old_yaw = copy.deepcopy(yaw)
        
        # Record motion update timing
        motion_time = time.time() - motion_start_time
        self.timing_history['motion_update'].append(motion_time)
        
        self.update_screen()

    # Perform SLAM computation based on observed landmarks
    def compute_slam(self, landmarks_in_sight):
        total_update_start_time = time.time()
        
        # Update landmark stability tracking
        self.update_landmark_stability(landmarks_in_sight)
        
        # Measurement processing timing
        measurement_start_time = time.time()
        
        # Initialize weights list
        weights_here = []
        landmark_update_start_time = time.time()
        
        # for each observed landmark
        for landmark in landmarks_in_sight:
            # extract landmark data from tuple
            landmark_dist, landmark_bearing_angle, landmark_id = landmark
            # get first particle pose (unused)
            x, y, theta = self.particles[0].pose
            # reset weight list
            weights_here = []
            
            weight_calc_start_time = time.time()
            for particle in self.particles:
                # update particle with landmark
                particle.handle_landmark(landmark_dist, math.radians(landmark_bearing_angle), landmark_id)
                # collect particle weight
                weights_here.append(particle.weight)
            
            # Record weight calculation timing
            weight_calc_time = time.time() - weight_calc_start_time
            self.timing_history['weight_calculation'].append(weight_calc_time)
        
        # Record landmark update timing
        landmark_update_time = time.time() - landmark_update_start_time
        self.timing_history['landmark_update'].append(landmark_update_time)
        
        # Record measurement processing timing
        measurement_time = time.time() - measurement_start_time
        self.timing_history['measurement_processing'].append(measurement_time)
        
        # Calculate effective particle count before resampling
        weights = np.array([p.weight for p in self.particles])
        n_eff = self.calculate_effective_particle_count(weights)
        self.n_eff_history.append(n_eff)
        
        # Resample particles based on their weights (with timing)
        resample_start_time = time.time()
        self.particles, self.best_particle_ID = resample(self.particles, self.num_particles, self.resample_method, self.best_particle_ID)
        resample_time = time.time() - resample_start_time
        self.timing_history['resampling'].append(resample_time)
        
        # Calculate and store current ATE
        current_ate = self.calculate_ate()
        self.ate_values.append(current_ate)
        
        # Calculate and store SSE metrics
        sse_metrics = self.calculate_sse_metrics()
        self.sse_values.append(sse_metrics['sse'])
        self.rmse_values.append(sse_metrics['rmse'])
        self.per_landmark_errors = sse_metrics['per_landmark_errors']
        self.detection_rate_history.append(sse_metrics['detection_rate'])
        
        # Record total update timing
        total_update_time = time.time() - total_update_start_time
        self.timing_history['total_update'].append(total_update_time)
        
        # Increment update count for real-time performance tracking
        self.update_count += 1
        
        self.update_screen(landmarks_in_sight)

    def get_timing_statistics(self):
        """Calculate timing statistics for performance analysis."""
        stats = {}
        
        for operation, times in self.timing_history.items():
            if times:
                stats[operation] = {
                    'mean': np.mean(times) * 1000,  # Convert to milliseconds
                    'std': np.std(times) * 1000,
                    'min': np.min(times) * 1000,
                    'max': np.max(times) * 1000,
                    'count': len(times)
                }
            else:
                stats[operation] = {
                    'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0
                }
        
        # Calculate real-time factor
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            actual_rate = self.update_count / elapsed_time
            real_time_factor = actual_rate / self.expected_update_rate
        else:
            real_time_factor = 0.0
        
        stats['real_time_performance'] = {
            'actual_rate_hz': actual_rate if elapsed_time > 0 else 0,
            'expected_rate_hz': self.expected_update_rate,
            'real_time_factor': real_time_factor,
            'total_updates': self.update_count,
            'elapsed_time': elapsed_time
        }
        
        return stats

    def get_particle_diversity_stats(self):
        """Calculate particle diversity statistics."""
        if not self.n_eff_history:
            return {
                'current_n_eff': 0,
                'mean_n_eff': 0,
                'min_n_eff': 0,
                'max_n_eff': 0,
                'n_eff_ratio': 0,
                'resampling_frequency': 0
            }
        
        current_n_eff = self.n_eff_history[-1] if self.n_eff_history else 0
        mean_n_eff = np.mean(self.n_eff_history)
        min_n_eff = np.min(self.n_eff_history)
        max_n_eff = np.max(self.n_eff_history)
        
        # Calculate how often n_eff falls below threshold (indicates resampling)
        below_threshold = sum(1 for n in self.n_eff_history if n < self.n_eff_threshold)
        resampling_frequency = below_threshold / len(self.n_eff_history) if self.n_eff_history else 0
        
        return {
            'current_n_eff': current_n_eff,
            'mean_n_eff': mean_n_eff,
            'min_n_eff': min_n_eff,
            'max_n_eff': max_n_eff,
            'n_eff_ratio': current_n_eff / self.num_particles,
            'resampling_frequency': resampling_frequency * 100  # As percentage
        }

    def draw_ground_truth_markers(self):
        """Draw ground truth markers on the screen."""
        for marker_id, (gt_x, gt_y) in self.ground_truth_markers.items():
            # Convert to pixel coordinates
            pixel_x = int(gt_x * self.SCREEN_WIDTH / self.width_meters + self.left_coordinate + self.SCREEN_WIDTH / 2)
            pixel_y = int(gt_y * self.SCREEN_HEIGHT / self.height_meters + self.SCREEN_HEIGHT / 2)
            
            # Draw ground truth marker as orange square
            pygame.draw.rect(self.screen, self.ORANGE, 
                           (pixel_x - 6, pixel_y - 6, 12, 12))
            
            # Draw ID text
            text_surface = self.small_font.render(f"GT{marker_id}", True, self.ORANGE)
            text_rect = text_surface.get_rect(center=(pixel_x, pixel_y - 20))
            self.screen.blit(text_surface, text_rect)

    def draw_estimated_markers(self):
        """Draw estimated markers from best particle."""
        best_particle = self.get_best_particle()
        for landmark_id, landmark in best_particle.landmarks.items():
            # get landmark position
            landmark_x, landmark_y = landmark.x, landmark.y
            
            # Convert to pixel coordinates
            pixel_x = int(landmark_x * self.SCREEN_WIDTH / self.width_meters + self.left_coordinate + self.SCREEN_WIDTH / 2)
            pixel_y = int(landmark_y * self.SCREEN_HEIGHT / self.height_meters + self.SCREEN_HEIGHT / 2)
            
            # Draw estimated marker as purple circle
            pygame.draw.circle(self.screen, self.PURPLE, (pixel_x, pixel_y), 5)
            
            # Draw ID text
            text_surface = self.small_font.render(f"EST{landmark_id}", True, self.PURPLE)
            text_rect = text_surface.get_rect(center=(pixel_x, pixel_y + 20))
            self.screen.blit(text_surface, text_rect)
            
            # Draw error line if ground truth exists
            if landmark_id in self.ground_truth_markers:
                gt_x, gt_y = self.ground_truth_markers[landmark_id]
                gt_pixel_x = int(gt_x * self.SCREEN_WIDTH / self.width_meters + self.left_coordinate + self.SCREEN_WIDTH / 2)
                gt_pixel_y = int(gt_y * self.SCREEN_HEIGHT / self.height_meters + self.SCREEN_HEIGHT / 2)
                
                # Draw error line (before alignment)
                pygame.draw.line(self.screen, self.RED, (pixel_x, pixel_y), (gt_pixel_x, gt_pixel_y), 1)

    def draw_aligned_markers(self):
        """Draw Kabsch-aligned markers in a different color."""
        if not self.aligned_landmarks:
            return
            
        for marker_id, (aligned_x, aligned_y) in self.aligned_landmarks.items():
            # Convert to pixel coordinates
            pixel_x = int(aligned_x * self.SCREEN_WIDTH / self.width_meters + self.left_coordinate + self.SCREEN_WIDTH / 2)
            pixel_y = int(aligned_y * self.SCREEN_HEIGHT / self.height_meters + self.SCREEN_HEIGHT / 2)
            
            # Draw aligned marker as lime green diamond
            diamond_points = [
                (pixel_x, pixel_y - 6),  # top
                (pixel_x + 6, pixel_y),  # right
                (pixel_x, pixel_y + 6),  # bottom
                (pixel_x - 6, pixel_y)   # left
            ]
            pygame.draw.polygon(self.screen, self.LIME, diamond_points)
            
            # Draw ID text
            text_surface = self.small_font.render(f"ALG{marker_id}", True, self.LIME)
            text_rect = text_surface.get_rect(center=(pixel_x + 15, pixel_y))
            self.screen.blit(text_surface, text_rect)
            
            # Draw aligned error line to ground truth
            if marker_id in self.ground_truth_markers:
                gt_x, gt_y = self.ground_truth_markers[marker_id]
                gt_pixel_x = int(gt_x * self.SCREEN_WIDTH / self.width_meters + self.left_coordinate + self.SCREEN_WIDTH / 2)
                gt_pixel_y = int(gt_y * self.SCREEN_HEIGHT / self.height_meters + self.SCREEN_HEIGHT / 2)
                
                # Draw post-alignment error line (should be much shorter!)
                pygame.draw.line(self.screen, self.LIME, (pixel_x, pixel_y), (gt_pixel_x, gt_pixel_y), 2)

    def draw_trajectory(self):
        """Draw robot trajectory."""
        if len(self.robot_trajectory) > 1:
            points = []
            for x, y, _ in self.robot_trajectory:
                pixel_x = int(x * self.SCREEN_WIDTH / self.width_meters + self.left_coordinate + self.SCREEN_WIDTH / 2)
                pixel_y = int(y * self.SCREEN_HEIGHT / self.height_meters + self.SCREEN_HEIGHT / 2)
                points.append((pixel_x, pixel_y))
            
            if len(points) > 1:
                pygame.draw.lines(self.screen, self.CYAN, False, points, 2)

    def draw_metrics_panel(self):
        """Draw comprehensive performance metrics on the screen."""
        panel_x = 10
        panel_y = 10
        panel_width = 500  # Increased width for more metrics
        panel_height = 450  # Increased height for more metrics
        
        # Draw background
        pygame.draw.rect(self.screen, (0, 0, 0, 180), (panel_x, panel_y, panel_width, panel_height))
        pygame.draw.rect(self.screen, self.WHITE, (panel_x, panel_y, panel_width, panel_height), 2)
        
        y_offset = panel_y + 10
        
        # Title
        title = self.font.render("SLAM Performance Metrics", True, self.WHITE)
        self.screen.blit(title, (panel_x + 10, y_offset))
        y_offset += 30
        
        # ATE Metrics
        current_ate = self.ate_values[-1] if self.ate_values else 0.0
        ate_text = self.font.render(f"MPD: {current_ate:.3f}m", True, self.WHITE)
        self.screen.blit(ate_text, (panel_x + 10, y_offset))
        y_offset += 20
        
        # SSE Metrics
        if self.sse_values:
            current_sse = self.sse_values[-1]
            sse_text = self.font.render(f"SSE: {current_sse:.3f}m²", True, self.WHITE)
            self.screen.blit(sse_text, (panel_x + 10, y_offset))
            y_offset += 20
            
        if self.rmse_values:
            current_rmse = self.rmse_values[-1]
            rmse_text = self.font.render(f"RMSE: {current_rmse:.3f}m", True, self.WHITE)
            self.screen.blit(rmse_text, (panel_x + 10, y_offset))
            y_offset += 20
        
        # Detection Rate
        if self.detection_rate_history:
            current_detection_rate = self.detection_rate_history[-1]
            detection_text = self.font.render(f"Detection Rate: {current_detection_rate:.1f}%", True, self.WHITE)
            self.screen.blit(detection_text, (panel_x + 10, y_offset))
            y_offset += 25
        
        # NEW: Effective Particle Count Section
        diversity_stats = self.get_particle_diversity_stats()
        n_eff_title = self.font.render("Particle Diversity:", True, self.CYAN)
        self.screen.blit(n_eff_title, (panel_x + 10, y_offset))
        y_offset += 20
        
        n_eff_text = self.font.render(f"n_eff: {diversity_stats['current_n_eff']:.1f}/{self.num_particles}", True, self.WHITE)
        self.screen.blit(n_eff_text, (panel_x + 20, y_offset))
        y_offset += 18
        
        n_eff_ratio_text = self.font.render(f"Diversity: {diversity_stats['n_eff_ratio']:.2%}", True, self.WHITE)
        self.screen.blit(n_eff_ratio_text, (panel_x + 20, y_offset))
        y_offset += 18
        
        resample_freq_text = self.font.render(f"Resample Freq: {diversity_stats['resampling_frequency']:.1f}%", True, self.WHITE)
        self.screen.blit(resample_freq_text, (panel_x + 20, y_offset))
        y_offset += 25
        
        # NEW: Computational Timing Section
        timing_stats = self.get_timing_statistics()
        timing_title = self.font.render("Computational Performance:", True, self.YELLOW)
        self.screen.blit(timing_title, (panel_x + 10, y_offset))
        y_offset += 20
        
        # Total update time
        if 'total_update' in timing_stats and timing_stats['total_update']['count'] > 0:
            total_time_text = self.font.render(f"Total: {timing_stats['total_update']['mean']:.1f}ms", True, self.WHITE)
            self.screen.blit(total_time_text, (panel_x + 20, y_offset))
            y_offset += 18
        
        # Motion update time
        if 'motion_update' in timing_stats and timing_stats['motion_update']['count'] > 0:
            motion_time_text = self.font.render(f"Motion: {timing_stats['motion_update']['mean']:.1f}ms", True, self.WHITE)
            self.screen.blit(motion_time_text, (panel_x + 20, y_offset))
            y_offset += 18
        
        # Landmark update time
        if 'landmark_update' in timing_stats and timing_stats['landmark_update']['count'] > 0:
            landmark_time_text = self.font.render(f"Landmarks: {timing_stats['landmark_update']['mean']:.1f}ms", True, self.WHITE)
            self.screen.blit(landmark_time_text, (panel_x + 20, y_offset))
            y_offset += 18
        
        # Real-time performance
        rt_stats = timing_stats.get('real_time_performance', {})
        if rt_stats:
            rt_factor_text = self.font.render(f"RT Factor: {rt_stats.get('real_time_factor', 0):.2f}x", True, self.WHITE)
            self.screen.blit(rt_factor_text, (panel_x + 20, y_offset))
            y_offset += 18
            
            rate_text = self.font.render(f"Rate: {rt_stats.get('actual_rate_hz', 0):.1f}Hz", True, self.WHITE)
            self.screen.blit(rate_text, (panel_x + 20, y_offset))
            y_offset += 25
        
        # Landmark Stability
        stability_title = self.font.render("Landmark Stability:", True, self.GREEN)
        self.screen.blit(stability_title, (panel_x + 10, y_offset))
        y_offset += 20
        
        # Show stability for detected landmarks
        stable_count = 0
        for marker_id, stability in self.landmark_stability.items():
            if stability['consecutive_detections'] >= 5:  # Consider stable if detected 5+ times consecutively
                stable_count += 1
        
        stability_text = self.font.render(f"Stable Landmarks: {stable_count}/{len(self.landmark_stability)}", True, self.WHITE)
        self.screen.blit(stability_text, (panel_x + 20, y_offset))
        y_offset += 20
        
        # Number of trajectory points
        traj_text = self.font.render(f"Trajectory points: {len(self.robot_trajectory)}", True, self.WHITE)
        self.screen.blit(traj_text, (panel_x + 20, y_offset))

    # Update the display screen with the current state of particles and landmarks
    def update_screen(self, landmarks_in_sight=None):
        # if no best particle selected, select random particle
        if self.best_particle_ID == -1:
            self.best_particle_ID = np.random.randint(len(self.particles))

        # get best particles pose
        x, y, theta = self.particles[self.best_particle_ID].pose
        # convert from pixel to coordinates
        pioneer_pos = (int((x) * self.SCREEN_WIDTH / self.width_meters + self.left_coordinate + self.SCREEN_WIDTH / 2), 
                       int((y) * self.SCREEN_HEIGHT / self.height_meters + self.SCREEN_HEIGHT / 2)) 
        # triangle size (representing robot orientation)
        triangle_length = 0.8 * self.pioneer_radius_pixel
        # front point
        triangle_tip_x = pioneer_pos[0] + triangle_length * math.cos(theta)
        triangle_tip_y = pioneer_pos[1] - triangle_length * math.sin(theta)
        # left point
        triangle_left_x = pioneer_pos[0] + triangle_length * math.cos(theta + 5 * math.pi / 6) 
        triangle_left_y = pioneer_pos[1] - triangle_length * math.sin(theta + 5 * math.pi / 6)
        # right point 
        triangle_right_x = pioneer_pos[0] + triangle_length * math.cos(theta - 5 * math.pi / 6)
        triangle_right_y = pioneer_pos[1] - triangle_length * math.sin(theta - 5 * math.pi / 6)

        # Draw the triangle representing the robot's orientation and the circle representing robot body
        triangle_points = [(triangle_tip_x, triangle_tip_y), (triangle_left_x, triangle_left_y), (triangle_right_x, triangle_right_y)]
        half_screen_rect = pygame.Rect(self.left_coordinate, 0, self.right_coordinate, self.SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, self.WHITE, half_screen_rect)
        
        # Draw trajectory first (so it appears behind other elements)
        self.draw_trajectory()
        
        # Draw ground truth markers
        self.draw_ground_truth_markers()
        
        # Draw estimated markers
        self.draw_estimated_markers()
        
        # Draw Kabsch-aligned markers
        self.draw_aligned_markers()
        
        # Draw robot
        pygame.draw.circle(self.screen, self.GREEN, pioneer_pos, self.pioneer_radius_pixel)
        pygame.draw.polygon(self.screen, self.BLUE, triangle_points)

        # Draw the particles
        for particle in self.particles:
            # get particle pose
            particle_x, particle_y, _ = particle.pose
            pygame.draw.circle(self.screen, self.RED, (int((particle_x) * self.SCREEN_WIDTH / self.width_meters + self.left_coordinate + self.SCREEN_WIDTH / 2), 
                                                       int((particle_y) * self.SCREEN_HEIGHT / self.height_meters + self.SCREEN_HEIGHT / 2)), 3)

        # Draw comprehensive metrics panel
        self.draw_metrics_panel()
        
        # Draw legend
        self.draw_legend()
        
        pygame.display.flip()

    def draw_legend(self):
        """Draw a legend explaining the colors."""
        legend_x = self.SCREEN_WIDTH - 250
        legend_y = 10
        
        # Background
        pygame.draw.rect(self.screen, (0, 0, 0, 128), (legend_x, legend_y, 230, 160))
        pygame.draw.rect(self.screen, self.WHITE, (legend_x, legend_y, 230, 160), 1)
        
        y_pos = legend_y + 10
        
        # Title
        title = self.font.render("Legend:", True, self.WHITE)
        self.screen.blit(title, (legend_x + 10, y_pos))
        y_pos += 25
        
        # Legend items
        legend_items = [
            ("Ground Truth", self.ORANGE),
            ("Estimated", self.PURPLE),
            ("Kabsch Aligned", self.LIME),
            ("Robot", self.GREEN),
            ("Trajectory", self.CYAN),
            ("Error Lines", self.RED)
        ]
        
        for text, color in legend_items:
            # Draw color indicator
            pygame.draw.circle(self.screen, color, (legend_x + 15, y_pos + 8), 5)
            # Draw text
            text_surface = self.small_font.render(text, True, self.WHITE)
            self.screen.blit(text_surface, (legend_x + 30, y_pos))
            y_pos += 20

    # Get the best trajectory from the best particle
    def get_best_trajectory(self):
        return self.particles[self.best_particle_ID].trajectory
    
    def get_current_metrics(self):
        """Return comprehensive performance metrics."""
        sse_metrics = self.calculate_sse_metrics()
        timing_stats = self.get_timing_statistics()
        diversity_stats = self.get_particle_diversity_stats()
        
        metrics = {
            # Existing metrics
            'current_mpd': self.ate_values[-1] if self.ate_values else 0.0,
            'average_mpd': sum(self.ate_values) / len(self.ate_values) if self.ate_values else 0.0,
            'current_sse': self.sse_values[-1] if self.sse_values else 0.0,
            'current_rmse': self.rmse_values[-1] if self.rmse_values else 0.0,
            'average_sse': sum(self.sse_values) / len(self.sse_values) if self.sse_values else 0.0,
            'average_rmse': sum(self.rmse_values) / len(self.rmse_values) if self.rmse_values else 0.0,
            'per_landmark_errors': self.per_landmark_errors.copy(),
            'detection_rate': sse_metrics['detection_rate'],
            'landmarks_detected': sse_metrics['num_detected'],
            'total_ground_truth': sse_metrics['num_ground_truth'],
            'trajectory_length': len(self.robot_trajectory),
            'particles_count': self.num_particles,
            
            # NEW: Particle diversity metrics
            'effective_particle_count': diversity_stats,
            
            # NEW: Computational performance metrics
            'timing_performance': timing_stats,
            
            # NEW: Landmark stability metrics
            'landmark_stability': {
                'total_landmarks': len(self.landmark_stability),
                'stable_landmarks': sum(1 for s in self.landmark_stability.values() if s['consecutive_detections'] >= 5),
                'stability_details': self.landmark_stability.copy()
            }
        }
        return metrics
    
    def save_metrics_to_file(self, filename="slam_comprehensive_metrics.txt"):
        """Save comprehensive performance metrics to a file."""
        try:
            metrics = self.get_current_metrics()
            with open(filename, 'w') as f:
                f.write("FastSLAM Comprehensive Performance Analysis\n")
                f.write("=" * 60 + "\n\n")
                
                # Trajectory Error Analysis
                f.write("TRAJECTORY ERROR ANALYSIS\n")
                f.write("-" * 30 + "\n")
                f.write(f"Current ATE: {metrics['current_mpd']:.6f} meters\n")
                f.write(f"Average ATE: {metrics['average_mpd']:.6f} meters\n\n")
                
                # Mapping Accuracy (After Kabsch Alignment)
                f.write("MAPPING ACCURACY (After Kabsch Alignment)\n")
                f.write("-" * 45 + "\n")
                f.write(f"Current SSE: {metrics['current_sse']:.6f} m²\n")
                f.write(f"Current RMSE: {metrics['current_rmse']:.6f} meters\n")
                f.write(f"Average SSE: {metrics['average_sse']:.6f} m²\n")
                f.write(f"Average RMSE: {metrics['average_rmse']:.6f} meters\n\n")
                
                # Detection Performance
                f.write("DETECTION PERFORMANCE\n")
                f.write("-" * 20 + "\n")
                f.write(f"Detection Rate: {metrics['detection_rate']:.1f}%\n")
                f.write(f"Landmarks Detected: {metrics['landmarks_detected']}/{metrics['total_ground_truth']}\n\n")
                
                # NEW: Particle Filter Performance
                f.write("PARTICLE FILTER PERFORMANCE\n")
                f.write("-" * 30 + "\n")
                diversity = metrics['effective_particle_count']
                f.write(f"Total Particles: {metrics['particles_count']}\n")
                f.write(f"Current n_eff: {diversity['current_n_eff']:.2f}\n")
                f.write(f"Mean n_eff: {diversity['mean_n_eff']:.2f}\n")
                f.write(f"Min n_eff: {diversity['min_n_eff']:.2f}\n")
                f.write(f"Max n_eff: {diversity['max_n_eff']:.2f}\n")
                f.write(f"Diversity Ratio: {diversity['n_eff_ratio']:.2%}\n")
                f.write(f"Resampling Frequency: {diversity['resampling_frequency']:.1f}%\n\n")
                
                # NEW: Computational Performance
                f.write("COMPUTATIONAL PERFORMANCE\n")
                f.write("-" * 28 + "\n")
                timing = metrics['timing_performance']
                
                if 'real_time_performance' in timing:
                    rt = timing['real_time_performance']
                    f.write(f"Real-time Factor: {rt['real_time_factor']:.3f}x\n")
                    f.write(f"Actual Rate: {rt['actual_rate_hz']:.2f} Hz\n")
                    f.write(f"Expected Rate: {rt['expected_rate_hz']:.2f} Hz\n")
                    f.write(f"Total Updates: {rt['total_updates']}\n")
                    f.write(f"Elapsed Time: {rt['elapsed_time']:.2f} seconds\n\n")
                
                f.write("Timing Breakdown (mean ± std ms):\n")
                for operation, stats in timing.items():
                    if operation != 'real_time_performance' and stats['count'] > 0:
                        f.write(f"  {operation.replace('_', ' ').title()}: {stats['mean']:.2f} ± {stats['std']:.2f} ms\n")
                f.write("\n")
                
                # NEW: Landmark Stability Analysis
                f.write("LANDMARK STABILITY ANALYSIS\n")
                f.write("-" * 30 + "\n")
                stability = metrics['landmark_stability']
                f.write(f"Total Landmarks Tracked: {stability['total_landmarks']}\n")
                f.write(f"Stable Landmarks (5+ consecutive detections): {stability['stable_landmarks']}\n")
                if stability['total_landmarks'] > 0:
                    stability_rate = (stability['stable_landmarks'] / stability['total_landmarks']) * 100
                    f.write(f"Stability Rate: {stability_rate:.1f}%\n")
                f.write("\n")
                
                f.write("Per-Landmark Stability Details:\n")
                for marker_id, details in stability['stability_details'].items():
                    f.write(f"  Marker {marker_id}:\n")
                    f.write(f"    Consecutive detections: {details['consecutive_detections']}\n")
                    f.write(f"    Time tracked: {details['time_tracked']:.2f} seconds\n")
                f.write("\n")
                
                # Per-Landmark Errors (After Alignment)
                f.write("Per-Landmark Errors (After Alignment):\n")
                f.write("-" * 40 + "\n")
                for marker_id, error in metrics['per_landmark_errors'].items():
                    f.write(f"Marker {marker_id}: {error:.6f} meters\n")
                f.write("\n")
                
                # Historical Data Summary
                if self.sse_values:
                    f.write(f"Historical Performance Summary:\n")
                    f.write(f"SSE History (last 10 values):\n")
                    recent_sses = self.sse_values[-10:]
                    for i, sse in enumerate(recent_sses):
                        f.write(f"  {len(self.sse_values)-len(recent_sses)+i+1}: {sse:.6f} m²\n")
                f.write("\n")
                
                if self.n_eff_history:
                    f.write(f"n_eff History (last 10 values):\n")
                    recent_neff = self.n_eff_history[-10:]
                    for i, neff in enumerate(recent_neff):
                        f.write(f"  {len(self.n_eff_history)-len(recent_neff)+i+1}: {neff:.2f}\n")
                        
            rospy.loginfo(f"Comprehensive metrics saved to {filename}")
        except Exception as e:
            rospy.logerr(f"Error saving metrics: {e}")