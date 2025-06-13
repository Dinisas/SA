#!/usr/bin/env python3
import numpy as np
import math
import time
import yaml
import os
import rospy
from collections import defaultdict
from scipy.spatial import KDTree



class SLAMMetricsTracker:
    """
    Comprehensive metrics tracking for FastSLAM implementation.
    Enhanced with ground truth trajectory support, automatic theta calculation.
    ETA REMOVED, ATE FIXED to compare SLAM vs Ground Truth.
    """
    
    def __init__(self, num_particles, groundtruth_file, expected_update_rate=10.0):
        """
        Initialize the metrics tracker.
        
        Args:
            num_particles: Total number of particles in the filter
            expected_update_rate: Expected update rate in Hz for real-time performance tracking
        """
        self.num_particles = num_particles
        self.expected_update_rate = expected_update_rate

        # choose ground truth yaml file
        self.groundtruth_file = groundtruth_file
        
        # Ground truth data
        self.ground_truth_markers = {}
        self.ground_truth_trajectory = []  # Ground truth robot trajectory
        self.rotation_angle_degrees = 0.0  # Clockwise rotation angle from YAML
        self.load_ground_truth_data()
        
        # ATE and trajectory metrics (ETA REMOVED)
        self.ate_values = []
        self.msp_values = []  # Mean Squared Position error
        # REMOVED: self.eta_values = []  # ETA completely removed
        self.robot_trajectory = []
        
        # SSE and Kabsch alignment metrics
        self.sse_values = []
        self.rmse_values = []
        self.per_landmark_errors = {}
        self.alignment_transformation = None
        self.aligned_landmarks = {}
        self.detection_rate_history = []
        
        # Particle diversity metrics
        self.n_eff_history = []
        self.n_eff_threshold = num_particles / 2  # Common threshold for resampling
        
        # Computational timing metrics
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
        
        # Landmark stability tracking
        self.landmark_stability = {}
        self.consecutive_detections = {}
        self.landmark_first_seen = {}
        
        # Trajectory interpolation for alignment
        self.trajectory_interpolator = None
    
    def apply_clockwise_rotation(self, points, angle_degrees):
        """
        Apply clockwise rotation to a list of points.
        
        Args:
            points: List of [x, y] or [x, y, theta] points
            angle_degrees: Clockwise rotation angle in degrees
            
        Returns:
            List of rotated points maintaining original format
        """
        if angle_degrees == 0.0:
            return points
        
        # Convert to radians (negative for clockwise)
        angle_rad = -math.radians(angle_degrees)
        
        # Rotation matrix for clockwise rotation
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        rotated_points = []
        for point in points:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                x, y = point[0], point[1]
                
                # Apply rotation
                new_x = x * cos_a - y * sin_a
                new_y = x * sin_a + y * cos_a
                
                # Preserve original format
                if len(point) >= 3:
                    # If theta is present, rotate it too
                    theta = point[2] + angle_rad
                    rotated_points.append([new_x, new_y, theta])
                else:
                    rotated_points.append([new_x, new_y])
            else:
                # If point format is unexpected, keep as is
                rotated_points.append(point)
        
        return rotated_points
    
    def apply_clockwise_rotation_to_markers(self, markers_dict, angle_degrees):
        """
        Apply clockwise rotation to marker positions.
        
        Args:
            markers_dict: Dictionary of {id: [x, y]} marker positions
            angle_degrees: Clockwise rotation angle in degrees
            
        Returns:
            Dictionary with rotated marker positions
        """
        if angle_degrees == 0.0:
            return markers_dict
        
        # Convert to radians (negative for clockwise)
        angle_rad = -math.radians(angle_degrees)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        rotated_markers = {}
        for marker_id, pos in markers_dict.items():
            if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                x, y = pos[0], pos[1]
                new_x = x * cos_a - y * sin_a
                new_y = x * sin_a + y * cos_a
                rotated_markers[marker_id] = [new_x, new_y]
            else:
                rotated_markers[marker_id] = pos
        
        return rotated_markers
    
    def calculate_trajectory_orientation(self, positions):
        """
        Calculate orientation (theta) for trajectory positions based on movement direction.
        
        Args:
            positions: List of [x, y] or [x, y, theta] positions
            
        Returns:
            List of (x, y, theta) tuples with calculated orientations
        """
        if not positions or len(positions) < 1:
            return []
        
        trajectory_with_theta = []
        
        for i, pos in enumerate(positions):
            x, y = pos[0], pos[1]
            
            # If theta is already provided, use it
            if len(pos) >= 3:
                theta = pos[2]
            else:
                # Calculate theta from movement direction
                if i == 0:
                    # For the first point, look ahead to next point if available
                    if len(positions) > 1:
                        next_x, next_y = positions[1][0], positions[1][1]
                        theta = math.atan2(next_y - y, next_x - x)
                    else:
                        theta = 0.0  # Default orientation if only one point
                elif i == len(positions) - 1:
                    # For the last point, use direction from previous point
                    prev_x, prev_y = positions[i-1][0], positions[i-1][1]
                    theta = math.atan2(y - prev_y, x - prev_x)
                else:
                    # For middle points, use average of incoming and outgoing directions
                    prev_x, prev_y = positions[i-1][0], positions[i-1][1]
                    next_x, next_y = positions[i+1][0], positions[i+1][1]
                    
                    # Direction from previous point
                    theta_in = math.atan2(y - prev_y, x - prev_x)
                    # Direction to next point
                    theta_out = math.atan2(next_y - y, next_x - x)
                    
                    # Average the angles (handling angle wraparound)
                    theta = self.average_angles(theta_in, theta_out)
            
            trajectory_with_theta.append((x, y, theta))
        
        return trajectory_with_theta
    
    def average_angles(self, angle1, angle2):
        """
        Calculate the average of two angles, handling wraparound properly.
        
        Args:
            angle1, angle2: Angles in radians
            
        Returns:
            Average angle in radians
        """
        # Convert to unit vectors and average
        x = (math.cos(angle1) + math.cos(angle2)) / 2
        y = (math.sin(angle1) + math.sin(angle2)) / 2
        
        # Convert back to angle
        return math.atan2(y, x)
    
    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi] range."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def load_ground_truth_data(self):
        """Load ground truth marker positions and trajectory from YAML file with rotation support."""
        try:     
            with open(self.groundtruth_file, 'r') as file:
                data = yaml.safe_load(file)
            
            # Check for clockwise rotation parameter
            rotation_config = data.get('coordinate_transform', {})
            self.rotation_angle_degrees = rotation_config.get('clockwise_rotation_degrees', 0.0)
            
            if self.rotation_angle_degrees != 0.0:
                rospy.loginfo(f"Applying {self.rotation_angle_degrees}° clockwise rotation to ground truth data")
            
            # Load markers
            if 'markers' in data:
                raw_markers = {}
                for marker_id, coords in data['markers'].items():
                    if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                        raw_markers[str(marker_id)] = [coords[0], coords[1]]
                
                # Apply rotation to markers if specified
                self.ground_truth_markers = self.apply_clockwise_rotation_to_markers(
                    raw_markers, self.rotation_angle_degrees
                )
                
                rospy.loginfo(f"Loaded {len(self.ground_truth_markers)} ground truth markers")
            else:
                rospy.logwarn("No 'markers' section found in ground truth YAML")
            
            # Load trajectory if available
            if 'positions' in data:
                positions = data['positions']
                if isinstance(positions, list):
                    # Apply rotation to trajectory if specified
                    rotated_positions = self.apply_clockwise_rotation(positions, self.rotation_angle_degrees)
                    
                    # Process positions to ensure they have proper orientation
                    self.ground_truth_trajectory = self.calculate_trajectory_orientation(rotated_positions)
                    
                rospy.loginfo(f"Loaded {len(self.ground_truth_trajectory)} ground truth trajectory points")
                
                if self.rotation_angle_degrees != 0.0:
                    rospy.loginfo(f"Applied {self.rotation_angle_degrees}° clockwise rotation to trajectory")
                
                # Log some trajectory information for debugging
                if self.ground_truth_trajectory:
                    rospy.loginfo("First few trajectory points (after rotation):")
                    for i, (x, y, theta) in enumerate(self.ground_truth_trajectory[:3]):
                        rospy.loginfo(f"  Point {i}: x={x:.3f}, y={y:.3f}, θ={theta:.3f} ({math.degrees(theta):.1f}°)")
            else:
                rospy.logwarn("No 'positions' section found in ground truth YAML")
                
        except Exception as e:
            rospy.logerr(f"Error loading ground truth data: {e}")
            rospy.logerr(f"File path: {self.groundtruth_file}")
    
    def update_robot_trajectory(self, x, y, theta):
        """Add a new pose to the robot trajectory."""
        self.robot_trajectory.append((x, y, theta))
        
        # Calculate trajectory-based metrics if ground truth is available
        if self.ground_truth_trajectory and len(self.robot_trajectory) > 1:
            self.calculate_trajectory_metrics()
    
    def calculate_trajectory_metrics(self):
        """Calculate MSP metrics based on current trajectory (ETA REMOVED)."""
        if not self.ground_truth_trajectory or not self.robot_trajectory:
            return
        
        # Calculate MSP (Mean Squared Position error)
        msp = self.calculate_msp()
        if msp is not None:
            self.msp_values.append(msp)
        
        # REMOVED: ETA calculation completely deleted
    
    def calculate_msp(self):

        if not self.robot_trajectory or not self.ground_truth_trajectory:
            return None

        # Extract positions only (ignore theta)
        est_positions = np.array([(x, y) for x, y, _ in self.robot_trajectory])
        gt_positions = np.array([(x, y) for x, y, _ in self.ground_truth_trajectory])

        total_squared_error = 0.0

        for est in est_positions:
            # Compute Euclidean distances to all ground truth points
            dists = np.linalg.norm(gt_positions - est, axis=1)
            nearest_dist = np.min(dists)
            total_squared_error += nearest_dist ** 2

        msp = total_squared_error / len(est_positions)
        return msp    
    # REMOVED: calculate_eta() method completely deleted
    
    def interpolate_ground_truth_trajectory(self, estimated_trajectory):
        """
        Interpolate ground truth trajectory to match the length of estimated trajectory.
        This provides better trajectory comparison when trajectories have different lengths.
        
        Args:
            estimated_trajectory: List of (x, y, theta) tuples
            
        Returns:
            Interpolated ground truth trajectory
        """
        if not self.ground_truth_trajectory or not estimated_trajectory:
            return []
        
        if len(self.ground_truth_trajectory) <= 1:
            return self.ground_truth_trajectory
        
        # Create parameter arrays for interpolation
        gt_length = len(self.ground_truth_trajectory)
        est_length = len(estimated_trajectory)
        
        # Parameter arrays (0 to 1)
        gt_params = np.linspace(0, 1, gt_length)
        est_params = np.linspace(0, 1, est_length)
        
        # Extract coordinates
        gt_x = [pos[0] for pos in self.ground_truth_trajectory]
        gt_y = [pos[1] for pos in self.ground_truth_trajectory]
        gt_theta = [pos[2] for pos in self.ground_truth_trajectory]
        
        # Interpolate
        interp_x = np.interp(est_params, gt_params, gt_x)
        interp_y = np.interp(est_params, gt_params, gt_y)
        interp_theta = np.interp(est_params, gt_params, gt_theta)
        
        # Return interpolated trajectory
        return [(x, y, theta) for x, y, theta in zip(interp_x, interp_y, interp_theta)]
    
    def record_timing(self, operation_name, duration):
        """Record timing for a specific operation."""
        if operation_name in self.timing_history:
            self.timing_history[operation_name].append(duration)
    
    def increment_update_count(self):
        """Increment the update counter for real-time performance tracking."""
        self.update_count += 1
    
    def calculate_effective_particle_count(self, weights):
        """
        Calculate effective particle count (n_eff).
        
        Args:
            weights: Array of normalized particle weights
            
        Returns:
            n_eff: Effective particle count
        """
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            normalized_weights = weights / weight_sum
        else:
            normalized_weights = np.ones(len(weights)) / len(weights)
        
        n_eff = 1.0 / np.sum(normalized_weights ** 2)
        self.n_eff_history.append(n_eff)
        return n_eff
    
    def update_landmark_stability(self, landmarks_in_sight):
        """
        Update landmark stability tracking.
        
        Args:
            landmarks_in_sight: List of currently detected landmarks
        """
        current_time = time.time()
        detected_ids = set()
        
        for landmark in landmarks_in_sight:
            _, _, landmark_id = landmark[:3]  # Handle both 3 and 4 element tuples
            landmark_id = str(landmark_id)
            detected_ids.add(landmark_id)
            
            if landmark_id in self.consecutive_detections:
                self.consecutive_detections[landmark_id] += 1
            else:
                self.consecutive_detections[landmark_id] = 1
                self.landmark_first_seen[landmark_id] = current_time
            
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
    
    @staticmethod
    def kabsch_alignment(estimated_points, ground_truth_points):
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
        
        P = np.array(estimated_points).T  # 2xN (estimated)
        Q = np.array(ground_truth_points).T  # 2xN (ground truth)
        
        centroid_P = np.mean(P, axis=1, keepdims=True)
        centroid_Q = np.mean(Q, axis=1, keepdims=True)
        
        P_centered = P - centroid_P
        Q_centered = Q - centroid_Q
        
        H = P_centered @ Q_centered.T
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        t = centroid_Q.flatten() - R @ centroid_P.flatten()
        aligned_points = (R @ P + t.reshape(-1, 1)).T
        
        return R, t, aligned_points
    
    def calculate_sse_metrics(self, particle_landmarks):
        """
        Calculate SSE and related metrics after Kabsch alignment.
        
        Args:
            particle_landmarks: Dictionary of landmarks from the best particle
            
        Returns:
            Dictionary containing SSE metrics
        """
        if not particle_landmarks or not self.ground_truth_markers:
            return {
                'sse': 0.0,
                'rmse': 0.0,
                'mean_error': 0.0,
                'per_landmark_errors': {},
                'detection_rate': 0.0,
                'num_detected': 0,
                'num_ground_truth': len(self.ground_truth_markers)
            }
        
        common_landmarks = []
        estimated_positions = []
        ground_truth_positions = []
        
        for marker_id, landmark in particle_landmarks.items():
            if marker_id in self.ground_truth_markers:
                common_landmarks.append(marker_id)
                estimated_positions.append([landmark.x, landmark.y])
                ground_truth_positions.append(self.ground_truth_markers[marker_id])
        
        if len(common_landmarks) < 2:
            return {
                'sse': 0.0,
                'rmse': 0.0,
                'mean_error': 0.0,
                'per_landmark_errors': {},
                'detection_rate': len(common_landmarks) / len(self.ground_truth_markers) * 100,
                'num_detected': len(common_landmarks),
                'num_ground_truth': len(self.ground_truth_markers)
            }
        
        R, t, aligned_positions = self.kabsch_alignment(estimated_positions, ground_truth_positions)
        self.alignment_transformation = (R, t)
        
        sse = 0.0
        per_landmark_errors = {}
        aligned_landmarks_dict = {}
        
        for i, marker_id in enumerate(common_landmarks):
            aligned_x, aligned_y = aligned_positions[i]
            aligned_landmarks_dict[marker_id] = [aligned_x, aligned_y]
            
            gt_x, gt_y = ground_truth_positions[i]
            error_squared = (aligned_x - gt_x)**2 + (aligned_y - gt_y)**2
            sse += error_squared
            per_landmark_errors[marker_id] = math.sqrt(error_squared)
        
        self.aligned_landmarks = aligned_landmarks_dict
        self.per_landmark_errors = per_landmark_errors
        
        num_landmarks = len(common_landmarks)
        rmse = math.sqrt(sse / num_landmarks) if num_landmarks > 0 else 0.0
        mean_error = sum(per_landmark_errors.values()) / num_landmarks if num_landmarks > 0 else 0.0
        detection_rate = len(common_landmarks) / len(self.ground_truth_markers) * 100
        
        # Store metrics history
        self.sse_values.append(sse)
        self.rmse_values.append(rmse)
        self.detection_rate_history.append(detection_rate)
        
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
    
    def calculate_ate(self, slam_trajectory):
        """
        Fast ATE calculation using KDTree for nearest neighbor search.
        """
        if not slam_trajectory or not self.ground_truth_trajectory:
            return None

        est_positions = np.array([(x, y) for x, y, _ in slam_trajectory])
        gt_positions = np.array([(x, y) for x, y, _ in self.ground_truth_trajectory])

        # Build KDTree for ground truth
        tree = KDTree(gt_positions)

        # Query nearest neighbor for all estimated points
        dists, _ = tree.query(est_positions, k=1)

        ate = np.sqrt(np.mean(dists ** 2))

        self.ate_values.append(ate)
        return ate
    
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
        
        current_n_eff = self.n_eff_history[-1]
        mean_n_eff = np.mean(self.n_eff_history)
        min_n_eff = np.min(self.n_eff_history)
        max_n_eff = np.max(self.n_eff_history)
        
        below_threshold = sum(1 for n in self.n_eff_history if n < self.n_eff_threshold)
        resampling_frequency = below_threshold / len(self.n_eff_history) * 100
        
        return {
            'current_n_eff': current_n_eff,
            'mean_n_eff': mean_n_eff,
            'min_n_eff': min_n_eff,
            'max_n_eff': max_n_eff,
            'n_eff_ratio': current_n_eff / self.num_particles,
            'resampling_frequency': resampling_frequency
        }
    
    def get_current_metrics(self):
        """Return comprehensive performance metrics (ETA REMOVED)."""
        timing_stats = self.get_timing_statistics()
        diversity_stats = self.get_particle_diversity_stats()
        
        metrics = {
            # Trajectory error metrics (ATE now correctly measures SLAM vs GT)
            'current_mpd': self.ate_values[-1] if self.ate_values else 0.0,
            'average_mpd': sum(self.ate_values) / len(self.ate_values) if self.ate_values else 0.0,
            
            # MSP metrics (keep these)
            'current_msp': self.msp_values[-1] if self.msp_values else 0.0,
            'average_msp': sum(self.msp_values) / len(self.msp_values) if self.msp_values else 0.0,
            
            # REMOVED: All ETA metrics deleted
            
            # Mapping accuracy metrics
            'current_sse': self.sse_values[-1] if self.sse_values else 0.0,
            'current_rmse': self.rmse_values[-1] if self.rmse_values else 0.0,
            'average_sse': sum(self.sse_values) / len(self.sse_values) if self.sse_values else 0.0,
            'average_rmse': sum(self.rmse_values) / len(self.rmse_values) if self.rmse_values else 0.0,
            'per_landmark_errors': self.per_landmark_errors.copy(),
            
            # Detection metrics
            'detection_rate': self.detection_rate_history[-1] if self.detection_rate_history else 0.0,
            'landmarks_detected': len(self.per_landmark_errors),
            'total_ground_truth': len(self.ground_truth_markers),
            
            # General metrics
            'trajectory_length': len(self.robot_trajectory),
            'ground_truth_trajectory_length': len(self.ground_truth_trajectory),
            'particles_count': self.num_particles,
            
            # Coordinate transformation info
            'rotation_applied_degrees': self.rotation_angle_degrees,
            
            # Advanced metrics
            'effective_particle_count': diversity_stats,
            'timing_performance': timing_stats,
            'landmark_stability': {
                'total_landmarks': len(self.landmark_stability),
                'stable_landmarks': sum(1 for s in self.landmark_stability.values() 
                                      if s['consecutive_detections'] >= 5),
                'stability_details': self.landmark_stability.copy()
            }
        }
        return metrics
    
    def save_metrics_to_file(self, filename="slam_comprehensive_metrics.txt"):
        """Save comprehensive performance metrics to a file (ETA REMOVED)."""
        try:
            metrics = self.get_current_metrics()
            with open(filename, 'w') as f:
                f.write("FastSLAM Comprehensive Performance Analysis\n")
                f.write("=" * 60 + "\n\n")
                
                # Coordinate Transform Information
                if self.rotation_angle_degrees != 0.0:
                    f.write("COORDINATE TRANSFORMATION\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"Clockwise Rotation Applied: {self.rotation_angle_degrees}°\n\n")
                
                # Trajectory Error Analysis (FIXED ATE)
                f.write("TRAJECTORY ERROR ANALYSIS\n")
                f.write("-" * 30 + "\n")
                f.write(f"Current ATE (SLAM vs GT): {metrics['current_mpd']:.6f} meters\n")
                f.write(f"Average ATE (SLAM vs GT): {metrics['average_mpd']:.6f} meters\n")
                f.write(f"Current MSP: {metrics['current_msp']:.6f} m²\n")
                f.write(f"Average MSP: {metrics['average_msp']:.6f} m²\n\n")
                # REMOVED: All ETA reporting deleted
                
                # Ground Truth Trajectory Information
                f.write("GROUND TRUTH TRAJECTORY INFORMATION\n")
                f.write("-" * 35 + "\n")
                f.write(f"Ground Truth Points: {metrics['ground_truth_trajectory_length']}\n")
                f.write(f"Estimated Points: {metrics['trajectory_length']}\n")
                if self.ground_truth_trajectory:
                    f.write("\nFirst few ground truth points (after rotation if applied):\n")
                    for i, (x, y, theta) in enumerate(self.ground_truth_trajectory[:5]):
                        f.write(f"  Point {i}: x={x:.3f}, y={y:.3f}, θ={theta:.3f} ({math.degrees(theta):.1f}°)\n")
                else:
                    f.write("No ground truth trajectory loaded!\n")
                f.write("\n")
                
                # Mapping Accuracy
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
                
                # Particle Filter Performance
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
                
                # Computational Performance
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
                        f.write(f"  {operation.replace('_', ' ').title()}: ")
                        f.write(f"{stats['mean']:.2f} ± {stats['std']:.2f} ms\n")
                f.write("\n")
                
                # Historical Data Summary (REMOVED ETA HISTORY)
                if self.msp_values:
                    f.write("MSP History (last 10 values):\n")
                    recent_msp = self.msp_values[-10:]
                    for i, msp in enumerate(recent_msp):
                        idx = len(self.msp_values) - len(recent_msp) + i + 1
                        f.write(f"  {idx}: {msp:.6f} m²\n")
                f.write("\n")
                
                # REMOVED: ETA history section completely deleted
                        
            rospy.loginfo(f"Enhanced metrics saved successfully to {filename}")
        except Exception as e:
            rospy.logerr(f"Error saving metrics: {e}")


class MetricsTimer:
    """Context manager for timing operations."""
    
    def __init__(self, metrics_tracker, operation_name):
        self.metrics_tracker = metrics_tracker
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.metrics_tracker.record_timing(self.operation_name, duration)