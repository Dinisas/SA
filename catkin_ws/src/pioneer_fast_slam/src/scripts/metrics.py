#!/usr/bin/env python3
import numpy as np
import math
import time
import yaml
import os
import rospy
from collections import defaultdict


class SLAMMetricsTracker:
    """
    Comprehensive metrics tracking for FastSLAM implementation.
    Handles all performance evaluation metrics including ATE, SSE, computational timing,
    particle diversity, and landmark stability.
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
        self.load_ground_truth_markers()
        
        # ATE and trajectory metrics
        self.ate_values = []
        self.robot_trajectory = []
        self.ground_truth_trajectory = []
        
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
    
    def load_ground_truth_markers(self):
        """Load ground truth marker positions from YAML file."""
        try:     
            with open(self.groundtruth_file, 'r') as file:
                data = yaml.safe_load(file)
                
            if 'markers' in data:
                for marker_id, coords in data['markers'].items():
                    self.ground_truth_markers[str(marker_id)] = coords
                rospy.loginfo(f"Loaded {len(self.ground_truth_markers)} ground truth markers")
            else:
                rospy.logwarn("No 'markers' section found in ground truth YAML")
                
        except Exception as e:
            rospy.logerr(f"Error loading ground truth markers: {e}")
    
    def update_robot_trajectory(self, x, y, theta):
        """Add a new pose to the robot trajectory."""
        self.robot_trajectory.append((x, y, theta))
    
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
    
    def calculate_ate(self, best_particle_trajectory):
        """
        Calculate Absolute Trajectory Error.
        
        Args:
            best_particle_trajectory: Trajectory from the best particle
            
        Returns:
            ATE value
        """
        if len(self.robot_trajectory) < 2 or not best_particle_trajectory:
            return 0.0
        
        total_error = 0.0
        count = 0
        min_len = min(len(self.robot_trajectory), len(best_particle_trajectory))
        
        for i in range(min_len):
            true_x, true_y, _ = self.robot_trajectory[i]
            est_x, est_y, _ = best_particle_trajectory[i]
            
            error = math.sqrt((true_x - est_x)**2 + (true_y - est_y)**2)
            total_error += error**2
            count += 1
        
        if count > 0:
            ate = math.sqrt(total_error / count)
            self.ate_values.append(ate)
            return ate
        return 0.0
    
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
        """Return comprehensive performance metrics."""
        timing_stats = self.get_timing_statistics()
        diversity_stats = self.get_particle_diversity_stats()
        
        metrics = {
            # Trajectory error metrics
            'current_mpd': self.ate_values[-1] if self.ate_values else 0.0,
            'average_mpd': sum(self.ate_values) / len(self.ate_values) if self.ate_values else 0.0,
            
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
            'particles_count': self.num_particles,
            
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
                
                # Landmark Stability Analysis
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
                
                # Per-Landmark Errors
                f.write("Per-Landmark Errors (After Alignment):\n")
                f.write("-" * 40 + "\n")
                for marker_id, error in metrics['per_landmark_errors'].items():
                    f.write(f"Marker {marker_id}: {error:.6f} meters\n")
                f.write("\n")
                
                # Historical Data Summary
                if self.sse_values:
                    f.write("Historical Performance Summary:\n")
                    f.write("SSE History (last 10 values):\n")
                    recent_sses = self.sse_values[-10:]
                    for i, sse in enumerate(recent_sses):
                        idx = len(self.sse_values) - len(recent_sses) + i + 1
                        f.write(f"  {idx}: {sse:.6f} m²\n")
                f.write("\n")
                
                if self.n_eff_history:
                    f.write("n_eff History (last 10 values):\n")
                    recent_neff = self.n_eff_history[-10:]
                    for i, neff in enumerate(recent_neff):
                        idx = len(self.n_eff_history) - len(recent_neff) + i + 1
                        f.write(f"  {idx}: {neff:.2f}\n")
                        
            rospy.loginfo(f"Comprehensive metrics saved to {filename}")
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