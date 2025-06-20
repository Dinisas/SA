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
from visualization_msgs.msg import Marker, MarkerArray
from utils import resample, normalize_angle
from particle import Particle
from metrics import SLAMMetricsTracker, MetricsTimer

class FastSlam:
    def __init__(self, tuning_option, window_size_pixel, size_m, pioneer_L, num_particles, groundtruth_file, screen=None, resample_method="low variance"):
        print(f"DEBUG: FastSlam.__init__ called with num_particles = {num_particles}")
        # Initialize various parameters and settings
        self.tuning_options = tuning_option
        self.SCREEN_WIDTH = window_size_pixel
        self.SCREEN_HEIGHT = window_size_pixel
        self.metrics_update_interval = 30  # You can adjust this

        # Motion detection thresholds
        self.min_motion_threshold = {
            'linear': 0.001,   # 1mm minimum linear motion
            'angular': 0.001   # ~0.057 degrees minimum angular motion
        }
        self.is_robot_moving = False
        self.last_motion_time = time.time()
        
        # Update synchronization
        self.last_odometry_update_time = time.time()
        self.last_measurement_update_time = time.time()
        self.min_time_between_updates = 0.033  # ~30Hz max update rate
# Decouple rendering from SLAM loop:
        self.render_interval = 0.1  # Render every 100ms (10 FPS)
        self.next_render_time = time.time() + self.render_interval
 
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
        
        # choose resampling method
        self.resample_method = resample_method
        
        # Define colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 160, 0)
        self.BLUE = (10, 10, 255)
        self.RED = (170, 0, 0)
        self.ORANGE = (255, 165, 0)
        self.PURPLE = (128, 0, 128)
        self.CYAN = (0, 255, 255)
        self.LIME = (50, 205, 50)
        self.YELLOW = (255, 255, 0)
        self.MAGENTA = (255, 0, 255)  # Color for ground truth trajectory

        # Screen and map dimensions
        self.screen = screen      
        self.width_meters = size_m 
        self.height_meters = size_m
        self.pioneer_radius = 0.22
        self.pioneer_L = pioneer_L
        self.pioneer_radius_pixel = int(self.pioneer_radius * self.SCREEN_WIDTH / self.width_meters)

        # Initialize SLAM-related variables
        self.old_odometry = [0.0, 0.0]
        self.old_yaw = 0
        self.num_particles = num_particles
        self.best_particle_ID = -1
        self.particles = self.initialize_particles()
        self.landmark_pub = rospy.Publisher('/landmarks', MarkerArray, queue_size=10)
        
        # Initialize the enhanced metrics tracker
        self.metrics = SLAMMetricsTracker(num_particles=num_particles, groundtruth_file=groundtruth_file, expected_update_rate=30.0)

        # Font for displaying metrics
        pygame.font.init()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # Motion accumulator
        self.accumulated_motion = {
            'distance': 0.0,
            'rotation1': 0.0,
            'rotation2': 0.0
        }
        
        # SLAM trajectory tracking for ATE calculation
        self.best_particle_trajectory = []  # Store best particle poses over time
        self.max_trajectory_length = 300   # Limit memory usage
        
        # Decoupled rendering also on odometry
        current_time = time.time()
        if current_time >= self.next_render_time:
            self.update_screen()
            self.next_render_time = current_time + self.render_interval

        return
    
    def publish_landmarks(self):
        marker_array = MarkerArray()

        if self.best_particle_ID >= 0 and self.best_particle_ID < len(self.particles):
            landmarks_copy = dict(self.particles[self.best_particle_ID].landmarks)

        for landmark_id, landmark in landmarks_copy.items():
            landmark_x, landmark_y = landmark.x, landmark.y
            
            if isinstance(landmark_id, str) and landmark_id.startswith("virtual_"):
                numeric_id = int(landmark_id.split("_")[1])
            else:
                try:
                    numeric_id = int(landmark_id)
                except ValueError:
                    numeric_id = abs(hash(landmark_id)) % 1000000

            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "landmarks"
            marker.id = numeric_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = -landmark_x
            marker.pose.position.y = landmark_y
            marker.pose.position.z = 0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 255.0
            marker.color.b = 0.0

            marker_array.markers.append(marker)

            text_marker = Marker()
            text_marker.header.frame_id = "map"
            text_marker.header.stamp = rospy.Time.now()
            text_marker.ns = "landmark_text"
            text_marker.id = numeric_id + 1000
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = -landmark_x
            text_marker.pose.position.y = landmark_y
            text_marker.pose.position.z = 0.5
            text_marker.pose.orientation.x = 0.0
            text_marker.pose.orientation.y = 0.0
            text_marker.pose.orientation.z = 0.0
            text_marker.pose.orientation.w = 1.0
            text_marker.scale.z = 0.4
            text_marker.color.a = 1.0
            text_marker.color.r = 1.0
            text_marker.color.g = 255.0
            text_marker.color.b = 1.0
            text_marker.text = f"id: {landmark_id}"

            marker_array.markers.append(text_marker)

        self.landmark_pub.publish(marker_array)
    
    def get_best_particle(self):
        return self.particles[self.best_particle_ID]
    
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
    
    def check_motion_significance(self, odometry):
        """Check if robot has moved significantly since last update"""
        quaternion = [odometry[2][0], odometry[2][1], odometry[2][2], odometry[2][3]]
        (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(quaternion)
        yaw = normalize_angle(yaw)
        
        # Calculate motion deltas
        delta_x = odometry[0] - self.old_odometry[0]
        delta_y = odometry[1] - self.old_odometry[1]
        delta_dist = math.sqrt(delta_x**2 + delta_y**2)
        delta_yaw = normalize_angle(yaw - self.old_yaw)
        
        # Check if motion is significant
        if delta_dist > self.min_motion_threshold['linear'] or abs(delta_yaw) > self.min_motion_threshold['angular']:
            self.is_robot_moving = True
            self.last_motion_time = time.time()
            return True
        else:
            # Check if robot has been stationary for a while
            if time.time() - self.last_motion_time > 0.5:  # 500ms threshold
                self.is_robot_moving = False
            return False
    
    def track_best_particle_pose(self):
        """
        Track the best particle's pose over time for accurate ATE calculation.
        Call this after SLAM updates.
        """
        try:
            if (self.best_particle_ID >= 0 and 
                self.best_particle_ID < len(self.particles)):
                
                best_particle = self.particles[self.best_particle_ID]
                x, y, theta = best_particle.pose
                
                # Store the current best particle pose
                self.best_particle_trajectory.append((x, y, theta))
                
                # Limit trajectory size
                if len(self.best_particle_trajectory) > self.max_trajectory_length:
                    self.best_particle_trajectory.pop(0)

                    
                rospy.logdebug(f"Tracked best particle pose: ({x:.3f}, {y:.3f}, {theta:.3f})")
                
        except Exception as e:
            rospy.logdebug(f"Error tracking best particle pose: {e}")

    def get_slam_trajectory_for_ate(self):
        """
        Create a SLAM trajectory for ATE calculation.
        This returns the best particle's trajectory which represents SLAM estimates.
        
        Returns:
            List of (x, y, theta) tuples representing SLAM-estimated trajectory
        """
        try:
            if hasattr(self, 'best_particle_trajectory') and self.best_particle_trajectory:
                # Use explicitly tracked best particle trajectory
                slam_trajectory = self.best_particle_trajectory.copy()
                rospy.logdebug(f"Created SLAM trajectory with {len(slam_trajectory)} points for ATE calculation")
                return slam_trajectory
            else:
                # Fallback to robot trajectory method (less accurate but still usable)
                slam_trajectory = []
                for pose in self.metrics.robot_trajectory:
                    if isinstance(pose, (list, tuple)) and len(pose) >= 3:
                        slam_trajectory.append((pose[0], pose[1], pose[2]))
                
                rospy.logdebug(f"Created fallback SLAM trajectory with {len(slam_trajectory)} points for ATE calculation")
                return slam_trajectory
                
        except Exception as e:
            rospy.logerr(f"Error creating SLAM trajectory for ATE: {e}")
            return []
    
    def update_odometry(self, odometry):
        """Update the particles based on odometry data with motion detection."""
        current_time = time.time()
        
        # Rate limiting
        if current_time - self.last_odometry_update_time < self.min_time_between_updates:
            return
        
        # Check if robot has moved significantly
        if not self.check_motion_significance(odometry):
            # Robot hasn't moved significantly - skip particle update
            rospy.logdebug("Robot stationary - skipping particle motion update")
            return
        
        # Use the metrics timer context manager
        with MetricsTimer(self.metrics, 'motion_update'):
            # Extract quaternion
            quaternion = [odometry[2][0], odometry[2][1], odometry[2][2], odometry[2][3]]
            (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(quaternion)
            yaw = normalize_angle(yaw)
        
            # Update robot trajectory in metrics tracker
            self.metrics.update_robot_trajectory(odometry[0], odometry[1], yaw)
        
            # Calculate motion deltas
            delta_dist = math.sqrt((odometry[0] - self.old_odometry[0])**2 + 
                                (odometry[1] - self.old_odometry[1])**2)
            delta_rot1 = normalize_angle(math.atan2(odometry[1] - self.old_odometry[1], 
                                                odometry[0] - self.old_odometry[0]) - self.old_yaw)
            delta_rot2 = normalize_angle(yaw - self.old_yaw - delta_rot1)

            # Accumulate motion
            self.accumulated_motion['distance'] += delta_dist
            self.accumulated_motion['rotation1'] += delta_rot1
            self.accumulated_motion['rotation2'] += delta_rot2

            # Update each particle with accumulated motion
            for particle in self.particles:
                particle.motion_model([
                    self.accumulated_motion['distance'],
                    self.accumulated_motion['rotation1'],
                    self.accumulated_motion['rotation2']
                ])
            
            # Reset accumulated motion
            self.accumulated_motion = {
                'distance': 0.0,
                'rotation1': 0.0,
                'rotation2': 0.0
            }
            
            # Update old odometry and yaw
            self.old_odometry = copy.deepcopy(odometry)
            self.old_yaw = copy.deepcopy(yaw)
            self.last_odometry_update_time = current_time
        
        self.update_screen()

    def compute_slam(self, landmarks_in_sight):
        """
        Compute SLAM with FIXED ATE calculation (ETA removed).
        """
        if not landmarks_in_sight:
            return
        
        current_time = time.time()
        
        # Rate limiting for measurement updates
        if current_time - self.last_measurement_update_time < self.min_time_between_updates:
            return
        
        # Track total update time
        with MetricsTimer(self.metrics, 'total_update'):
            # Update landmark stability tracking
            self.metrics.update_landmark_stability(landmarks_in_sight)
            
            # Group landmarks by type
            data_assoc_landmarks = [l for l in landmarks_in_sight if l[2] == -1]
            unique_id_landmarks = [l for l in landmarks_in_sight if l[2] != -1]
            
            # Process landmarks with timing
            with MetricsTimer(self.metrics, 'landmark_update'):
                # Process unique ID landmarks (correspondence problem)
                if unique_id_landmarks:
                    self._process_unique_id_landmarks(unique_id_landmarks)
                
                # Process data association landmarks (non-correspondence problem)
                if data_assoc_landmarks:
                    self._process_data_association_landmarks(data_assoc_landmarks)
                
                # Collect weights after all landmark processing
                with MetricsTimer(self.metrics, 'weight_calculation'):
                    weights = np.array([p.weight for p in self.particles])
            
            # Calculate effective particle count
            n_eff = self.metrics.calculate_effective_particle_count(weights)
            
            # Resample particles if needed
            with MetricsTimer(self.metrics, 'resampling'):
                # More conservative resampling threshold when stationary
                resample_threshold = 0.5 if self.is_robot_moving else 0.3
                
                if n_eff < self.num_particles * resample_threshold:
                    self.particles, self.best_particle_ID = resample(
                        self.particles, self.num_particles, self.resample_method, self.best_particle_ID
                    )
                else:
                    # Update best particle without resampling
                    self.best_particle_ID = np.argmax(weights)
            
            # Track best particle pose always
            self.track_best_particle_pose()

            # Always compute landmark SSE (since it's fast)
            best_particle = self.get_best_particle()
            self.metrics.calculate_sse_metrics(best_particle.landmarks)

            # Only calculate heavy metrics every N updates
            if self.metrics.update_count % self.metrics_update_interval == 0:
                slam_trajectory = self.get_slam_trajectory_for_ate()

                if slam_trajectory and len(slam_trajectory) > 1:
                    self.metrics.calculate_ate(slam_trajectory)
                    self.metrics.calculate_msp()

            
            # Increment update count
            self.metrics.increment_update_count()
            self.last_measurement_update_time = current_time
        
        # Update visualization
# Decoupled rendering
            current_time = time.time()
            if current_time >= self.next_render_time:
                self.update_screen(landmarks_in_sight)
                self.next_render_time = current_time + self.render_interval

    def _process_unique_id_landmarks(self, unique_id_landmarks):
        """Process landmarks with unique IDs (correspondence problem solved)."""
        for landmark in unique_id_landmarks:
            landmark_dist, landmark_bearing_angle, landmark_id, marker_pixel_size = landmark
            
            # Process this landmark for all particles
            for particle in self.particles:
                particle.handle_landmark_with_id(
                    landmark_dist, 
                    math.radians(landmark_bearing_angle), 
                    str(landmark_id), 
                    marker_pixel_size
                )

    def _process_data_association_landmarks(self, data_assoc_landmarks):
        """Process landmarks requiring data association (non-correspondence problem)."""
        # Get decision-making particle (best or first)
        decision_particle_id = self.best_particle_ID if self.best_particle_ID >= 0 else 0
        decision_particle = self.particles[decision_particle_id]
        
        # Initialize landmark creation tracking if needed
        if not hasattr(self, 'landmark_creation_cooldown'):
            self.landmark_creation_cooldown = 0
        
        # Make association decisions for all observations
        association_decisions = []
        
        for i, landmark in enumerate(data_assoc_landmarks):
            landmark_dist, landmark_bearing_angle, _, marker_pixel_size = landmark
            
            # Find best association
            best_match_id, min_distance = decision_particle.find_best_landmark_association(
                landmark_dist, 
                math.radians(landmark_bearing_angle), 
                marker_pixel_size
            )
            
            # Decide whether to create new landmark
            should_create = self._should_create_new_landmark(
                best_match_id, min_distance, len(decision_particle.landmarks)
            )
            
            if should_create:
                new_id = f"virtual_{len(decision_particle.landmarks)}"
                association_decisions.append(('create', new_id))
                self.landmark_creation_cooldown = 10
            else:
                association_decisions.append(('update', best_match_id))
        
        # Apply decisions to all particles
        for i, (landmark, (action, target_id)) in enumerate(zip(data_assoc_landmarks, association_decisions)):
            landmark_dist, landmark_bearing_angle, _, marker_pixel_size = landmark
            
            for particle in self.particles:
                if action == 'create':
                    particle.create_landmark(
                        landmark_dist, 
                        math.radians(landmark_bearing_angle), 
                        target_id, 
                        marker_pixel_size
                    )
                else:  # update
                    particle.handle_landmark_with_id(
                        landmark_dist, 
                        math.radians(landmark_bearing_angle), 
                        target_id, 
                        marker_pixel_size
                    )
        
        # Decrease cooldown
        if self.landmark_creation_cooldown > 0:
            self.landmark_creation_cooldown -= 1

    def _should_create_new_landmark(self, best_match_id, min_distance, num_landmarks):
        """Decide whether to create a new landmark based on multiple criteria."""
        # No existing landmarks - create first one
        if best_match_id is None:
            return True
        
        # Dynamic threshold based on number of landmarks and robot motion
        base_threshold = 4.0
        
        # Adjust threshold based on robot motion status
        if not self.is_robot_moving:
            # More conservative when stationary
            threshold = base_threshold * 1.5
        else:
            # Normal thresholds when moving
            if num_landmarks < 3:
                threshold = base_threshold * 0.8
            elif num_landmarks < 5:
                threshold = base_threshold
            else:
                threshold = base_threshold * 1.2
        
        # Check if we're in creation cooldown
        if self.landmark_creation_cooldown > 0 and min_distance < threshold * 1.5:
            return False
        
        # Make decision based on distance
        return min_distance > threshold

    def draw_ground_truth_trajectory(self):
        """Draw ground truth trajectory from metrics tracker."""
        if len(self.metrics.ground_truth_trajectory) > 1:
            points = []
            for position in self.metrics.ground_truth_trajectory:
                # Handle both (x, y, theta) tuples and potentially other formats
                if isinstance(position, (list, tuple)) and len(position) >= 2:
                    x, y = position[0], position[1]
                    # theta is position[2] if available, but we don't need it for drawing the path
                    
                    pixel_x = int(x * self.SCREEN_WIDTH / self.width_meters + 
                                self.left_coordinate + self.SCREEN_WIDTH / 2)
                    pixel_y = int(y * self.SCREEN_HEIGHT / self.height_meters + 
                                self.SCREEN_HEIGHT / 2)
                    points.append((pixel_x, pixel_y))
            
            if len(points) > 1:
                # Draw ground truth trajectory with distinct color and style
                pygame.draw.lines(self.screen, self.MAGENTA, False, points, 3)
                
                # Draw waypoints on ground truth trajectory
                for i, point in enumerate(points[::2]):  # Every other point to avoid clutter
                    pygame.draw.circle(self.screen, self.MAGENTA, point, 4)

    def draw_trajectory(self):
        """Draw robot trajectory from metrics tracker."""
        if len(self.metrics.robot_trajectory) > 1:
            points = []
            for position in self.metrics.robot_trajectory:
                # Handle both (x, y, theta) tuples and potentially other formats
                if isinstance(position, (list, tuple)) and len(position) >= 2:
                    x, y = position[0], position[1]
                    
                    pixel_x = int(x * self.SCREEN_WIDTH / self.width_meters + 
                                self.left_coordinate + self.SCREEN_WIDTH / 2)
                    pixel_y = int(-y * self.SCREEN_HEIGHT / self.height_meters + 
                                self.SCREEN_HEIGHT / 2)
                    points.append((pixel_x, pixel_y))
            
            if len(points) > 1:
                pygame.draw.lines(self.screen, self.CYAN, False, points, 2)

    def draw_markers(self, marker_type):
        """Draw different types of markers (ground_truth, estimated, aligned)."""
        if marker_type == 'ground_truth':
            # Draw ground truth markers in orange
            for marker_id, pos in self.metrics.ground_truth_markers.items():
                pixel_x = int(pos[0] * self.SCREEN_WIDTH / self.width_meters + 
                            self.left_coordinate + self.SCREEN_WIDTH / 2)
                pixel_y = int(pos[1] * self.SCREEN_HEIGHT / self.height_meters + 
                            self.SCREEN_HEIGHT / 2)
                pygame.draw.circle(self.screen, self.ORANGE, (pixel_x, pixel_y), 8)
                # Draw marker ID
                text_surface = self.small_font.render(str(marker_id), True, self.BLACK)
                text_rect = text_surface.get_rect(center=(pixel_x, pixel_y))
                self.screen.blit(text_surface, text_rect)
        
        elif marker_type == 'estimated':
            # Draw estimated markers from best particle in purple
            if self.best_particle_ID >= 0 and self.best_particle_ID < len(self.particles):
                best_particle = self.particles[self.best_particle_ID]
                for marker_id, landmark in best_particle.landmarks.items():
                    pixel_x = int(landmark.x * self.SCREEN_WIDTH / self.width_meters + 
                                self.left_coordinate + self.SCREEN_WIDTH / 2)
                    pixel_y = int(landmark.y * self.SCREEN_HEIGHT / self.height_meters + 
                                self.SCREEN_HEIGHT / 2)
                    pygame.draw.circle(self.screen, self.PURPLE, (pixel_x, pixel_y), 6)
        
        elif marker_type == 'aligned':
            # Draw Kabsch-aligned markers in lime green
            if hasattr(self.metrics, 'aligned_landmarks') and self.metrics.aligned_landmarks:
                for marker_id, pos in self.metrics.aligned_landmarks.items():
                    pixel_x = int(pos[0] * self.SCREEN_WIDTH / self.width_meters + 
                                self.left_coordinate + self.SCREEN_WIDTH / 2)
                    pixel_y = int(pos[1] * self.SCREEN_HEIGHT / self.height_meters + 
                                self.SCREEN_HEIGHT / 2)
                    pygame.draw.circle(self.screen, self.LIME, (pixel_x, pixel_y), 4)

    def draw_enhanced_legend(self):
        """Draw an enhanced legend explaining the colors and metrics (ETA removed)."""
        legend_x = self.SCREEN_WIDTH - 280
        legend_y = 10
        
        # Background
        pygame.draw.rect(self.screen, (0, 0, 0, 128), (legend_x, legend_y, 260, 200))
        pygame.draw.rect(self.screen, self.WHITE, (legend_x, legend_y, 260, 200), 1)
        
        y_pos = legend_y + 10
        
        # Title
        title = self.font.render("Enhanced Legend:", True, self.WHITE)
        self.screen.blit(title, (legend_x + 10, y_pos))
        y_pos += 25
        
        # Legend items
        legend_items = [
            ("Ground Truth Markers", self.ORANGE),
            ("Estimated Markers", self.PURPLE),
            ("Kabsch Aligned", self.LIME),
            ("Ground Truth Trajectory", self.MAGENTA),
            ("Estimated Trajectory", self.CYAN),
            ("Robot (Moving)", self.GREEN),
            ("Robot (Stationary)", self.ORANGE),
            ("Error Lines", self.RED)
        ]
        
        for text, color in legend_items:
            # Draw color indicator
            pygame.draw.circle(self.screen, color, (legend_x + 15, y_pos + 8), 5)
            # Draw text
            text_surface = self.small_font.render(text, True, self.WHITE)
            self.screen.blit(text_surface, (legend_x + 30, y_pos))
            y_pos += 20
        
        # Add metrics explanation (ETA removed)
        y_pos += 10
        metrics_title = self.small_font.render("Key Metrics:", True, self.YELLOW)
        self.screen.blit(metrics_title, (legend_x + 10, y_pos))
        y_pos += 18
        
        metric_explanations = [
            "ATE: SLAM vs Ground Truth",
            "MSP: Mean Squared Position"
        ]
        
        for text in metric_explanations:
            text_surface = self.small_font.render(text, True, self.WHITE)
            self.screen.blit(text_surface, (legend_x + 15, y_pos))
            y_pos += 16

    def draw_metrics_panel(self):
        """Draw performance metrics panel (ETA REMOVED)."""
        # Get current metrics
        current_metrics = self.metrics.get_current_metrics()
        diversity_stats = current_metrics['effective_particle_count']
        timing_stats = current_metrics['timing_performance']
        
        panel_x = 10
        panel_y = 10
        panel_width = 520
        panel_height = 580
        
        # Draw background
        pygame.draw.rect(self.screen, (0, 0, 0, 180), 
                        (panel_x, panel_y, panel_width, panel_height))
        pygame.draw.rect(self.screen, self.WHITE, 
                        (panel_x, panel_y, panel_width, panel_height), 2)
        
        y_offset = panel_y + 10
        
        # Title
        title = self.font.render("Enhanced SLAM Performance Metrics", True, self.WHITE)
        self.screen.blit(title, (panel_x + 10, y_offset))
        y_offset += 30
        
        # Robot status
        status_color = self.GREEN if self.is_robot_moving else self.RED
        status_text = "MOVING" if self.is_robot_moving else "STATIONARY"
        robot_status = self.font.render(f"Robot Status: {status_text}", True, status_color)
        self.screen.blit(robot_status, (panel_x + 10, y_offset))
        y_offset += 25
        
        # Display trajectory metrics (FIXED ATE, REMOVED ETA)
        trajectory_metrics = [
            (f"ATE (SLAM vs GT): {current_metrics['current_mpd']:.3f}m", self.WHITE),
            (f"MSP: {current_metrics['current_msp']:.3f}m²", self.CYAN),
            (f"RMSE: {current_metrics['current_rmse']:.3f}m", self.WHITE),
            (f"Detection Rate: {current_metrics['detection_rate']:.1f}%", self.WHITE),
        ]
        
        for text, color in trajectory_metrics:
            text_surface = self.font.render(text, True, color)
            self.screen.blit(text_surface, (panel_x + 10, y_offset))
            y_offset += 20
        
        y_offset += 5
        
        # Trajectory Information Section (REMOVE ETA)
        traj_title = self.font.render("Trajectory Information:", True, self.YELLOW)
        self.screen.blit(traj_title, (panel_x + 10, y_offset))
        y_offset += 20
        
        traj_info = [
            f"Estimated Points: {current_metrics['trajectory_length']}",
            f"Ground Truth Points: {current_metrics['ground_truth_trajectory_length']}",
            f"Avg ATE (SLAM vs GT): {current_metrics['average_mpd']:.3f}m",
            f"Avg MSP: {current_metrics['average_msp']:.3f}m²"
        ]
        
        for text in traj_info:
            text_surface = self.font.render(text, True, self.WHITE)
            self.screen.blit(text_surface, (panel_x + 20, y_offset))
            y_offset += 18
        
        # Add theta calculation info
        theta_info = self.font.render("θ calc: Auto from trajectory", True, self.CYAN)
        self.screen.blit(theta_info, (panel_x + 20, y_offset))
        y_offset += 25
        
        # Particle Diversity Section
        diversity_title = self.font.render("Particle Diversity:", True, self.CYAN)
        self.screen.blit(diversity_title, (panel_x + 10, y_offset))
        y_offset += 20
        
        diversity_metrics = [
            f"n_eff: {diversity_stats['current_n_eff']:.1f}/{self.num_particles}",
            f"Diversity: {diversity_stats['n_eff_ratio']:.2%}",
            f"Resample Freq: {diversity_stats['resampling_frequency']:.1f}%"
        ]
        
        for text in diversity_metrics:
            text_surface = self.font.render(text, True, self.WHITE)
            self.screen.blit(text_surface, (panel_x + 20, y_offset))
            y_offset += 18
        
        y_offset += 7
        
        # Computational Performance Section
        if 'total_update' in timing_stats and timing_stats['total_update']['count'] > 0:
            perf_title = self.font.render("Computational Performance:", True, self.YELLOW)
            self.screen.blit(perf_title, (panel_x + 10, y_offset))
            y_offset += 20
            
            perf_metrics = [
                f"Total: {timing_stats['total_update']['mean']:.1f}ms",
                f"Motion: {timing_stats['motion_update']['mean']:.1f}ms",
                f"Landmarks: {timing_stats['landmark_update']['mean']:.1f}ms"
            ]
            
            for text in perf_metrics:
                text_surface = self.font.render(text, True, self.WHITE)
                self.screen.blit(text_surface, (panel_x + 20, y_offset))
                y_offset += 18
            
            # Real-time performance
            rt = timing_stats.get('real_time_performance', {})
            if rt:
                rt_metrics = [
                    f"RT Factor: {rt.get('real_time_factor', 0):.2f}x",
                    f"Rate: {rt.get('actual_rate_hz', 0):.1f}Hz / 30Hz target"
                ]
                
                for text in rt_metrics:
                    text_surface = self.font.render(text, True, self.WHITE)
                    self.screen.blit(text_surface, (panel_x + 20, y_offset))
                    y_offset += 18
        
        y_offset += 7
        
        # Landmark Stability
        stability_title = self.font.render("Landmark Stability:", True, self.GREEN)
        self.screen.blit(stability_title, (panel_x + 10, y_offset))
        y_offset += 20
        
        stability = current_metrics['landmark_stability']
        stability_text = (f"Stable Landmarks: "
                        f"{stability['stable_landmarks']}/{stability['total_landmarks']}")
        text_surface = self.font.render(stability_text, True, self.WHITE)
        self.screen.blit(text_surface, (panel_x + 20, y_offset))

    def update_screen(self, landmarks_in_sight=None):
        """Update the display screen with the current state."""
        try:
            # [Keep the initialization checks]
            if not pygame.get_init():
                pygame.init()
                pygame.display.init()
                
            if pygame.display.get_surface() is None:
                self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        
            if self.best_particle_ID == -1:
                self.best_particle_ID = np.random.randint(len(self.particles))

            # Get best particle pose
            x, y, theta = self.particles[self.best_particle_ID].pose
            pioneer_pos = (int((x) * self.SCREEN_WIDTH / self.width_meters + 
                              self.left_coordinate + self.SCREEN_WIDTH / 2), 
                          int((y) * self.SCREEN_HEIGHT / self.height_meters + 
                              self.SCREEN_HEIGHT / 2))
            
            # Calculate triangle points for robot orientation
            triangle_length = 0.8 * self.pioneer_radius_pixel
            triangle_tip_x = pioneer_pos[0] + triangle_length * math.cos(theta)
            triangle_tip_y = pioneer_pos[1] - triangle_length * math.sin(theta)
            triangle_left_x = pioneer_pos[0] + triangle_length * math.cos(theta + 5 * math.pi / 6) 
            triangle_left_y = pioneer_pos[1] - triangle_length * math.sin(theta + 5 * math.pi / 6)
            triangle_right_x = pioneer_pos[0] + triangle_length * math.cos(theta - 5 * math.pi / 6)
            triangle_right_y = pioneer_pos[1] - triangle_length * math.sin(theta - 5 * math.pi / 6)
            triangle_points = [(triangle_tip_x, triangle_tip_y), 
                              (triangle_left_x, triangle_left_y), 
                              (triangle_right_x, triangle_right_y)]
            
            # Clear screen
            half_screen_rect = pygame.Rect(self.left_coordinate, 0, 
                                         self.right_coordinate, self.SCREEN_HEIGHT)
            pygame.draw.rect(self.screen, self.WHITE, half_screen_rect)
            
            # Draw elements in order
            # Draw ground truth trajectory FIRST (behind everything else)
            self.draw_ground_truth_trajectory()
            
            # Draw estimated trajectory
            self.draw_trajectory()
           
            # Draw ground truth, estimated and aligned markers
            self.draw_markers('ground_truth')
            self.draw_markers('estimated')
            self.draw_markers('aligned')    
            
            # Draw robot
            robot_color = self.GREEN if self.is_robot_moving else self.ORANGE
            pygame.draw.circle(self.screen, robot_color, pioneer_pos, self.pioneer_radius_pixel)
            pygame.draw.polygon(self.screen, self.BLUE, triangle_points)

            # Draw particles
            for particle in self.particles:
                particle_x, particle_y, _ = particle.pose
                pygame.draw.circle(self.screen, self.RED, 
                                 (int((particle_x) * self.SCREEN_WIDTH / self.width_meters + 
                                      self.left_coordinate + self.SCREEN_WIDTH / 2), 
                                  int((particle_y) * self.SCREEN_HEIGHT / self.height_meters + 
                                      self.SCREEN_HEIGHT / 2)), 3)

            # Draw metrics and legend
            self.draw_metrics_panel()
            self.draw_enhanced_legend()
            
            pygame.display.flip()
            
        except Exception as e:
            rospy.logerr(f"Error updating screen: {e}")

    def get_best_trajectory(self):
        return self.particles[self.best_particle_ID].trajectory

    def get_current_metrics(self):
        """Return comprehensive performance metrics from the enhanced metrics tracker."""
        return self.metrics.get_current_metrics()
    
    def save_metrics_to_file(self, filename="slam_enhanced_metrics.txt"):
        """Save comprehensive performance metrics to a file."""
        self.metrics.save_metrics_to_file(filename)