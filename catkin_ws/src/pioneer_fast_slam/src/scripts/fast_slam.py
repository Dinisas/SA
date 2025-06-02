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
#custom Metrics class
from metrics import SLAMMetricsTracker, MetricsTimer

class FastSlam:
    def __init__(self, tuning_option, window_size_pixel, size_m, pioneer_L,num_particles,groundtruth_file,screen=None, resample_method="low variance"):
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
        # index of best particle
        self.best_particle_ID = -1
        # create initial particles
        self.particles = self.initialize_particles()
        # Ros publisher for landmarks
        self.landmark_pub = rospy.Publisher('/landmarks', MarkerArray, queue_size=10)
        
        # Initialize the metrics tracker
        self.metrics = SLAMMetricsTracker(num_particles=num_particles, groundtruth_file = groundtruth_file,expected_update_rate=10.0)

        # Font for displaying metrics
        pygame.font.init()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        self.update_screen()  # Update screen with initial state
        return
    
    # Publish landmark positions to a ROS topic
    def publish_landmarks(self):
        # create ROS marker array message
        marker_array = MarkerArray()

        # SAFE COPY - prevents "dictionary changed size during iteration"
        if self.best_particle_ID >= 0 and self.best_particle_ID < len(self.particles):
            landmarks_copy = dict(self.particles[self.best_particle_ID].landmarks)

        for landmark_id, landmark in landmarks_copy.items():
            # extract landmark coordinates
            landmark_x, landmark_y = landmark.x, landmark.y
            
            # Extract numeric ID from virtual IDs or use the ID directly
            if isinstance(landmark_id, str) and landmark_id.startswith("virtual_"):
                # Extract number from "virtual_X" format
                numeric_id = int(landmark_id.split("_")[1])
            else:
                # For regular IDs, try to convert to int
                try:
                    numeric_id = int(landmark_id)
                except ValueError:
                    # If conversion fails, use hash of the ID
                    numeric_id = abs(hash(landmark_id)) % 1000000

            # Create a marker for the landmark
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "landmarks"
            marker.id = numeric_id  # Use the numeric ID
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
            text_marker.id = numeric_id + 1000  # Ensure unique ID for text
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
            text_marker.text = f"id: {landmark_id}"  # Display the original ID

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
        """Update the particles based on odometry data."""
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

            # Update each particle with motion model
            for particle in self.particles:
                particle.motion_model([delta_dist, delta_rot1, delta_rot2])
            
            # Update old odometry and yaw
            self.old_odometry = copy.deepcopy(odometry)
            self.old_yaw = copy.deepcopy(yaw)
        
        self.update_screen()

    def compute_slam(self, landmarks_in_sight):
        """
        Compute SLAM with support for both correspondence and non-correspondence problems.
        
        Automatically detects mode based on landmark IDs:
        - If landmark_id == -1: Use data association (non-correspondence)
        - If landmark_id != -1: Use unique IDs (correspondence)
        """
        if not landmarks_in_sight:
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
                self.particles, self.best_particle_ID = resample(
                    self.particles, self.num_particles, self.resample_method, self.best_particle_ID
                )
            
            # Update metrics after resampling
            best_particle = self.get_best_particle()
            
            # Calculate ATE if trajectory exists
            if hasattr(best_particle, 'trajectory') and best_particle.trajectory:
                self.metrics.calculate_ate(best_particle.trajectory)
            
            # Calculate SSE metrics
            self.metrics.calculate_sse_metrics(best_particle.landmarks)
            
            # Increment update count
            self.metrics.increment_update_count()
        
        # Update visualization
        self.update_screen(landmarks_in_sight)

    def draw_trajectory(self):
        """Draw robot trajectory from metrics tracker."""
        if len(self.metrics.robot_trajectory) > 1:
            points = []
            for x, y, _ in self.metrics.robot_trajectory:
                pixel_x = int(x * self.SCREEN_WIDTH / self.width_meters + 
                            self.left_coordinate + self.SCREEN_WIDTH / 2)
                pixel_y = int(y * self.SCREEN_HEIGHT / self.height_meters + 
                            self.SCREEN_HEIGHT / 2)
                points.append((pixel_x, pixel_y))
            
            if len(points) > 1:
                pygame.draw.lines(self.screen, self.CYAN, False, points, 2)

    def draw_markers(self, marker_type, marker_data=None):
        """
        Unified function to draw different types of markers.
        
        Args:
            marker_type: 'ground_truth', 'estimated', or 'aligned'
            marker_data: Optional custom data to use
        """
        # Configure based on marker type
        configs = {
            'ground_truth': {
                'color': self.ORANGE,
                'shape': 'square',
                'prefix': 'GT',
                'data_source': self.metrics.ground_truth_markers,
                'text_offset': (0, -20),
                'draw_error': False
            },
            'estimated': {
                'color': self.PURPLE,
                'shape': 'circle',
                'prefix': 'EST',
                'data_source': None,  # Will extract from best particle
                'text_offset': (0, 20),
                'draw_error': True,
                'error_color': self.RED,
                'error_width': 1
            },
            'aligned': {
                'color': self.LIME,
                'shape': 'diamond',
                'prefix': 'ALG',
                'data_source': self.metrics.aligned_landmarks,
                'text_offset': (15, 0),
                'draw_error': True,
                'error_color': self.LIME,
                'error_width': 2
            }
        }
        
        if marker_type not in configs:
            raise ValueError(f"Unknown marker type: {marker_type}")
        
        config = configs[marker_type]
        
        # Get data
        if marker_data is not None:
            data = marker_data
        elif marker_type == 'estimated':
            # Extract from best particle
            best_particle = self.get_best_particle()
            data = {lid: (lm.x, lm.y) for lid, lm in best_particle.landmarks.items()}
        else:
            data = config['data_source']
        
        if not data:
            return
        
        # Draw each marker
        for marker_id, position in data.items():
            marker_x, marker_y = position
            
            # Convert to pixel coordinates
            pixel_x = int(marker_x * self.SCREEN_WIDTH / self.width_meters + 
                        self.left_coordinate + self.SCREEN_WIDTH / 2)
            pixel_y = int(marker_y * self.SCREEN_HEIGHT / self.height_meters + 
                        self.SCREEN_HEIGHT / 2)
            
            # Draw shape
            if config['shape'] == 'square':
                pygame.draw.rect(self.screen, config['color'], 
                            (pixel_x - 6, pixel_y - 6, 12, 12))
            elif config['shape'] == 'circle':
                pygame.draw.circle(self.screen, config['color'], (pixel_x, pixel_y), 5)
            elif config['shape'] == 'diamond':
                diamond_points = [
                    (pixel_x, pixel_y - 6),
                    (pixel_x + 6, pixel_y),
                    (pixel_x, pixel_y + 6),
                    (pixel_x - 6, pixel_y)
                ]
                pygame.draw.polygon(self.screen, config['color'], diamond_points)
            
            # Draw text
            text_surface = self.small_font.render(f"{config['prefix']}{marker_id}", 
                                                True, config['color'])
            text_rect = text_surface.get_rect(
                center=(pixel_x + config['text_offset'][0], 
                    pixel_y + config['text_offset'][1])
            )
            self.screen.blit(text_surface, text_rect)
            
            # Draw error line if needed
            if config['draw_error'] and marker_id in self.metrics.ground_truth_markers:
                gt_x, gt_y = self.metrics.ground_truth_markers[marker_id]
                gt_pixel_x = int(gt_x * self.SCREEN_WIDTH / self.width_meters + 
                            self.left_coordinate + self.SCREEN_WIDTH / 2)
                gt_pixel_y = int(gt_y * self.SCREEN_HEIGHT / self.height_meters + 
                            self.SCREEN_HEIGHT / 2)
                
                pygame.draw.line(self.screen, config['error_color'], 
                            (pixel_x, pixel_y), (gt_pixel_x, gt_pixel_y), 
                            config['error_width'])

    def draw_metrics_panel(self):
        """Draw performance metrics panel using metrics tracker."""
        # Get current metrics
        current_metrics = self.metrics.get_current_metrics()
        diversity_stats = current_metrics['effective_particle_count']
        timing_stats = current_metrics['timing_performance']
        
        panel_x = 10
        panel_y = 10
        panel_width = 500
        panel_height = 450
        
        # Draw background
        pygame.draw.rect(self.screen, (0, 0, 0, 180), 
                        (panel_x, panel_y, panel_width, panel_height))
        pygame.draw.rect(self.screen, self.WHITE, 
                        (panel_x, panel_y, panel_width, panel_height), 2)
        
        y_offset = panel_y + 10
        
        # Title
        title = self.font.render("SLAM Performance Metrics", True, self.WHITE)
        self.screen.blit(title, (panel_x + 10, y_offset))
        y_offset += 30
        
        # Display metrics
        metrics_to_display = [
            (f"MPD: {current_metrics['current_mpd']:.3f}m", self.WHITE),
            (f"SSE: {current_metrics['current_sse']:.3f}m²", self.WHITE),
            (f"RMSE: {current_metrics['current_rmse']:.3f}m", self.WHITE),
            (f"Detection Rate: {current_metrics['detection_rate']:.1f}%", self.WHITE),
        ]
        
        for text, color in metrics_to_display:
            text_surface = self.font.render(text, True, color)
            self.screen.blit(text_surface, (panel_x + 10, y_offset))
            y_offset += 20
        
        y_offset += 5
        
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
                    f"Rate: {rt.get('actual_rate_hz', 0):.1f}Hz"
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
        y_offset += 20
        
        # Trajectory points
        traj_text = f"Trajectory points: {current_metrics['trajectory_length']}"
        text_surface = self.font.render(traj_text, True, self.WHITE)
        self.screen.blit(text_surface, (panel_x + 20, y_offset))

    def _process_unique_id_landmarks(self, unique_id_landmarks):
        """
        Process landmarks with unique IDs (correspondence problem).
        Each particle independently handles the landmark.
        """
        for landmark in unique_id_landmarks:
            landmark_dist, landmark_bearing_angle, landmark_id, marker_pixel_size = landmark
            landmark_id_str = str(landmark_id)
            
            # Each particle independently decides to create or update
            for particle in self.particles:
                particle.handle_landmark_with_id(
                    landmark_dist, 
                    math.radians(landmark_bearing_angle), 
                    landmark_id_str, 
                    marker_pixel_size
                )

    def _process_data_association_landmarks(self, data_assoc_landmarks):
        """
        Process landmarks requiring data association (non-correspondence problem).
        Uses shared decision making across all particles to maintain consistency.
        """
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
                self.landmark_creation_cooldown = 10  # Cooldown for 10 updates
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
        """
        Decide whether to create a new landmark based on multiple criteria.
        
        Returns:
            bool: True if should create new landmark, False otherwise
        """
        # No existing landmarks - create first one
        if best_match_id is None:
            return True
        
        # Dynamic threshold based on number of landmarks
        base_threshold = 4.0
        if num_landmarks < 3:
            threshold = base_threshold * 0.8  # More permissive early on
        elif num_landmarks < 5:
            threshold = base_threshold
        else:
            threshold = base_threshold * 1.2  # More conservative with many landmarks
        
        # Check if we're in creation cooldown
        if self.landmark_creation_cooldown > 0 and min_distance < threshold * 1.5:
            return False  # Don't create during cooldown unless very far
        
        # Make decision based on distance
        return min_distance > threshold

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
            self.draw_trajectory()
           
            #Draw ground truth, estimated and aligned markers
            self.draw_markers('ground_truth')
            self.draw_markers('estimated')
            self.draw_markers('aligned')    
            
            # Draw robot
            pygame.draw.circle(self.screen, self.GREEN, pioneer_pos, self.pioneer_radius_pixel)
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
            self.draw_legend()
            
            pygame.display.flip()
            
        except Exception as e:
            rospy.logerr(f"Error updating screen: {e}")

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
        """Return comprehensive performance metrics from the metrics tracker."""
        return self.metrics.get_current_metrics()
    
    def save_metrics_to_file(self, filename="slam_comprehensive_metrics.txt"):
        """Save comprehensive performance metrics to a file."""
        self.metrics.save_metrics_to_file(filename)