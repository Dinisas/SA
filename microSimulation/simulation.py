import pygame
import sys
import numpy as np
import math
import random
import copy

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 1000, 700
DARK_GRAY = (50, 50, 50)
LIGHT_GRAY = (150, 150, 150)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 255, 255)
YELLOW = (255, 255, 0)
PURPLE = (180, 0, 255)
ORANGE = (255, 165, 0)
PINK = (255, 105, 180)

# Simulation parameters
ARUCO_SIZE = 20         # Size of ArUco marker (pixels)
ROBOT_RADIUS = 15
PARTICLE_RADIUS = 2
MOTION_NOISE = 0.1      # Noise in movement
MEASUREMENT_NOISE = 10.0  # Noise in sensor readings (distance units)
ORIENTATION_NOISE = 0.05  # Noise in orientation readings (radians)
RESAMPLE_THRESHOLD = 0.5  # Threshold for resampling
SENSOR_RANGE = 200      # Maximum distance robot can sense ArUco markers
CAMERA_FOV = math.pi/2  # Field of view of the camera (90 degrees)
TOP_PARTICLES_COUNT = 4  # Number of top particles to display data for

# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("FastSLAM Simulation with ArUco Markers")
clock = pygame.time.Clock()

class ArUcoMarker:
    """
    Represents an ArUco marker with a unique ID, position, and orientation.
    """
    def __init__(self, marker_id, x, y, orientation=0.0):
        self.id = marker_id  # Unique ID for this marker
        self.x = x
        self.y = y
        self.orientation = orientation  # Orientation in radians
        self.size = ARUCO_SIZE
        
    def get_corners(self):
        """Return the four corners of the ArUco marker for drawing purposes."""
        half_size = self.size / 2
        corners = [
            (-half_size, -half_size),
            (half_size, -half_size),
            (half_size, half_size),
            (-half_size, half_size)
        ]
        
        # Rotate corners by the marker's orientation
        rotated_corners = []
        for corner_x, corner_y in corners:
            # Apply rotation matrix
            rx = corner_x * math.cos(self.orientation) - corner_y * math.sin(self.orientation)
            ry = corner_x * math.sin(self.orientation) + corner_y * math.cos(self.orientation)
            # Translate to marker position
            rotated_corners.append((self.x + rx, self.y + ry))
            
        return rotated_corners

class Robot:
    """
    Represents the true robot in the simulation.
    In a real-world scenario, this would be the physical robot.
    """
    def __init__(self, x, y, theta=0.0):
        self.x = x
        self.y = y
        self.theta = theta  # Orientation in radians
        self.speed = 3.0    # Linear movement speed
        self.turn_rate = 0.1  # Angular movement speed
    
    def move(self, control):
        """
        Update robot position and orientation based on control input.
        
        Args:
            control: Tuple (linear_velocity, angular_velocity)
                    Linear velocity in pixels/frame
                    Angular velocity in radians/frame
        """
        # Apply control using basic kinematic model
        self.x += control[0] * math.cos(self.theta)
        self.y += control[0] * math.sin(self.theta)
        self.theta += control[1]
        self.theta = normalize_angle(self.theta)  # Keep angle in [-π, π]
    
    def sense(self, aruco_markers):
        """
        Detect ArUco markers within sensor range and camera field of view.
        Simulates robot's camera and ArUco detection algorithms.
        
        Args:
            aruco_markers: List of ArUcoMarker objects
            
        Returns:
            List of tuples (marker_id, distance, bearing, marker_orientation)
              - marker_id: Unique identifier for the detected marker
              - distance: Distance to the marker (with noise)
              - bearing: Angle to the marker relative to robot heading (with noise)
              - marker_orientation: Detected marker orientation relative to robot (with noise)
        """
        measurements = []
        for marker in aruco_markers:
            # Calculate true distance and bearing to marker
            dx = marker.x - self.x
            dy = marker.y - self.y
            distance = math.sqrt(dx*dx + dy*dy)
            
            # Only measure markers within sensor range
            if distance <= SENSOR_RANGE:
                # Calculate bearing relative to robot's heading
                bearing = math.atan2(dy, dx) - self.theta
                bearing = normalize_angle(bearing)  # Normalize to [-π, π]
                
                # Check if marker is within camera field of view
                if abs(bearing) <= CAMERA_FOV/2:
                    # Calculate marker's orientation relative to robot
                    relative_orientation = marker.orientation - self.theta
                    relative_orientation = normalize_angle(relative_orientation)
                    
                    # Add Gaussian noise to measurements
                    noisy_distance = distance + np.random.normal(0, MEASUREMENT_NOISE)
                    noisy_bearing = bearing + np.random.normal(0, 0.1)
                    noisy_bearing = normalize_angle(noisy_bearing)
                    
                    # Add noise to orientation detection
                    noisy_orientation = relative_orientation + np.random.normal(0, ORIENTATION_NOISE)
                    noisy_orientation = normalize_angle(noisy_orientation)
                    
                    measurements.append((marker.id, noisy_distance, noisy_bearing, noisy_orientation))
        
        return measurements

class Particle:
    """
    Represents a single particle in the FastSLAM filter.
    Each particle maintains its own robot pose estimate and map of ArUco markers.
    """
    def __init__(self, x, y, theta=0.0):
        self.x = x  # x-coordinate
        self.y = y  # y-coordinate
        self.theta = theta  # orientation
        self.weight = 1.0  # Particle weight (importance factor)
        
        # Dictionary to store ArUco marker estimates: {marker_id: (mean, covariance, orientation)}
        # Each marker is tracked by an EKF:
        # - mean: (x, y) position estimate
        # - covariance: 2x2 matrix representing uncertainty in position
        # - orientation: estimated orientation of the marker
        self.markers = {}
    
    def move(self, control):
        """Sample a new robot pose based on control input and motion model."""
        # Add Gaussian noise to controls
        noise_x = control[0] + np.random.normal(0, MOTION_NOISE * abs(control[0]))
        noise_theta = control[1] + np.random.normal(0, MOTION_NOISE * abs(control[1]))
        
        # Apply noisy control to update particle pose
        self.x += noise_x * math.cos(self.theta)
        self.y += noise_x * math.sin(self.theta)
        self.theta += noise_theta
        self.theta = normalize_angle(self.theta)
    
    def update_marker(self, marker_id, true_marker, measured_distance, measured_bearing, measured_orientation):
        """
        Update a single ArUco marker estimate using Extended Kalman Filter (EKF).
        Also updates the marker's orientation estimate.
        
        Args:
            marker_id: ID of the marker being updated
            true_marker: True ArUcoMarker object (in simulation only)
            measured_distance: Measured distance to marker
            measured_bearing: Measured bearing to marker
            measured_orientation: Measured orientation of the marker
        """
        # If marker not seen before, initialize it
        if marker_id not in self.markers:
            # Calculate initial position estimate using particle's pose and measurement
            lm_x = self.x + measured_distance * math.cos(self.theta + measured_bearing)
            lm_y = self.y + measured_distance * math.sin(self.theta + measured_bearing)
            
            # Initialize orientation (global frame)
            global_orientation = self.theta + measured_orientation
            global_orientation = normalize_angle(global_orientation)
            
            # Initial covariance matrix with high uncertainty
            self.markers[marker_id] = ((lm_x, lm_y), np.eye(2) * 1000.0, global_orientation)
        else:
            # ---- EKF UPDATE STEP ----
            # Get current estimate
            (mu_x, mu_y), sigma, old_orientation = self.markers[marker_id]
            
            # Calculate expected measurement based on current estimate
            dx = mu_x - self.x
            dy = mu_y - self.y
            expected_distance = math.sqrt(dx*dx + dy*dy)
            expected_bearing = normalize_angle(math.atan2(dy, dx) - self.theta)
            
            # Calculate orientation in robot frame
            expected_orientation = normalize_angle(old_orientation - self.theta)
            
            # Measurement innovation (difference between measured and expected)
            z_diff = np.array([
                measured_distance - expected_distance,
                normalize_angle(measured_bearing - expected_bearing)
            ])
            
            # Jacobian of measurement model
            q = dx*dx + dy*dy
            H = np.array([
                [dx/math.sqrt(q), dy/math.sqrt(q)],
                [-dy/q, dx/q]
            ])
            
            # Measurement noise covariance matrix
            Q = np.array([
                [MEASUREMENT_NOISE*MEASUREMENT_NOISE, 0],
                [0, 0.01]
            ])
            
            # Kalman gain
            K = sigma @ H.T @ np.linalg.inv(H @ sigma @ H.T + Q)
            
            # Update position estimate
            new_mu = np.array([mu_x, mu_y]) + K @ z_diff
            
            # Update covariance
            new_sigma = (np.eye(2) - K @ H) @ sigma
            
            # Update orientation using a simple weighted average
            # We weigh the old orientation more to make it more stable
            orientation_weight = 0.8
            orientation_diff = normalize_angle(measured_orientation - expected_orientation)
            
            # Calculate new global orientation
            new_orientation = old_orientation + (1 - orientation_weight) * orientation_diff
            new_orientation = normalize_angle(new_orientation)
            
            # Store updated marker estimate
            self.markers[marker_id] = ((new_mu[0], new_mu[1]), new_sigma, new_orientation)
    
    def measurement_probability(self, marker_id, true_marker, measured_distance, measured_bearing, measured_orientation):
        """
        Calculate how likely the particle is to have produced the given measurement.
        
        Args:
            marker_id: ID of the measured marker
            true_marker: True ArUcoMarker object
            measured_distance: Measured distance to marker
            measured_bearing: Measured bearing to marker
            measured_orientation: Measured orientation of the marker
            
        Returns:
            Probability value
        """
        # If we haven't seen this marker before, return a small probability
        if marker_id not in self.markers:
            return 0.1
        
        # Get current marker estimate
        (mu_x, mu_y), sigma, marker_orientation = self.markers[marker_id]
        
        # Calculate expected measurements
        dx = mu_x - self.x
        dy = mu_y - self.y
        expected_distance = math.sqrt(dx*dx + dy*dy)
        expected_bearing = normalize_angle(math.atan2(dy, dx) - self.theta)
        expected_orientation = normalize_angle(marker_orientation - self.theta)
        
        # Calculate differences
        distance_diff = measured_distance - expected_distance
        bearing_diff = normalize_angle(measured_bearing - expected_bearing)
        orientation_diff = normalize_angle(measured_orientation - expected_orientation)
        
        # Calculate probability using Gaussian models
        distance_prob = math.exp(-(distance_diff**2) / (2 * MEASUREMENT_NOISE**2))
        bearing_prob = math.exp(-(bearing_diff**2) / (2 * 0.1**2))
        orientation_prob = math.exp(-(orientation_diff**2) / (2 * ORIENTATION_NOISE**2))
        
        # Return the product of probabilities
        return distance_prob * bearing_prob * orientation_prob

class FastSLAM:
    """
    Main FastSLAM algorithm implementation for ArUco markers.
    """
    def __init__(self, num_particles, init_x, init_y):
        """Initialize FastSLAM filter with particles."""
        self.num_particles = num_particles
        self.particles = [Particle(init_x, init_y) for _ in range(num_particles)]
        self.aruco_markers = []  # List of ArUcoMarker objects
    
    def update(self, control, measurements):
        """
        Main FastSLAM update function.
        
        Args:
            control: Motion command (linear_vel, angular_vel)
            measurements: List of (marker_id, distance, bearing, orientation) tuples
        """
        # Move each particle according to control
        for p in self.particles:
            p.move(control)
        
        # Update weights based on measurements
        if measurements:
            # For each particle
            for p in self.particles:
                p.weight = 1.0  # Reset weight
                
                # Process each measurement
                for marker_id, distance, bearing, orientation in measurements:
                    # Find the corresponding true marker (for simulation only)
                    true_marker = next((m for m in self.aruco_markers if m.id == marker_id), None)
                    if true_marker:
                        # Update this marker in the particle's map
                        p.update_marker(marker_id, true_marker, distance, bearing, orientation)
                        
                        # Update particle weight
                        probability = p.measurement_probability(
                            marker_id, true_marker, distance, bearing, orientation)
                        p.weight *= probability
            
            # Normalize weights
            total_weight = sum(p.weight for p in self.particles)
            if total_weight > 0:
                for p in self.particles:
                    p.weight /= total_weight
            
            # Calculate effective number of particles
            n_eff = 1.0 / sum(p.weight**2 for p in self.particles)
            
            # Resample if effective number is too low
            if n_eff < self.num_particles * RESAMPLE_THRESHOLD:
                self.resample()
    
    def resample(self):
        """Resample particles based on their weights."""
        new_particles = []
        
        # Compute cumulative weights
        weights = [p.weight for p in self.particles]
        cumulative_weights = np.cumsum(weights)
        cumulative_weights /= cumulative_weights[-1]  # Normalize to [0,1]
        
        # Resample using low variance sampling
        r = random.random() / self.num_particles
        i = 0
        
        for m in range(self.num_particles):
            u = r + m / self.num_particles
            
            while u > cumulative_weights[i]:
                i += 1
            
            # Deep copy the selected particle
            new_particles.append(copy.deepcopy(self.particles[i]))
            new_particles[-1].weight = 1.0 / self.num_particles
        
        # Replace old particles with resampled set
        self.particles = new_particles
    
    def get_best_particle(self):
        """Return the particle with the highest weight."""
        if not self.particles:
            return None
        return max(self.particles, key=lambda p: p.weight)
    
    def get_top_particles(self, n=TOP_PARTICLES_COUNT):
        """Return the top n particles by weight."""
        sorted_particles = sorted(self.particles, key=lambda p: p.weight, reverse=True)
        return sorted_particles[:min(n, len(sorted_particles))]

def normalize_angle(angle):
    """Normalize angle to range [-π, π]."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle

# UI and Visualization Functions
def draw_robot(robot):
    """Draw the robot on the screen."""
    pygame.draw.circle(screen, RED, (int(robot.x), int(robot.y)), ROBOT_RADIUS)
    end_x = robot.x + ROBOT_RADIUS * math.cos(robot.theta)
    end_y = robot.y + ROBOT_RADIUS * math.sin(robot.theta)
    pygame.draw.line(screen, WHITE, (int(robot.x), int(robot.y)), (int(end_x), int(end_y)), 2)
    
    # Draw the camera field of view
    fov_left = robot.theta - CAMERA_FOV/2
    fov_right = robot.theta + CAMERA_FOV/2
    
    line_length = SENSOR_RANGE  # Length of FOV visualization lines
    left_x = robot.x + line_length * math.cos(fov_left)
    left_y = robot.y + line_length * math.sin(fov_left)
    right_x = robot.x + line_length * math.cos(fov_right)
    right_y = robot.y + line_length * math.sin(fov_right)
    
    # Draw FOV lines
    pygame.draw.line(screen, (100, 100, 100), (int(robot.x), int(robot.y)), 
                    (int(left_x), int(left_y)), 1)
    pygame.draw.line(screen, (100, 100, 100), (int(robot.x), int(robot.y)), 
                    (int(right_x), int(right_y)), 1)
    
    # Draw FOV arc
    arc_rect = pygame.Rect(
        int(robot.x - SENSOR_RANGE), 
        int(robot.y - SENSOR_RANGE),
        SENSOR_RANGE * 2, 
        SENSOR_RANGE * 2
    )
    start_angle = math.degrees(fov_left)
    end_angle = math.degrees(fov_right)
    
    # Adjust angles for pygame's coordinate system
    if start_angle < 0:
        start_angle += 360
    if end_angle < 0:
        end_angle += 360
        
    # Draw the arc (semi-transparent)
    arc_surface = pygame.Surface((SENSOR_RANGE * 2, SENSOR_RANGE * 2), pygame.SRCALPHA)
    pygame.draw.arc(arc_surface, (100, 100, 100, 50), 
                   pygame.Rect(0, 0, SENSOR_RANGE * 2, SENSOR_RANGE * 2),
                   math.radians(start_angle), math.radians(end_angle), 2)
    screen.blit(arc_surface, (int(robot.x - SENSOR_RANGE), int(robot.y - SENSOR_RANGE)))

def draw_aruco_markers(aruco_markers):
    """Draw the ArUco markers on the screen."""
    for marker in aruco_markers:
        # Draw the ArUco marker square
        corners = marker.get_corners()
        
        # Draw marker outline
        pygame.draw.polygon(screen, BLUE, corners)
        
        # Draw marker ID
        font = pygame.font.Font(None, 24)
        text = font.render(str(marker.id), True, BLACK)
        # Calculate center position
        center_x = sum(x for x, y in corners) / 4
        center_y = sum(y for x, y in corners) / 4
        text_rect = text.get_rect(center=(center_x, center_y))
        screen.blit(text, text_rect)
        
        # Draw orientation indicator (direction the marker is facing)
        front_x = marker.x + (marker.size/2) * math.cos(marker.orientation)
        front_y = marker.y + (marker.size/2) * math.sin(marker.orientation)
        pygame.draw.line(screen, GREEN, (marker.x, marker.y), (front_x, front_y), 2)

def draw_particles(particles):
    """Draw all particles."""
    for p in particles:
        pygame.draw.circle(screen, LIGHT_GRAY, (int(p.x), int(p.y)), PARTICLE_RADIUS)

def draw_top_particles(particles, colors):
    """Draw top particles with distinct colors."""
    for i, p in enumerate(particles):
        if i < len(colors):
            pygame.draw.circle(screen, colors[i], (int(p.x), int(p.y)), PARTICLE_RADIUS+2)

def draw_top_particles_markers(particles, colors):
    """Draw markers from top particles with matching colors."""
    for i, p in enumerate(particles):
        if i < len(colors):
            color = colors[i]
            for marker_id, ((x, y), cov, orientation) in p.markers.items():
                # Draw estimated marker position
                pygame.draw.circle(screen, color, (int(x), int(y)), 3)
                
                # Optionally, draw estimated orientation (if desired)
                end_x = x + 10 * math.cos(orientation)
                end_y = y + 10 * math.sin(orientation)
                pygame.draw.line(screen, color, (int(x), int(y)), (int(end_x), int(end_y)), 1)

def draw_sensor_range(robot):
    """Draw circle showing robot's sensing range."""
    pygame.draw.circle(screen, LIGHT_GRAY, (int(robot.x), int(robot.y)), SENSOR_RANGE, 1)

def draw_buttons(buttons):
    """Draw UI buttons."""
    for button in buttons:
        pygame.draw.rect(screen, button['color'], button['rect'])
        font = pygame.font.Font(None, 24)
        text = font.render(button['text'], True, BLACK)
        text_rect = text.get_rect(center=button['rect'].center)
        screen.blit(text, text_rect)

def draw_route(route_points):
    """Draw predefined route."""
    for i in range(1, len(route_points)):
        pygame.draw.line(screen, GREEN, route_points[i-1], route_points[i], 2)
    
    # Draw waypoints
    for point in route_points:
        pygame.draw.circle(screen, GREEN, point, 5)

def draw_data_panel(robot, top_particles, colors, measurements, slam):
    """Draw data panel showing statistics."""
    panel_rect = pygame.Rect(WIDTH - 280, 170, 270, HEIGHT - 180)
    pygame.draw.rect(screen, (30, 30, 30), panel_rect)
    pygame.draw.rect(screen, WHITE, panel_rect, 1)
    
    font = pygame.font.Font(None, 22)
    y_pos = panel_rect.top + 10
    
    # Draw panel title
    title = font.render("Top Particles Data", True, WHITE)
    screen.blit(title, (panel_rect.left + 10, y_pos))
    y_pos += 25
    
    # Draw robot data
    robot_title = font.render("Robot Position:", True, RED)
    screen.blit(robot_title, (panel_rect.left + 10, y_pos))
    y_pos += 20
    
    robot_data = font.render(f"x: {robot.x:.2f}, y: {robot.y:.2f}", True, WHITE)
    screen.blit(robot_data, (panel_rect.left + 20, y_pos))
    y_pos += 20
    
    robot_theta = font.render(f"θ: {robot.theta:.2f} rad ({math.degrees(robot.theta):.1f}°)", True, WHITE)
    screen.blit(robot_theta, (panel_rect.left + 20, y_pos))
    y_pos += 30
    
    # Draw data for each top particle
    for i, p in enumerate(top_particles):
        if i < len(colors):
            particle_title = font.render(f"Particle {i+1}:", True, colors[i])
            screen.blit(particle_title, (panel_rect.left + 10, y_pos))
            y_pos += 20
            
            pos_text = font.render(f"x: {p.x:.2f}, y: {p.y:.2f}", True, WHITE)
            screen.blit(pos_text, (panel_rect.left + 20, y_pos))
            y_pos += 20
            
            theta_text = font.render(f"θ: {p.theta:.2f} rad ({math.degrees(p.theta):.1f}°)", True, WHITE)
            screen.blit(theta_text, (panel_rect.left + 20, y_pos))
            y_pos += 20
            
            weight_text = font.render(f"weight: {p.weight:.6f}", True, WHITE)
            screen.blit(weight_text, (panel_rect.left + 20, y_pos))
            y_pos += 20
            
            markers_text = font.render(f"markers: {len(p.markers)}", True, WHITE)
            screen.blit(markers_text, (panel_rect.left + 20, y_pos))
            y_pos += 30
    
    # Draw marker detection data if space permits
    if y_pos < panel_rect.bottom - 30 and measurements:
        measurements_title = font.render("Latest Detections:", True, YELLOW)
        screen.blit(measurements_title, (panel_rect.left + 10, y_pos))
        y_pos += 20
        
        for i, (marker_id, distance, bearing, orientation) in enumerate(measurements):
            if y_pos < panel_rect.bottom - 20:
                meas_text = font.render(f"ArUco{marker_id}: {distance:.1f}m, {math.degrees(bearing):.1f}°", True, WHITE)
                screen.blit(meas_text, (panel_rect.left + 20, y_pos))
                y_pos += 20

def save_synthetic_data(robot_positions, measurements_history):
    """Save synthetic data generated during the simulation."""
    import csv
    
    # Save robot trajectory
    with open('robot_trajectory.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['x', 'y', 'theta'])
        for pos in robot_positions:
            writer.writerow(pos)
    
    # Save ArUco measurements
    with open('aruco_measurements.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['step', 'aruco_id', 'distance', 'bearing', 'orientation'])
        for step, measurements in enumerate(measurements_history):
            for aruco_id, distance, bearing, orientation in measurements:
                writer.writerow([step, aruco_id, distance, bearing, orientation])
    
    print("Synthetic data saved to robot_trajectory.csv and aruco_measurements.csv")

def main():
    """Main function: setup and game loop"""
    # Initialize robot at center of screen
    robot = Robot(WIDTH/2 - 140, HEIGHT/2)
    
    # Initialize FastSLAM
    num_particles = 100
    slam = FastSLAM(num_particles, WIDTH/2 - 140, HEIGHT/2)
    
    # Colors for top particles
    top_particles_colors = [YELLOW, GREEN, PURPLE, ORANGE, PINK]
    
    # UI states
    place_aruco_mode = False
    auto_route_mode = False
    
    # Define a predefined route (circular pattern)
    center_x, center_y = WIDTH/2 - 140, HEIGHT/2
    route_radius = 150
    num_waypoints = 20
    
    # Create a circular route
    route_points = []
    for i in range(num_waypoints+1):  # +1 to close the loop
        angle = 2 * math.pi * i / num_waypoints
        x = center_x + route_radius * math.cos(angle)
        y = center_y + route_radius * math.sin(angle)
        route_points.append((x, y))
    
    # Data collection for synthetic data
    robot_positions = []  # List to store robot positions (x, y, theta)
    measurements_history = []  # List to store measurements at each position
    current_waypoint = 0
    
    # Flag for data generation completed
    data_generated = False
    
    # Buttons
    buttons = [
        {
            'text': 'Place ArUco',
            'rect': pygame.Rect(WIDTH - 150, 10, 140, 30),
            'color': YELLOW,
            'action': lambda: toggle_aruco_mode()
        },
        {
            'text': 'Start Auto Route',
            'rect': pygame.Rect(WIDTH - 150, 50, 140, 30),
            'color': YELLOW,
            'action': lambda: toggle_auto_route()
        },
        {
            'text': 'Clear ArUco',
            'rect': pygame.Rect(WIDTH - 150, 90, 140, 30),
            'color': YELLOW,
            'action': lambda: clear_aruco()
        },
        {
            'text': 'Save Data',
            'rect': pygame.Rect(WIDTH - 150, 130, 140, 30),
            'color': YELLOW,
            'action': lambda: save_data()
        }
    ]
    
    def toggle_aruco_mode():
        nonlocal place_aruco_mode, auto_route_mode
        if not auto_route_mode:
            place_aruco_mode = not place_aruco_mode
            buttons[0]['text'] = 'Drive Robot' if place_aruco_mode else 'Place ArUco'
    
    def toggle_auto_route():
        nonlocal auto_route_mode, place_aruco_mode, current_waypoint, data_generated
        if not place_aruco_mode:
            auto_route_mode = not auto_route_mode
            if auto_route_mode:
                # Reset route and data collection when starting
                current_waypoint = 0
                robot_positions.clear()
                measurements_history.clear()
                data_generated = False
                buttons[1]['text'] = 'Stop Auto Route'
            else:
                buttons[1]['text'] = 'Start Auto Route'
    
    def clear_aruco():
        slam.aruco_markers = []
        # Reset each particle's markers
        for p in slam.particles:
            p.markers = {}
    
    def save_data():
        if robot_positions and measurements_history:
            save_synthetic_data(robot_positions, measurements_history)
    
    # Next ArUco ID to assign
    next_aruco_id = 0
    
    # Main game loop
    running = True
    measurements = []  # Initialize measurements
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Check button clicks
                pos = pygame.mouse.get_pos()
                for button in buttons:
                    if button['rect'].collidepoint(pos):
                        button['action']()
                        break
                else:  # If no button was clicked
                    # Place ArUco marker if in marker placement mode
                    if place_aruco_mode and pos[0] < WIDTH - 280:
                        # Generate random orientation for the new marker
                        orientation = random.uniform(0, 2*math.pi)
                        new_marker = ArUcoMarker(next_aruco_id, pos[0], pos[1], orientation)
                        slam.aruco_markers.append(new_marker)
                        next_aruco_id += 1
        
        # Handle control and movement
        control = [0, 0]  # [forward/backward, rotation]
        
        if auto_route_mode and not data_generated:
            # Follow predefined route
            if current_waypoint < len(route_points):
                # Get current waypoint
                target_x, target_y = route_points[current_waypoint]
                
                # Calculate direction and distance to waypoint
                dx = target_x - robot.x
                dy = target_y - robot.y
                distance = math.sqrt(dx*dx + dy*dy)
                target_angle = math.atan2(dy, dx)
                
                # Calculate angle difference (bearing to waypoint)
                angle_diff = normalize_angle(target_angle - robot.theta)
                
                # If we're not facing the waypoint, turn towards it
                if abs(angle_diff) > 0.05:
                    control[1] = 0.05 * np.sign(angle_diff)  # Turn towards waypoint
                    control[0] = 0  # Don't move forward while turning
                else:
                    # Move towards waypoint
                    control[0] = min(robot.speed, distance)  # Forward speed
                
                # Check if we've reached the waypoint
                if distance < 5:
                    current_waypoint += 1
                    
                    # If we've completed the route, stop auto mode
                    if current_waypoint >= len(route_points):
                        data_generated = True
                        auto_route_mode = False
                        buttons[1]['text'] = 'Start Auto Route'
                        print("Route completed, data generation finished.")
            
            # Move robot
            robot.move(control)
            
            # Get measurements
            measurements = robot.sense(slam.aruco_markers)
            
            # Record data for synthetic dataset
            robot_positions.append((robot.x, robot.y, robot.theta))
            measurements_history.append(measurements.copy())
            
            # Update FastSLAM
            slam.update(control, measurements)
            
        elif not place_aruco_mode and not auto_route_mode:
            # Manual control with arrow keys
            keys = pygame.key.get_pressed()
            
            if keys[pygame.K_UP]:
                control[0] = robot.speed
            if keys[pygame.K_DOWN]:
                control[0] = -robot.speed
            if keys[pygame.K_LEFT]:
                control[1] = robot.turn_rate
            if keys[pygame.K_RIGHT]:
                control[1] = -robot.turn_rate
            
            # Move robot
            robot.move(control)
            
            # Get measurements
            measurements = robot.sense(slam.aruco_markers)
            
            # Update FastSLAM
            slam.update(control, measurements)
        
        # Get top particles for visualization and data display
        top_particles = slam.get_top_particles(TOP_PARTICLES_COUNT)
        
        # Draw everything
        screen.fill(DARK_GRAY)
        draw_sensor_range(robot)
        draw_aruco_markers(slam.aruco_markers)
        draw_route(route_points)  # Draw the predefined route
        draw_robot(robot)
        draw_particles(slam.particles)
        draw_top_particles(top_particles, top_particles_colors)
        draw_top_particles_markers(top_particles, top_particles_colors)
        draw_buttons(buttons)
        draw_data_panel(robot, top_particles, top_particles_colors, measurements, slam)
        
        # Display information
        font = pygame.font.Font(None, 24)
        
        # Display mode
        if auto_route_mode:
            mode_text = "Mode: AUTO ROUTE"
        elif place_aruco_mode:
            mode_text = "Mode: PLACE ARUCO"
        else:
            mode_text = "Mode: DRIVE ROBOT"
            
        mode_surface = font.render(mode_text, True, WHITE)
        screen.blit(mode_surface, (10, 10))
        
        # Display controls
        controls_text = "Controls: Arrow Keys to move"
        screen.blit(font.render(controls_text, True, WHITE), (10, 40))
        
        # Display visible markers
        visible_text = f"Visible ArUco: {len(measurements)}"
        screen.blit(font.render(visible_text, True, WHITE), (10, 70))
        
        # Display total markers
        total_markers_text = f"Total ArUco: {len(slam.aruco_markers)}"
        screen.blit(font.render(total_markers_text, True, WHITE), (10, 100))
        
        # Display route progress if in auto mode
        if auto_route_mode:
            progress_text = f"Route Progress: {current_waypoint}/{len(route_points)}"
            screen.blit(font.render(progress_text, True, GREEN), (10, 130))
        
        # Update display
        pygame.display.flip()
        
        # Cap frame rate
        clock.tick(60)
    
    # Clean up
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()