import pygame
import math
import numpy as np
import copy
import tf.transformations
import rospy
# ROS message types for landmark visualization
from visualization_msgs.msg import Marker, MarkerArray
from utils import resample, normalize_angle
# custom Particle class
from particle import Particle

class FastSlam:
    def __init__(self, tuning_option, window_size_pixel, size_m, pioneer_L, num_particles=50, screen=None, resample_method="low variance"):
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

        # Screen and map dimensions
        self.screen = screen      
        self.width_meters = size_m 
        self.height_meters = size_m
        # Pioneer P3-DX radius (approximately 44cm diameter)
        self.pioneer_radius = 0.22
        # robot wheel base
        self.pioneer_L = pioneer_L
        # convert radius to pixels
        self.pioneer_radius_pixel = self.pioneer_radius * self.SCREEN_WIDTH / self.width_meters

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
        
        self.update_screen()  # Update screen with initial state
        return
    
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
        particles = []
        for _ in range(self.num_particles):
            x = 0
            y = 0
            theta = 0
            pose = np.array([x, y, theta])
            particles.append(Particle(pose, self.num_particles, self.pioneer_L, self.tuning_options))
        return particles
    
    # Update the particles based on odometry data
    def update_odometry(self, odometry):
        # extract quarternion
        quaternion = [odometry[2][0], odometry[2][1], odometry[2][2], odometry[2][3]]
        # convert to euler angles
        (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(quaternion)
        # normalize yaw to -pi to pi
        yaw = normalize_angle(yaw)
    
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
        self.update_screen()

    # Perform SLAM computation based on observed landmarks
    def compute_slam(self, landmarks_in_sight):
        # Initialize weights list
        weights_here = []
        # for each observed landmark
        for landmark in landmarks_in_sight:
            # extract landmark data from tuple
            landmark_dist, landmark_bearing_angle, landmark_id = landmark
            # get first particle pose (unused)
            x, y, theta = self.particles[0].pose
            # reset weight list
            weights_here = []
            for particle in self.particles:
                # update particle with landmark
                particle.handle_landmark(landmark_dist, math.radians(landmark_bearing_angle), landmark_id)
                # collect particle weight
                weights_here.append(particle.weight)
        
        # Resample particles based on their weights
        self.particles, self.best_particle_ID = resample(self.particles, self.num_particles, self.resample_method, self.best_particle_ID)
        self.update_screen(landmarks_in_sight)

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
        pygame.draw.circle(self.screen, self.GREEN, pioneer_pos, self.pioneer_radius_pixel)
        pygame.draw.polygon(self.screen, self.BLUE, triangle_points)

        # Draw the particles
        for particle in self.particles:
            # get particle pose
            particle_x, particle_y, _ = particle.pose
            pygame.draw.circle(self.screen, self.RED, (int((particle_x) * self.SCREEN_WIDTH / self.width_meters + self.left_coordinate + self.SCREEN_WIDTH / 2), 
                                                       int((particle_y) * self.SCREEN_HEIGHT / self.height_meters + self.SCREEN_HEIGHT / 2)), 3)

        # Draw the landmarks
        for landmark_id, landmark in self.particles[self.best_particle_ID].landmarks.items():
            # get landmark position
            landmark_x, landmark_y = landmark.x, landmark.y
            # draw black circle
            pygame.draw.circle(self.screen, self.BLACK, (int(landmark_x * self.SCREEN_WIDTH / self.width_meters + self.left_coordinate + self.SCREEN_WIDTH / 2), 
                                                         int(landmark_y * self.SCREEN_HEIGHT / self.height_meters + self.SCREEN_HEIGHT / 2)), 5)
            # create font
            font = pygame.font.Font(None, 30)  
            # Render text surface
            text_surface = font.render("id:" + str(landmark_id), True, self.BLACK)
            # position text above circle
            text_rect = text_surface.get_rect(center=(int(landmark_x * self.SCREEN_WIDTH / self.width_meters + self.left_coordinate + self.SCREEN_WIDTH / 2), 
                                                      int(landmark_y * self.SCREEN_HEIGHT / self.height_meters + self.SCREEN_HEIGHT / 2) - 15))  # Position text surface above the circle
            self.screen.blit(text_surface, text_rect)  # Blit text surface onto the screen
            
        pygame.display.flip()

    # Get the best trajectory from the best particle
    def get_best_trajectory(self):
        return self.particles[self.best_particle_ID].trajectory