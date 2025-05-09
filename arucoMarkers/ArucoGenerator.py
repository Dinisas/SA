import cv2
import numpy as np
from PIL import Image
import os

def open_images(image_paths):
    for image_path in image_paths:
        with Image.open(image_path) as img:
            img.show()  # This will open the image using your default image viewer

def generate_aruco_marker(dictionary_id=cv2.aruco.DICT_6X6_250, marker_id=0, size=600, border_bits=1, border_size=50):
    """
    Generate an ArUco marker image
    
    Args:
        dictionary_id: The ArUco dictionary to use
        marker_id: The ID of the marker to generate
        size: Size of the marker in pixels
        border_bits: Number of bits in marker border (default=1)
        border_size: Additional white border size in pixels (default=50)
        
    Returns:
        The generated marker image
    """
    # Get the dictionary
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    
    # Generate the marker
    marker_image = np.zeros((size, size), dtype=np.uint8)
    cv2.aruco.generateImageMarker(dictionary, marker_id, size, marker_image, border_bits)
    
    # Add extra white border for better printing
    if border_size > 0:
        total_size = size + 2 * border_size
        bordered_image = np.ones((total_size, total_size), dtype=np.uint8) * 255
        bordered_image[border_size:border_size+size, border_size:border_size+size] = marker_image
        marker_image = bordered_image
    
    # Convert to BGR for saving
    marker_image_color = cv2.cvtColor(marker_image, cv2.COLOR_GRAY2BGR)
    
    return marker_image_color

def resize_marker(marker_image, scale_factor):
    """Resize a marker image by a scale factor"""
    new_width = int(marker_image.shape[1] * scale_factor)
    new_height = int(marker_image.shape[0] * scale_factor)
    return cv2.resize(marker_image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

def save_aruco_marker(marker_image, filename):
    """Save an ArUco marker image to a file"""
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    cv2.imwrite(filename, marker_image)
    return filename

def generate_multiple_markers(dictionary_id=cv2.aruco.DICT_6X6_250, num_markers=10, size=600, 
                             border_bits=1, border_size=50, prefix="marker"):
    """Generate multiple ArUco markers"""
    image_paths = []
    for i in range(num_markers):
        marker = generate_aruco_marker(dictionary_id, i, size, border_bits, border_size)
        path = f"./arucoMarkersImages/{prefix}_{i}.png"
        save_aruco_marker(marker, path)
        image_paths.append(path)
        print(f"Generated marker with ID {i}")
    return image_paths

# Example usage
if __name__ == "__main__":
    # Available dictionaries (examples):
    # DICT_4X4_50, DICT_4X4_100, DICT_4X4_250, DICT_4X4_1000
    # DICT_5X5_50, DICT_5X5_100, DICT_5X5_250, DICT_5X5_1000
    # DICT_6X6_50, DICT_6X6_100, DICT_6X6_250, DICT_6X6_1000
    # DICT_7X7_50, DICT_7X7_100, DICT_7X7_250, DICT_7X7_1000
    
    # Create a list to store all generated image paths
    all_image_paths = []
    
    # Generate a single marker with large size (800px) and border (100px)
    # marker = generate_aruco_marker(cv2.aruco.DICT_6X6_250, 23, 800, 1, 100)
    # path = save_aruco_marker(marker, "./arucoMarkersImages/aruco_marker_23_large.png")
    # all_image_paths.append(path)
    
    # Generate multiple markers (600px with 50px border)
    paths = generate_multiple_markers(cv2.aruco.DICT_5X5_100, 5, 600, 1, 50, "marker_5x5_large")
    all_image_paths.extend(paths)
    
    # Open all generated images
    print("Opening all generated images...")
    open_images(all_image_paths)