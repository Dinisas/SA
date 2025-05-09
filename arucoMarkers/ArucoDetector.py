import cv2
import numpy as np
import os

def detect_aruco_markers(camera_id, dictionary_id, marker_length, camera_matrix, dist_coeffs):
    """
    Detect ArUco markers from camera feed.
    """
    # Set up detector
    detector_params = cv2.aruco.DetectorParameters()
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
    
    # Set up video capture for camera index 1
    cap = cv2.VideoCapture(camera_id)
    
    # Set optimal properties for the webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera at index {camera_id}")
        return
    
    # Define marker corner points
    obj_points = np.zeros((4, 1, 3), dtype=np.float32)
    obj_points[0, 0] = [-marker_length/2.0, marker_length/2.0, 0]
    obj_points[1, 0] = [marker_length/2.0, marker_length/2.0, 0]
    obj_points[2, 0] = [marker_length/2.0, -marker_length/2.0, 0]
    obj_points[3, 0] = [-marker_length/2.0, -marker_length/2.0, 0]
    
    print("Starting ArUco marker detection...")
    print("Press ESC key to exit")
    
    # Main detection loop
    while cap.grab():
        ret, image = cap.retrieve()
        if not ret:
            break
        
        # Detect markers
        corners, ids, rejected = detector.detectMarkers(image)
        
        # Calculate pose with camera calibration
        rvecs = tvecs = None
        if ids is not None and len(ids) > 0:
            rvecs = []
            tvecs = []
            for i in range(len(ids)):
                # Use solvePnP for each marker
                _, rvec, tvec = cv2.solvePnP(
                    obj_points, corners[i], camera_matrix, dist_coeffs)
                rvecs.append(rvec)
                tvecs.append(tvec)
        
        # Draw results
        image_copy = image.copy()
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(image_copy, corners, ids)
            
            # Draw axes for pose estimation
            for i in range(len(ids)):
                cv2.drawFrameAxes(image_copy, camera_matrix, dist_coeffs, 
                                  rvecs[i], tvecs[i], marker_length * 1.5, 2)
        
        cv2.imshow("Detected Markers", image_copy)
        
        # Exit on ESC key
        key = cv2.waitKey(10)
        if key == 27:  # ESC key
            break
    
    cap.release()
    cv2.destroyAllWindows()

def load_calibration(calibration_file):
    """Load camera calibration parameters from a file"""
    print(f"Loading calibration from: {calibration_file}")
    try:
        data = np.load(calibration_file)
        camera_matrix = data['camera_matrix']
        dist_coeffs = data['dist_coeffs']
        print("Calibration loaded successfully!")
        return camera_matrix, dist_coeffs
    except Exception as e:
        print(f"Error loading calibration: {e}")
        print("Please check if the calibration file exists at the specified path.")
        exit(1)

# Main code - load the calibration and run detection
if __name__ == "__main__":
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up one level from the script directory to find the calibration folder
    # This works regardless of which computer it runs on, as long as the directory structure is preserved
    calibration_path = os.path.join(os.path.dirname(script_dir), "calibration", "camera_calibration.npz")
    
    # Load the calibration
    camera_matrix, dist_coeffs = load_calibration(calibration_path)
    
    # Run detection with camera index 1
    detect_aruco_markers(
    camera_id=1,
    dictionary_id=cv2.aruco.DICT_5X5_100,  # Change to match your generated markers
    marker_length=0.16,
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs
    )