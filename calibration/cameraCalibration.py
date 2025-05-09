import numpy as np
import cv2
import glob
import os
import time

def calibrate_camera(chessboard_size=(9,6), square_size=0.025, camera_id=0, 
                     num_images=20, output_file="camera_calibration.npz"):
    """
    Calibrate a camera using a chessboard pattern.
    
    Args:
        chessboard_size: Number of inner corners in the chessboard (width, height)
        square_size: Size of a chessboard square in meters
        camera_id: Camera ID to use
        num_images: Number of images to capture for calibration
        output_file: File to save the calibration parameters
        
    Returns:
        ret: RMS re-projection error
        mtx: Camera matrix
        dist: Distortion coefficients
        rvecs: Rotation vectors
        tvecs: Translation vectors
    """
    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ... (8,5,0)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size  # Convert to real world units (meters)

    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    # Initialize camera capture
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None

    # Set camera resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Create directory for calibration images
    calib_dir = "calibration_images"
    if not os.path.exists(calib_dir):
        os.makedirs(calib_dir)

    # Clear any old images
    for file in glob.glob(f"{calib_dir}/*.jpg"):
        os.remove(file)

    print("Camera calibration started...")
    print("Hold a chessboard in front of the camera.")
    print(f"Press SPACE to capture an image (need {num_images} images).")
    print("Press ESC to cancel.")

    captured = 0
    last_capture_time = 0

    while captured < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Create a copy for display
        display = frame.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners
        ret_corners, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        # If found, refine and draw corners
        if ret_corners:
            # Refine corners to sub-pixel accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Draw the corners
            cv2.drawChessboardCorners(display, chessboard_size, corners, ret_corners)
            
            # Add text indicating good pattern
            cv2.putText(display, "Chessboard detected! Press SPACE to capture", 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            # Add text indicating no pattern found
            cv2.putText(display, "No chessboard detected", 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Display image counter
        cv2.putText(display, f"Captured: {captured}/{num_images}", 
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Show the frame
        cv2.imshow('Camera Calibration', display)

        # Handle keyboard input
        key = cv2.waitKey(1)
        
        # ESC key to exit
        if key == 27:
            print("Calibration cancelled")
            cap.release()
            cv2.destroyAllWindows()
            return None
        
        # SPACE key to capture when chessboard is detected
        current_time = time.time()
        if key == 32 and ret_corners and (current_time - last_capture_time) > 1.0:
            img_file = f"{calib_dir}/calib_{captured:02d}.jpg"
            cv2.imwrite(img_file, frame)
            
            # Store image and object points
            imgpoints.append(corners)
            objpoints.append(objp)
            
            captured += 1
            last_capture_time = current_time
            print(f"Image {captured}/{num_images} captured")

    cap.release()
    cv2.destroyAllWindows()

    if captured < 5:
        print("Not enough images captured for calibration")
        return None

    print("Processing calibration...")

    # Get a sample image shape
    sample_img = cv2.imread(f"{calib_dir}/calib_00.jpg")
    gray = cv2.cvtColor(sample_img, cv2.COLOR_BGR2GRAY)
    img_shape = gray.shape[::-1]  # width, height

    # Calculate camera calibration parameters
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None)

    # Save the camera calibration results
    np.savez(output_file, 
             camera_matrix=mtx, 
             dist_coeffs=dist, 
             rvecs=rvecs, 
             tvecs=tvecs)
    
    print(f"Calibration complete! Parameters saved to {output_file}")
    print(f"Re-projection error: {ret}")
    
    # Show the calibration results
    print("\nCamera Matrix:")
    print(mtx)
    print("\nDistortion Coefficients:")
    print(dist)

    # Test undistortion on a sample image
    print("\nDisplaying test undistortion. Press any key to continue.")
    test_undistortion(output_file, f"{calib_dir}/calib_00.jpg")

    return ret, mtx, dist, rvecs, tvecs

def test_undistortion(calibration_file, test_image):
    """Test undistortion on an image using calibration parameters"""
    # Load the calibration parameters
    data = np.load(calibration_file)
    camera_matrix = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']
    
    # Load the test image
    img = cv2.imread(test_image)
    if img is None:
        print(f"Error: Could not load test image {test_image}")
        return
    
    h, w = img.shape[:2]
    
    # Refine camera matrix
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    
    # Undistort the image
    undist = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_mtx)
    
    # Crop the image
    x, y, w, h = roi
    undist = undist[y:y+h, x:x+w]
    
    # Display the original and undistorted images
    result = np.hstack((img, cv2.resize(undist, (img.shape[1], img.shape[0]))))
    cv2.imshow("Original vs. Undistorted", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def load_calibration(calibration_file):
    """Load camera calibration parameters from a file"""
    if not os.path.exists(calibration_file):
        print(f"Error: Calibration file {calibration_file} not found")
        return None
    
    data = np.load(calibration_file)
    camera_matrix = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']
    
    return camera_matrix, dist_coeffs

# Example usage
if __name__ == "__main__":
    # Calibrate camera
    # Note: Adjust chessboard_size to match your printed chessboard
    # For a standard 9x6 chessboard (10x7 squares), use (9,6)
    calibrate_camera(
        chessboard_size=(9, 6),   # Number of inner corners on the chessboard
        square_size=0.025,        # Size of chessboard square in meters (2.5cm)
        camera_id=1,              # Camera ID
        num_images=20,            # Number of images to capture
        output_file="camera_calibration.npz"  # Output file
    )