import cv2
import time

def test_logitech_c505():
    """
    Test script specifically for Logitech C505 USB webcam
    Native resolution: 1280x720 @ 30fps
    """
    print("Testing Logitech C505 webcam...")
    
    # Try different camera indices with different backends
    backends = [None, cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    indices = [ 1, 2]  # Most common camera indices
    
    for index in indices:
        for backend in backends:
            backend_name = "Default" if backend is None else "DirectShow" if backend == cv2.CAP_DSHOW else "Media Foundation" if backend == cv2.CAP_MSMF else "Any"
            
            print(f"\nTrying camera index {index} with {backend_name} backend...")
            
            try:
                # Open camera with or without specified backend
                if backend is None:
                    cap = cv2.VideoCapture(index)
                else:
                    cap = cv2.VideoCapture(index, backend)
                
                if not cap.isOpened():
                    print(f"Failed to open camera at index {index} with {backend_name} backend")
                    continue
                
                print(f"Successfully opened camera at index {index} with {backend_name} backend")
                
                # Set Logitech C505 native resolution and framerate
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                # Test if we can read frames
                ret, frame = cap.read()
                if not ret:
                    print("Could open camera but couldn't read frames")
                    cap.release()
                    continue
                
                # Show camera information
                actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                actual_fps = cap.get(cv2.CAP_PROP_FPS)
                
                print(f"Camera details:")
                print(f"  - Resolution: {actual_width}x{actual_height}")
                print(f"  - FPS: {actual_fps}")
                print(f"  - Frame shape: {frame.shape}")
                
                # Show live feed for 5 seconds
                print("Showing camera feed for 5 seconds...")
                start_time = time.time()
                frame_count = 0
                
                while time.time() - start_time < 5:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    
                    # Add info text
                    cv2.putText(frame, f"Logitech C505", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Index: {index}, Backend: {backend_name}", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imshow("Logitech C505 Test", frame)
                    
                    # Check for ESC key
                    key = cv2.waitKey(1)
                    if key == 27:  # ESC key
                        break
                
                # Clean up
                cap.release()
                cv2.destroyAllWindows()
                
                print(f"Test complete for camera index {index} with {backend_name} backend")
                print(f"Processed {frame_count} frames in {elapsed:.2f} seconds ({fps:.2f} FPS)")
                
                # If we got this far, we found a working configuration
                return index, backend
                
            except Exception as e:
                print(f"Error with camera index {index}, {backend_name} backend: {str(e)}")
                try:
                    cap.release()
                except:
                    pass
    
    print("\nNo working camera configuration found.")
    return None, None

if __name__ == "__main__":
    index, backend = test_logitech_c505()
    
    if index is not None:
        print(f"\nSUCCESS: Use camera index {index} with backend {backend} in your ArUco code")
        
        # Code snippet to use in ArUco detection
        backend_name = "Default" if backend is None else "cv2.CAP_DSHOW" if backend == cv2.CAP_DSHOW else "cv2.CAP_MSMF" if backend == cv2.CAP_MSMF else "cv2.CAP_ANY"
        
        print("\nCode to use in your ArUco detection script:")
        if backend is None:
            print(f"cap = cv2.VideoCapture({index})")
        else:
            print(f"cap = cv2.VideoCapture({index}, {backend_name})")
        print("cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)")
        print("cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)")
        print("cap.set(cv2.CAP_PROP_FPS, 30)")
    else:
        print("\nPlease check your webcam connection and drivers.")
        print("If using Windows, try updating the Logitech drivers from the Logitech website.")