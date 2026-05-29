import cv2
import numpy as np

def map_arena():
    """
    Task 2B: Perspective Transformation and Coordinate Mapping
    """
    # Initialize the output dictionary
    result = {
        "corner_points_detected": [],
        "robot_pixel_coord": [],
        "robot_real_world_coord": []
    }

    # ==========================================
    # STEP 1: Corner Detection (Color Tracking)
    # ==========================================
    # TODO: Read the target image 'test_images/angled_arena.png'
    image = cv2.imread("test_images/angled_arena.png")
    if image is None:
        image = cv2.imread("angled_arena.png")

    
    # TODO: Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # TODO: Create HSV masks to isolate the Red, Green, Blue, and Yellow corners
    mask_red1 = cv2.inRange(hsv, np.array([0,   100, 100]), np.array([10,  255, 255]))
    mask_red2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
    mask_red  = cv2.bitwise_or(mask_red1, mask_red2)
 
    mask_green  = cv2.inRange(hsv, np.array([40,  50, 50]),  np.array([90,  255, 255]))
    mask_blue   = cv2.inRange(hsv, np.array([100, 50, 50]),  np.array([130, 255, 255]))
    mask_yellow = cv2.inRange(hsv, np.array([20,  100, 100]), np.array([35,  255, 255]))
 
    def get_centroid(mask):
        """Return (cx, cy) of the largest contour in the mask."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        # pick largest contour
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return [cx, cy]
    
    # TODO: Find contours for each mask and calculate the centroid (cx, cy) using moments (M["m10"] / M["m00"])
    cx_r = get_centroid(mask_red)     # Top-Left     (Red)
    cx_g = get_centroid(mask_green)   # Top-Right    (Green)
    cx_b = get_centroid(mask_blue)    # Bottom-Right (Blue)
    cx_y = get_centroid(mask_yellow)  # Bottom-Left  (Yellow)
    
    # TODO: Store the coordinates in the exact order: [Top-Left(Red), Top-Right(Green), Bottom-Right(Blue), Bottom-Left(Yellow)]
    # result["corner_points_detected"] = [[cx_r, cy_r], [cx_g, cy_g], [cx_b, cy_b], [cx_y, cy_y]]
    result["corner_points_detected"] = [cx_r, cx_g, cx_b, cx_y]


    # ==========================================
    # STEP 2: Perspective Transformation
    # ==========================================
    # TODO: Define your source points as a float32 numpy array (the 4 centroids calculated above)
    src_points = np.array(result["corner_points_detected"], dtype=np.float32)
    
    # TODO: Define your destination points as a flat 500x500 pixel square
    # Example: [[0,0], [500,0], [500,500], [0,500]]
    dst_points = np.array([[0, 0], [500, 0], [500, 500], [0, 500]], dtype=np.float32)

    
    # TODO: Use cv2.getPerspectiveTransform() to calculate the 3x3 Homography Matrix
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # TODO: Apply cv2.warpPerspective() to flatten the angled arena into a 500x500 top-down view
    warped = cv2.warpPerspective(image, M, (500, 500))
    cv2.imwrite("birdsview.png", warped)


    # ==========================================
    # STEP 3: Robot Detection on Warped Arena
    # ==========================================
    # TODO: On the NEW warped 500x500 image, initialize an ArUco detector (DICT_4X4_50)
    aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector     = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners, ids, _ = detector.detectMarkers(warped)

        
    if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id == 1:  # Robot marker ID
                    # Calculate center pixel coordinates of the detected marker
                    c = corners[i][0]  # Get the 4 corner points of the detected marker
                    cx = int(c[:, 0].mean())  # Average x-coordinates
                    cy = int(c[:, 1].mean())  # Average y-coordinates
                    result["robot_pixel_coord"] = [cx, cy]
                    break
    # TODO: Detect the marker representing the robot (ID 1)


    
    # TODO: Calculate the center pixel coordinates (cx, cy) of the detected marker
    # result["robot_pixel_coord"] = [cx, cy]
    




    # ==========================================
    # STEP 4: Real-World Coordinate Conversion
    # ==========================================
    # Context: The 500x500 pixel warped image represents a physical arena of 200cm x 200cm.
    # The top-left corner is the origin [0.0, 0.0] cm.
    
    # TODO: Use linear scaling to convert the robot's pixel coordinates to real-world centimeters.
    # result["robot_real_world_coord"] = [x_cm, y_cm]
    if result["robot_pixel_coord"]:
        x_cm = round((result["robot_pixel_coord"][0] / 500) * 200, 1)
        y_cm = round((result["robot_pixel_coord"][1] / 500) * 200, 1)
        result["robot_real_world_coord"] = [x_cm, y_cm]


    return result

if __name__ == "__main__":
    # Test your function
    output = map_arena()
    print("Task 2B Output:")
    print(output)