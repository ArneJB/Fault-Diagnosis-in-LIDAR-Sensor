import cv2
import numpy as np

def detect_circle_opencv(image_path):
    # Load image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur to reduce noise
    gray_blurred = cv2.medianBlur(gray, 5)

    # Hough Circle Transform
    circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=40,      # Canny high threshold
        param2=20,      # lower threshold for circle center detection (important!)
        minRadius=20,    # ~20 px diameter → 10 px radius
        maxRadius=100
    )
    
    landmarks = []

    if circles is not None:
        #circles = np.squeeze(circles)        
        #for x, y, r in circles:
        #    landmarks.append((int(x), int(y), int(r)))
        circles = np.around(circles[0]).astype(int)   # shape: (N, 3)

        # If you want a single landmark (first / strongest)
        x, y, r = circles[0]

        #print(f"{len(landmarks)} circles detected:")
        #for lm in landmarks:
        #    print(lm)

    else:
        print("No circles detected")

    return x,y,r