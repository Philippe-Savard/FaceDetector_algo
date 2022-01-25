"""
Created on Mon Jan 24 17:57:15 2022



@author: Philippe Savard
"""
from FaceDetector import FaceDetector
import cv2
import numpy as np
# WEBCAM INPUT
webcam = cv2.VideoCapture(0)

# Create the main frame window
cv2.namedWindow("Facial Detection")

detector = FaceDetector()

while True:
    try:

        _, frame = webcam.read()
        
        detector.findLandmarks(frame)
        
        i = 0
        for (x, y) in detector.landmarks:
            if i==36 or i==39 or i==54 or i==48:
                cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)
            i+=1
        
        # Eye dimensions
        dx = pow(detector.landmarks[39][0] - detector.landmarks[36][0], 2)
        dy = pow(detector.landmarks[39][1] - detector.landmarks[36][1], 2)
        
        rightEyePixel = np.sqrt(dx + dy)
        
        detector.scale = detector.rightEyeWidth / rightEyePixel # cm/pixel
        
        # Mouth dimensions for testing
        dmx = pow(detector.landmarks[54][0] - detector.landmarks[48][0], 2)
        dmy = pow(detector.landmarks[54][1] - detector.landmarks[48][1], 2)
        
        mouthCM = np.sqrt(dmx + dmy) * detector.scale
        
        # Display keyboard input menu
        cv2.putText(frame, str(mouthCM), (0, 18),cv2.FONT_HERSHEY_PLAIN, 1.25, (0, 255, 0), 2)
        
        cv2.imshow("Facial Detection", frame)
        
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            print("[INFO] Received stop instruction. Stopping execution now...")
            break
        
    except:
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            print("[INFO] Received stop instruction. Stopping execution now...")
            break

        print("[ERROR] An unexpected error occurred. Press [ESC] to quit")
        continue

webcam.release()
cv2.destroyAllWindows()