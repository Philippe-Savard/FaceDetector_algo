"""
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
        
        # Display filters for landmarks
        i = 0
        for (x, y) in detector.landmarks:
            if i==31 or i==33 or i==35 or i==36 or i==39 or i==42 or i==45 or i==48 or i==54:
                cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)
            if i==27 or i==57:
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            else:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
            i+=1
        
        # Eye dimensions and scale predictor
        rightEyePixel = detector.getDistance(36, 39)
        leftEyePixel = detector.getDistance(42,45)
        
        detector.scale = ((detector.rightEyeWidth / rightEyePixel) + (detector.leftEyeWidth / leftEyePixel))/2 # average scale in cm/pixel
        
        # Other dimensions
        #mouthCM = detector.getDistance(48, 54) * detector.scale
        rightNostrilWidth = round(detector.getDistance(31,32) * detector.scale, 3)
        leftNostrilWidth = round(detector.getDistance(34,35) * detector.scale, 3)
        ForeheadToChin = round(detector.getDistance(27,57) * detector.scale, 3)
        
        # Display lines
        # Right and left Eyes
        cv2.line(frame, (detector.landmarks[39][0], detector.landmarks[39][1]), (detector.landmarks[36][0], detector.landmarks[36][1]),(0,255,255), 1, cv2.LINE_AA )
        cv2.line(frame, (detector.landmarks[45][0], detector.landmarks[45][1]), (detector.landmarks[42][0], detector.landmarks[42][1]),(0,255,255), 1, cv2.LINE_AA )
        
        # Right and left Nostrils
        cv2.line(frame, (detector.landmarks[32][0], detector.landmarks[32][1]), (detector.landmarks[31][0], detector.landmarks[31][1]),(255, 0, 255), 1, cv2.LINE_AA )
        cv2.line(frame, (detector.landmarks[35][0], detector.landmarks[35][1]), (detector.landmarks[34][0], detector.landmarks[34][1]),(255, 0, 255), 1, cv2.LINE_AA )
        
        # ForeheadToChin
        cv2.line(frame, (detector.landmarks[57][0], detector.landmarks[57][1]), (detector.landmarks[27][0], detector.landmarks[27][1]),(0,255,0), 1, cv2.LINE_AA )
        
        # Display info
        #cv2.putText(frame,'Mouth width (cm): ' + str(mouthCM), (0, 18), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
        cv2.putText(frame,'Eyes width (pixels) - Right: '+ str(rightEyePixel) + ' Left: ' + str(leftEyePixel), (0, 18), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2)
        cv2.putText(frame,'Scale (cm/pixel): '+ str(detector.scale), (0, 36), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        cv2.putText(frame,'Nostrils width (cm) - Right: '+ str(rightNostrilWidth) + ' Left: ' + str(leftNostrilWidth), (0, 54), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
        cv2.putText(frame,'Forehead to chin length (cm): '+ str(ForeheadToChin), (0, 72), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
        
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