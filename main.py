"""
@author: Philippe Savard
"""
from FaceDetector import FaceDetector
import cv2

# WEBCAM INPUT
webcam = cv2.VideoCapture(0)

# Create the main frame window
cv2.namedWindow("Facial Detection")

detector = FaceDetector()

while True:
    try:

        _, frame = webcam.read()
        h, w, c = frame.shape
        
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
        
        detector.rightEye = detector.getDistance(36, 39)
        detector.leftEye = detector.getDistance(42, 45)
        
        detector.scale = ((detector.referenceRight / detector.rightEye) + (detector.referenceLeft / detector.leftEye))/2 # average scale in cm/pixel
        
        detector.rightNostril = detector.getDistance(31,32) * detector.scale
        detector.leftNostril = detector.getDistance(34,35) * detector.scale
        detector.foreheadChin = detector.getDistance(27,57) * detector.scale
        detector.nose = detector.getDistance(31, 35) * detector.scale
        
        # Display measurements line
        
        # Right and left Eyes
        cv2.line(frame, (detector.landmarks[39][0], detector.landmarks[39][1]), (detector.landmarks[36][0], detector.landmarks[36][1]),(0,255,255), 1, cv2.LINE_AA )
        cv2.line(frame, (detector.landmarks[45][0], detector.landmarks[45][1]), (detector.landmarks[42][0], detector.landmarks[42][1]),(0,255,255), 1, cv2.LINE_AA )
        
        # Right and left Nostrils
        cv2.line(frame, (detector.landmarks[32][0], detector.landmarks[32][1]), (detector.landmarks[31][0], detector.landmarks[31][1]),(255, 0, 255), 1, cv2.LINE_AA )
        cv2.line(frame, (detector.landmarks[35][0], detector.landmarks[35][1]), (detector.landmarks[34][0], detector.landmarks[34][1]),(255, 0, 255), 1, cv2.LINE_AA )
        
        # Nose
        cv2.line(frame, (detector.landmarks[31][0], detector.landmarks[31][1]), (detector.landmarks[35][0], detector.landmarks[35][1]),(255,165,0), 1, cv2.LINE_AA )
        
        # ForeheadToChin
        cv2.line(frame, (detector.landmarks[57][0], detector.landmarks[57][1]), (detector.landmarks[27][0], detector.landmarks[27][1]),(0,255,0), 1, cv2.LINE_AA )
        
        # Frame pading
        cv2.rectangle(frame, (0,0),(w, 100), (0,0,0), -1) # Top
        cv2.rectangle(frame, (0,h-92),(w, h), (0,0,0), -1) # Bottom  
        cv2.rectangle(frame, (0,0),(160, h), (0,0,0), -1) # Right
        cv2.rectangle(frame, (w-160,0),(w, h), (0,0,0), -1) # Left
        
        # Display info on screen
        cv2.putText(frame,'Eyes width (pixels) - Right: '+ str(round(detector.rightEye, 3)) + ' Left: ' + str(round(detector.leftEye, 3)), (0, 18), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2)
        cv2.putText(frame,'Scale (cm/pixel): '+ str(round(detector.scale, 4)), (0, 36), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        cv2.putText(frame,'Nostrils width (cm) - Right: '+ str(round(detector.rightNostril, 3)) + ' Left: ' + str(round(detector.leftNostril, 3)), (0, 54), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
        cv2.putText(frame,'Nose width (cm): '+ str(round(detector.nose, 3)), (0, 72), cv2.FONT_HERSHEY_PLAIN, 1, (255,165,0), 2)
        cv2.putText(frame,'Forehead to chin length (cm): '+ str(round(detector.foreheadChin, 3)), (0, 90), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
        

        cv2.imshow("Facial Detection", frame)
        
        # Exit state
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

# Close the application
webcam.release()
cv2.destroyAllWindows()