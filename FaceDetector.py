# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 18:40:40 2022

This file contains a draft for the face measurements algorithm

@author: Philippe Savard
"""

import cv2
import numpy as np
from imutils import face_utils
import dlib
from os import path

class FaceDetector:
    """
        FaceDetector contains the necessary 
    """
    # Find the predictor on the dlib official website: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 
    PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
    
    AVERAGE_EYE_WIDTH = 3.0 # in cm
    
    def __init__(self, image=None, predictorPath = PREDICTOR_PATH):
        """
            Default constructor for the FaceDetector class
            
            Parameters
            ----------
            image : image
                input image to be measured
            
            predictorPath : string
                Path to the shape predictor used to find the landmarks on the image.
        """
        try:
            
            self.landmarks = []
            
            self.scale = 0 # Scale in cm / 
            
            self.referenceRight = 2.9
            self.referenceLeft = 2.591
            
            self.rightEye = 0
            self.leftEye = 0
            
            self.rightNostril = 0
            self.leftNostril = 0
            self.nose = 0
            
            self.foreheadChin = 0
            
            # DLIB FACE DETECTOR AND KEYPOINTS PREDICTOR
            self.detector = dlib.get_frontal_face_detector()

            self.predictor = None
            if path.exists(predictorPath):
                self.predictor = dlib.shape_predictor(predictorPath)
            else: 
                raise
            
            # Extract landmarks if an image is provided
            if image != None:
                self.landmarks = self.findLandmarks(image)
            
        except:
            print("Damn, that sucks part 1")
            # some error code
    
    def findLandmarks(self, image):
        
        """
            This method normalizes the input picture and finds the landmarks associated to the faces in the picture.
            
            Returns
            -------
            Returns an array of landmarks in the pictures space
        """
        try:
            # Convert the input image to grayscale
            grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            if self.predictor != None:
                
                # Detect faces in the grayscale image
                faces = self.detector(grayImage, 0)
                
                # For all faces in the frame
                for (i, face) in enumerate(faces):
    
                    # Finding the facial landmarks
                    self.landmarks = self.predictor(grayImage, face)
    
                    # Converts the landmarks into a 2D numpy array of x, y coordinnates
                    self.landmarks = face_utils.shape_to_np(self.landmarks)
                
        except:
            print("Damn, that sucks part 2")
            # some error code
    
    def getDistance(self, point1, point2):
        """
            This method return the approximate distance in cm between two landmarks.
            
            Parameters
            ----------
            point1 : number
                Represents the reference number of the first landmark
                
            point2 : number
                Represents the reference number of the second landmark
            
            Returns
            -------
            Returns the approximate distance between two landmarks
        """
        dx = pow(self.landmarks[point2][0] - self.landmarks[point1][0], 2)
        dy = pow(self.landmarks[point2][1] - self.landmarks[point1][1], 2)
        
        return np.sqrt(dx + dy)