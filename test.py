import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np 
import math
import time
import tensorflow



cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20 
imagesize = 300 
counter = 0
labels = ["A", "B" , "C", "D" ,"E" , "F" , "G" , "H" , "I" , "Y"  ]

folder = "data/Y"

""""overlay here need to understand how it works """
while True: 
    success , img = cap.read()
    imgoutput = img.copy()
    hands , img = detector.findHands(img)
                                          #img = cv2.flip(img, 1) flipping the image 
    if hands: 
        hand = hands[0]
        x ,y, w , h = hand['bbox']
        imgwhite = np.ones((imagesize , imagesize, 3 ) , np.uint8)*255 #keeping it square 
        
        imgcrop = img[y - offset : y + h + offset , x - offset : x + w + offset]
        imgcropshape = imgcrop.shape

        aspectratio = h/w

        if aspectratio >1 :
            k = imagesize/h 
            wcalculated = math.ceil(k*w)
            imgresize = cv2.resize(imgcrop , (wcalculated , imagesize))
            imgresizeshape = imgresize.shape
            wGap= math.ceil((imagesize - wcalculated)/2)
            imgwhite[: , wGap: wcalculated+wGap]= imgresize
            prediction , index = classifier.getPrediction(imgwhite, draw=False)
            print(prediction , index)

        else:
            k = imagesize/w
            hcalculated = math.ceil(k*h)
            imgresize = cv2.resize(imgcrop , (imagesize , hcalculated))
            imgresizeshape = imgresize.shape
            hGap= math.ceil((imagesize - hcalculated)/2)
            imgwhite[hGap: hcalculated + hGap , : ]= imgresize
            prediction , index = classifier.getPrediction(imgwhite, draw=False)
        
        cv2.rectangle(imgoutput , (x- offset,y-offset -50 ), (x - offset +90  , y - offset), (255 , 0 , 255) , cv2.FILLED)
        cv2.putText(imgoutput, labels[index] , (x,y-26), cv2.FONT_HERSHEY_COMPLEX,1.7,(255,255,255),2)
        cv2.rectangle(imgoutput , (x- offset,y-offset ), (x +w + offset  , y +h +offset), (255 , 0 , 255) , 4)
        cv2.imshow("Imagecrop", imgcrop)
        cv2.imshow("ImageWhite", imgwhite)




    cv2.imshow("Image", imgoutput)
    key = cv2.waitKey(1)
    if key & 0xFF == 27:
        break # ESC key

  