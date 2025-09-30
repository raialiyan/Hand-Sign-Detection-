import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np 
import math
import time



cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20 
imagesize = 300 
counter = 0

folder = "data/Y"

""""overlay here need to understand how it works """
while True: 
    success , img = cap.read()
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

        else:
            k = imagesize/w
            hcalculated = math.ceil(k*h)
            imgresize = cv2.resize(imgcrop , (imagesize , hcalculated))
            imgresizeshape = imgresize.shape
            hGap= math.ceil((imagesize - hcalculated)/2)
            imgwhite[hGap: hcalculated + hGap , : ]= imgresize

        cv2.imshow("Imagecrop", imgcrop)
        cv2.imshow("ImageWhite", imgwhite)




    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key & 0xFF == 27:
        break # ESC key
    if key == ord("s"):
        counter +=1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgwhite)
        print(counter)
     
  