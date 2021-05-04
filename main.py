import cv2
import numpy as np
from FromCamera import camera
from FromMobile import MobileCamera

while True:
    img=MobileCamera()
    camimg=camera()
    cv2.imshow("Camera Feed",camimg)
    cv2.imshow("Mobile Feed", img)
    cv2.waitKey(1)

