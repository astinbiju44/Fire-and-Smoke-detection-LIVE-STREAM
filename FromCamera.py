import cv2
cam=cv2.VideoCapture(0)

def camera():
    while True:
        _,img=cam.read()
        return img
    