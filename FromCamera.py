import cv2
cam=cv2.VideoCapture(0)

def camera():
        _,img=cam.read()
        return img
