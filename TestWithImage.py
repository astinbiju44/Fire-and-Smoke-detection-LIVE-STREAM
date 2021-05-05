from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import cv2
model = load_model('detection.h5')
print("Model Loaded Successfully")

def classify(img_file):
    img_name = img_file
    test_image=cv2.imread(img_name)
    test_image=cv2.resize(test_image,(96,96))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    arr = np.array(result[0])
    print(arr)
    maxx = np.amax(arr)
    max_prob = arr.argmax(axis=0)
    max_prob = max_prob + 1
    classes=["Neutral","Fire","Smoke"]
    result = classes[max_prob - 1]
    print(img_name,result)


import os
path = 'D:\OTHERS\python\Fire-and-Smoke-detection-LIVE-STREAM\Testing'
files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
   for file in f:
       files.append(os.path.join(r, file))

for f in files:
   classify(f)
   print('\n')