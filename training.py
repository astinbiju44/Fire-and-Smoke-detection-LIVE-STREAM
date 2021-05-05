from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical,plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization,Conv2D,MaxPooling2D,Activation,Flatten,Dropout,Dense
from tensorflow.keras import backend as k
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import os
import glob





#To append images and its labels
data=[]
labels=[]




#To load image files from the dataset
imagefiles=[f for f in glob.glob(r'FIRE-SMOKE-DATASET'+"/**/*",recursive=True) if not os.path.isdir(f)]
random.shuffle(imagefiles) #to balance the weight
print("Total number image readed : "+str(len(imagefiles)))



(height,width,channels)=(96,96,3)



#converting images to array and labeling into categories
for img in imagefiles:
    #print(img)
    image=cv2.imread(img)
    image=cv2.resize(image,(96,96))
    image=img_to_array(image)
    data.append(image)

    label=img.split(os.path.sep)[-2]  #every seperation is made whereever there is backslash and second last is the class name ie -2
    if label=="Neutral":
        label=0
    elif label=="Fire":
        label=1
    else:
        label=2
    #print(label)
    labels.append([label]) #[[0],[2],[1],......]

#data preprocessing
data=np.array(data,dtype="float")/255.0  #it will convert into array of values between 0 & 1
labels=np.array(labels)


#spliting dataset into train and test
(trainx,testx,trainy,testy)=train_test_split(data,labels,test_size=0.2,random_state=42)
trainy=to_categorical(trainy,num_classes=3)
testy=to_categorical(testy,num_classes=3)


#augmenting dataset
aug=ImageDataGenerator(rotation_range=25,width_shift_range=0.1,
                       height_shift_range=0.1,shear_range=0.2,zoom_range=0.2,
                       horizontal_flip=True,fill_mode="nearest")


#defining model
def buildmodel(width,height,depth,classes):
    model=Sequential()
    inputshape=(height,width,depth)
    changeDim=-1

    if k.image_data_format()=="channels_first": #return a string either 'Channels first' or 'chancels last'
        inputshape=(depth,height,width)
        changeDim = 1

    model.add(Conv2D(filters=32, kernel_size=(3, 3),padding="same", activation='relu', input_shape=inputshape))
    model.add(BatchNormalization(axis=changeDim))
    model.add((MaxPooling2D(pool_size=(3, 3))))
    model.add(Dropout(rate=0.25))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same",activation='relu'))
    model.add(BatchNormalization(axis=changeDim))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation='relu'))
    model.add(BatchNormalization(axis=changeDim))
    model.add((MaxPooling2D(pool_size=(2, 2))))
    model.add(Dropout(rate=0.25))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation='relu'))
    model.add(BatchNormalization(axis=changeDim))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation='relu'))
    model.add(BatchNormalization(axis=changeDim))
    model.add((MaxPooling2D(pool_size=(2, 2))))
    model.add(Dropout(rate=0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.5))

    model.add(Dense((classes), activation='softmax'))

    return model

model=buildmodel(width=width,height=height,depth=channels,classes=3)

















