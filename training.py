from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array
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



model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(height,width,channels),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3),activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=150, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(units=3, activation='softmax'))


# Compilation of the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


history=model.fit(aug.flow(trainx,trainy,batch_size=32),
                  validation_data=(testx,testy),
                  steps_per_epoch=len(trainx)//32,
                  epochs=100,verbose=1)

model.save('detection.h5')


plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.savefig('Accuracy.png')

plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig('Loss.png')

print("Saved Model & Graph to disk")

















