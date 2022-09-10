# ======================== Import basic python Libraries ================
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import dlib

# ======================= Import deep learning libariries
import keras
from keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout
from keras.models import Sequential
import tensorflow as tf

folder_left = 'database/LEFT'
folder_right = 'database/RIGHT'

subfolders_left=os.listdir(folder_left)
subfolders_right=os.listdir(folder_right)

fullpaths = []
labels = []

part=[]
counter=0
for f in subfolders_left:
    fi=os.path.join(folder_left,f)
    for file in os.listdir(fi):
        fullpaths.append(os.path.join(fi,file))
        labels.append(counter)
        part.append('left')
    counter+=1


counter=0       
for f in subfolders_right:
    fi=os.path.join(folder_right,f)
    for file in os.listdir(fi):
        fullpaths.append(os.path.join(fi,file))
        labels.append(counter)
        part.append('right')
    counter+=1

## ============== specify the 68 landmark predictor ===============
predictor_path = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


## =========== Creating image array =================
X = []
image_size = 32
for i in range(len(fullpaths)):
    img =  cv2.imread(fullpaths[i])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    w,h = gray.shape

    rectangle = dlib.rectangle(left=0, top=0, right=w, bottom=h)
    landmarks_dlib = predictor(gray, rectangle)

    def tuple_from_dlib_shape(index):
        p = landmarks_dlib.part(index)
        return (p.x, p.y)

    num_landmarks = landmarks_dlib.num_parts
    #print('Number of facial landmark :',num_landmarks)

    landmark = np.array([tuple_from_dlib_shape(i) for i in range(num_landmarks)])
    #print(landmark)

    x1,y1=landmark[17]
    x2,y2=landmark[21]
    x3,y3=landmark[41]

    x1,y1=landmark[22]
    x2,y2=landmark[26]
    x3,y3=landmark[47]
            
    try:
        im =  img[y1-100:y3+50,x1-200:x2+100]
        im = cv2.resize(im, (image_size,image_size))

    except:
        im =  img[y1-20:y3+20,x1-50:x2+20]
        im = cv2.resize(im, (image_size,image_size))
 
    X.append(im)

#cv2.imshow('testimage',im)
#cv2.waitKey(0)

BATCH_NORM = False

epochs = 25
data_augmentation = True


X=np.array(X)
X=np.reshape(X, (len(X),image_size,image_size,3))

X= X/255
y=np.array(labels)
num_classes = len(np.unique(y))

test_index=[]
train_index= []
test_size = .2
np.random.seed(0)
for t in range(num_classes):
  idx=np.where(y==t)[0]
  
  n=int(test_size*len(idx))
  tidx=list(np.random.choice(idx,n, replace=False))

  t2idx = [i for i in idx if i not in tidx]
  test_index.extend(tidx)
  train_index.extend(t2idx)
  
x_train,x_test,y_train,y_test= X[train_index],X[test_index],y[train_index],y[test_index]


### VGG16 Model ###
nclasses=len(np.unique(labels))
# Define Model

cnn = Sequential()
cnn.add(Conv2D(input_shape=(image_size, image_size, 3), filters=64, kernel_size=(3, 3),
                 padding='same', activation='relu'))
cnn.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

cnn.add(Flatten())
cnn.add(Dense(128, activation="relu"))
cnn.add(Dropout(0.2))
cnn.add(Dense(num_classes, activation="sigmoid"))
print(cnn.summary())


cnn.compile(loss="sparse_categorical_crossentropy", optimizer="adam",metrics=['accuracy'])


hist=cnn.fit(x_train, y_train, batch_size=10, epochs=100, validation_data=(x_test,y_test))
       

plt.plot(hist.history["accuracy"])
plt.plot(hist.history['val_accuracy'])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])

cnn_json=cnn.to_json()
with open ("weights_json.json","w") as file:
    file.write(cnn_json)

cnn.save_weights("weights.h5") 
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
plt.show(block=True)
