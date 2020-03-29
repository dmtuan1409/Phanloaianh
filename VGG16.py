from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.applications import VGG16
from keras.layers import Input
from keras.models import Model
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten
import numpy as np
import random
import os
import cv2
def load_sequence(folder):
    X = []
    name = []
    for j in os.listdir(os.path.expanduser(folder)):
        path2 = folder + j + "/"
        for i in os.listdir(os.path.expanduser(path2)):
            path3 = path2 + i
            img = cv2.imread(path3)
            img = img.reshape(224,224,3)
            image_name = j
            name.append(image_name)
            X.append(img)
    return np.array(X), np.array(name)

x, y= load_sequence("../Images/Flower/")

# Chuyen nhan sang so
labelencoder=LabelEncoder()
y = labelencoder.fit_transform(y)
lb = LabelBinarizer()
y = lb.fit_transform(y)

# Load model VGG 16 của ImageNet dataset, include_top=False để bỏ phần Fully connected layer ở cuối.
baseModel = VGG16(weights='imagenet', include_top=False,  input_tensor=Input(shape=(224, 224, 3)))
# Lấy output của ConvNet trong VGG16
fcHead = baseModel.output
# Flatten trước khi dùng FCs
fcHead = Flatten(name='flatten')(fcHead)
# Thêm FC
fcHead = Dense(256, activation='relu')(fcHead)
fcHead = Dropout(0.5)(fcHead)
# Output layer với softmax activation
fcHead = Dense(17, activation='softmax')(fcHead)
# Xây dựng model bằng việc nối ConvNet của VGG16 và fcHead
model = model = Model(inputs=baseModel.input, outputs=fcHead)
#chia tap train test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# augmentation cho training data
aug_train = ImageDataGenerator(rescale=1./255, rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2,zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
# augementation cho test
aug_test= ImageDataGenerator(rescale=1./255)
# freeze VGG model
for layer in baseModel.layers:
    layer.trainable = False
opt = RMSprop(0.001)
model.compile(opt, 'categorical_crossentropy', ['accuracy'])
numOfEpoch = 25
H = model.fit_generator(aug_train.flow(X_train, y_train, batch_size=32),steps_per_epoch=len(X_train) // 32,validation_data=(aug_test.flow(X_test, y_test, batch_size=32)),validation_steps=len(X_test) // 32,epochs=numOfEpoch)
# unfreeze some last CNN layer:
for layer in baseModel.layers[15:]:
    layer.trainable = True
numOfEpoch = 35
opt = SGD(0.001)
model.compile(opt, 'categorical_crossentropy', ['accuracy'])
H = model.fit_generator(aug_train.flow(X_train, y_train, batch_size=32),steps_per_epoch=len(X_train)//32,validation_data=(aug_test.flow(X_test, y_test, batch_size=32)),validation_steps=len(X_test)//32,epochs=numOfEpoch)