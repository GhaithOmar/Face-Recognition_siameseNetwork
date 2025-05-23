import os
import cv2
import dlib
import pickle
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from datetime import datetime
from imutils import face_utils
from vgg_face import preprocess_input, VGG16
from siameseNetwork import SiameseNetwork
base_dir = ".\\model\\"
checkpoint_path = os.path.join(base_dir, 'siamese-1')
'''
The following code cell is taken from the source code of keras_vggface.'
I tried using the preprocess_input function provided by tf.keras but they provide different results.
To my knowledge, it seems that the mean values which are subtracted in each image are different.
'''
K = tf.keras.backend
vggface = VGG16()


for layer in vggface.layers[:-1]:
    layer.trainable = False
model = SiameseNetwork(vggface)

checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore(checkpoint_path)
data_dir = r'.\img'
people = sorted(os.listdir(data_dir))
face_detector = dlib.get_frontal_face_detector()
features = []
dumpable_features = {}

for person in people:
    person_path = os.path.join(data_dir, person)
    images = []
    for image in os.listdir(person_path):
        image_path = os.path.join(person_path, image)
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 0)
        if len(faces) == 1:
            for face in faces:
                face_bounding_box = face_utils.rect_to_bb(face)
                if all(i >= 0 for i in face_bounding_box):
                    [x, y, w, h] = face_bounding_box
                    frame = img[y:y + h, x:x + w]
                    frame = cv2.resize(frame, (224, 224))
                    frame = np.asarray(frame, dtype=np.float64)
                    images.append(frame)
    images = np.asarray(images)
    images = preprocess_input(images)
    images = tf.convert_to_tensor(images)
    feature = model.get_features(images)
    feature = tf.reduce_mean(feature, axis=0)
    features.append(feature.numpy())
    dumpable_features[person] = feature.numpy()

cap = cv2.VideoCapture(0)
count = 0
name = 'not identified'
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray, 0)
    for face in faces:
        face_bounding_box = face_utils.rect_to_bb(face)
        if all(i >= 0 for i in face_bounding_box):
            [x, y, w, h] = face_bounding_box
            frame = img[y:y + h, x:x + w]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            frame = cv2.resize(frame, (224, 224))
            frame = np.asarray(frame, dtype=np.float64)
            frame = np.expand_dims(frame, axis=0)
            frame = preprocess_input(frame)
            feature = model.get_features(frame)
                
            dist = tf.norm(features - feature, axis=1)
            name = 'not identified'
            loc = tf.argmin(dist)
            if dist[loc] < 0.9:
                name = people[loc]
            else:
#                     print(dist.numpy())
                pass
                    
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, name, (x, y-5), font_face, 0.8, (0,0,255), 3)
    cv2.imshow('Image', img)
    k = cv2.waitKey(1)
    if k ==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
 