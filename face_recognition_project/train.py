import os
import cv2
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from datetime import datetime
from sklearn.utils import shuffle
from vgg_face import preprocess_input, VGG16
from siameseNetwork import SiameseNetwork

epochs = 50

optimizer = tf.keras.optimizers.Adam(learning_rate=0.00006)
binary_cross_entropy = tf.keras.losses.BinaryCrossentropy()

# base_dir = "."
base_dir = r"C:\Users\Desktop\face_recognition_project\model"

stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = os.path.join(base_dir, 'logs/func/%s' % stamp)
writer = tf.summary.create_file_writer(logdir)

scalar_logdir = os.path.join(base_dir, 'logs/scalars/%s' % stamp)
file_writer = tf.summary.create_file_writer(scalar_logdir + "/metrics")

checkpoint_path = os.path.join(base_dir, 'logs/model/siamese')
weights_path = r'.\weights\rcmalli_vggface_tf_vgg16.h5'

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataset_path, batch_size=5, shuffle=True):
        self.dataset = self.curate_dataset(dataset_path)
        self.dataset_path = dataset_path
        self.shuffle = shuffle
        self.batch_size =batch_size
        self.no_of_people = len(list(self.dataset.keys()))
        self.on_epoch_end()
        
    def __getitem__(self, index):
        people = list(self.dataset.keys())[index * self.batch_size: (index + 1) * self.batch_size]
        P = []
        A = []
        N = []
        
        for person in people:
            anchor_index = random.randint(0, len(self.dataset[person])-1)
            a = self.get_image(person, anchor_index)
            positive_index = random.randint(0, len(self.dataset[person])-1)
            while positive_index == anchor_index:
                positive_index = random.randint(0, len(self.dataset[person])-1)
            p = self.get_image(person, positive_index)
            
            negative_person_index = random.randint(0, self.no_of_people - 1)
            negative_person = list(self.dataset.keys())[negative_person_index]
            while negative_person == person:
                negative_person_index = random.randint(0, self.no_of_people - 1)
                negative_person = list(self.dataset.keys())[negative_person_index]
            
            negative_index = random.randint(0, len(self.dataset[negative_person])-1)
            n = self.get_image(negative_person, negative_index)
            
            P.append(p)
            A.append(a)
            N.append(n)
        A = np.asarray(A)
        N = np.asarray(N)
        P = np.asarray(P)
        return [A, P, N]
        
    def __len__(self):
        return self.no_of_people // self.batch_size
        
    def curate_dataset(self, dataset_path):
        with open(os.path.join(dataset_path, 'list.txt'), 'r') as f:
            dataset = {}
            image_list = f.read().split()
            for image in image_list:
                folder_name, file_name = image.split('/')
                if folder_name in dataset.keys():
                    dataset[folder_name].append(file_name)
                else:
                    dataset[folder_name] = [file_name]
        return dataset
    
    def on_epoch_end(self):
        if self.shuffle:
            keys = list(self.dataset.keys())
            random.shuffle(keys)
            dataset_ =  {}
            for key in keys:
                dataset_[key] = self.dataset[key]
            self.dataset = dataset_
            
    def get_image(self, person, index):
        # print(os.path.join(self.dataset_path, os.path.join('images/' + person, self.dataset[person][index])))
        img = cv2.imread(os.path.join(self.dataset_path, os.path.join('images/' + person, self.dataset[person][index])))
        img = cv2.resize(img, (224, 224))
        img = np.asarray(img, dtype=np.float64)
        img = preprocess_input(img)
        return img


vggface = VGG16(load_weights=True ,weights_path=weights_path)



#vggface.pop()
#vggface.add(tf.keras.layers.Dense(128, use_bias=False))
print(vggface.summary())
for layer in vggface.layers[:-1]:
    layer.trainable = False

model = SiameseNetwork(vggface)


data_generator = DataGenerator(dataset_path='.\\dataset\\')



model.compile(optimizer=optimizer)
checkpoint = tf.train.Checkpoint(model=model)
model.fit(data_generator[0], epochs=10)

checkpoint.save(checkpoint_path)
print("Checkpoint Saved")


