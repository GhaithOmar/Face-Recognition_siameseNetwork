import numpy as np 
import tensorflow as tf 


from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Input, Flatten,Activation ,BatchNormalization
from keras.models import Model, Sequential


def preprocess_input(x, data_format=None, version=1):
    x_temp = np.copy(x)
    if data_format is None:
        data_format = tf.keras.backend.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if version == 1:
        if data_format == 'channels_first':
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= 93.5940
            x_temp[:, 1, :, :] -= 104.7624
            x_temp[:, 2, :, :] -= 129.1863
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= 93.5940
            x_temp[..., 1] -= 104.7624
            x_temp[..., 2] -= 129.1863

    elif version == 2:
        if data_format == 'channels_first':
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= 91.4953
            x_temp[:, 1, :, :] -= 103.8827
            x_temp[:, 2, :, :] -= 131.0912
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= 91.4953
            x_temp[..., 1] -= 103.8827
            x_temp[..., 2] -= 131.0912
    else:
        raise NotImplementedError

    return x_temp

def VGG16(inputs=None):
	vggface = tf.keras.models.Sequential()
	vggface.add(tf.keras.layers.Convolution2D(64, (3, 3), activation='relu', padding="SAME", input_shape=(224,224, 3)))
	vggface.add(tf.keras.layers.Convolution2D(64, (3, 3), activation='relu', padding="SAME"))
	vggface.add(tf.keras.layers.MaxPooling2D((2,2), strides=(2,2)))
	 
	vggface.add(tf.keras.layers.Convolution2D(128, (3, 3), activation='relu', padding="SAME"))
	vggface.add(tf.keras.layers.Convolution2D(128, (3, 3), activation='relu', padding="SAME"))
	vggface.add(tf.keras.layers.MaxPooling2D((2,2), strides=(2,2)))
	 
	vggface.add(tf.keras.layers.Convolution2D(256, (3, 3), activation='relu', padding="SAME"))
	vggface.add(tf.keras.layers.Convolution2D(256, (3, 3), activation='relu', padding="SAME"))
	vggface.add(tf.keras.layers.Convolution2D(256, (3, 3), activation='relu', padding="SAME"))
	vggface.add(tf.keras.layers.MaxPooling2D((2,2), strides=(2,2)))
	 
	vggface.add(tf.keras.layers.Convolution2D(512, (3, 3), activation='relu', padding="SAME"))
	vggface.add(tf.keras.layers.Convolution2D(512, (3, 3), activation='relu', padding="SAME"))
	vggface.add(tf.keras.layers.Convolution2D(512, (3, 3), activation='relu', padding="SAME"))
	vggface.add(tf.keras.layers.MaxPooling2D((2,2), strides=(2,2)))
	 
	vggface.add(tf.keras.layers.Convolution2D(512, (3, 3), activation='relu', padding="SAME"))
	vggface.add(tf.keras.layers.Convolution2D(512, (3, 3), activation='relu', padding="SAME"))
	vggface.add(tf.keras.layers.Convolution2D(512, (3, 3), activation='relu', padding="SAME"))
	vggface.add(tf.keras.layers.MaxPooling2D((2,2), strides=(2,2)))

	vggface.add(tf.keras.layers.Flatten())

	vggface.add(tf.keras.layers.Dense(4096, activation='relu'))
	vggface.add(tf.keras.layers.Dense(4096, activation='relu'))
	vggface.add(tf.keras.layers.Dense(2622, activation='softmax'))
	vggface.load_weights(r'.\weights\rcmalli_vggface_tf_vgg16.h5')
	vggface.pop()




	return vggface