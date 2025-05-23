import numpy as np 
import tensorflow as tf 


from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Input, Flatten,Activation ,BatchNormalization
from keras.models import Model


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

def VGG16(inputs=None,load_weights=False ,weights_path=None):
	img_input = Input(shape=(224,224, 3))



	# Block 1
	x = Conv2D(64, (3,3), activation='relu', padding='same', name='conv1_1')(img_input)
	x = Conv2D(64, (3,3), activation='relu', padding='same', name='conv1_2')(x)
	x = MaxPooling2D((2,2), strides=(2,2), name='pool1')(x)

	# Block 2
	x = Conv2D(128, (3,3), activation='relu', padding='same', name='conv2_1')(x)
	x = Conv2D(128, (3,3), activation='relu', padding='same', name='conv2_2')(x)
	x = MaxPooling2D((2,2), strides=(2,2), name='pool2')(x)

	# Block 3 
	x = Conv2D(256, (3,3), activation='relu', padding='same', name='conv3_1')(x)
	x = Conv2D(256, (3,3), activation='relu', padding='same', name='conv3_2')(x)
	x = Conv2D(256, (3,3), activation='relu', padding='same', name='conv3_3')(x)
	x = MaxPooling2D((2,2), strides=(2,2), name='pool3')(x)

	# Block 4 
	x = Conv2D(512, (3,3), activation='relu', padding='same', name='conv4_1')(x)
	x = Conv2D(512, (3,3), activation='relu', padding='same', name='conv4_2')(x)
	x = Conv2D(512, (3,3), activation='relu', padding='same', name='conv4_3')(x)
	x = MaxPooling2D((2,2), strides=(2,2), name='pool4')(x)

	# Block 5 
	x = Conv2D(512, (3,3), activation='relu', padding='same', name='conv5_1')(x)
	x = Conv2D(512, (3,3), activation='relu', padding='same', name='conv5_2')(x)
	x = Conv2D(512, (3,3), activation='relu', padding='same', name='conv5_3')(x)
	x = MaxPooling2D((2,2), strides=(2,2), name='pool5')(x)

	# include top
	x = Flatten(name='flatten')(x)
	x = Dense(4096, name='fc6')(x)
	x = Activation('relu', name='fc6/relu')(x)
	x = Dense(4096, name='fc7')(x)
	x = Activation('relu', name='fc7/relu')(x)
	output = Dense(128, use_bias=False)(x)
	inputs = img_input
	model = Model(inputs, output , name='vggface_vgg16')
	if load_weights and weights_path:
		model.load_weights(weights_path, by_name=True)

	return model