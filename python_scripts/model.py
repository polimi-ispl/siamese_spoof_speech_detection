import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.constraints import unit_norm
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import Sequence
import tensorflow.keras.backend as K
import logmelspectr_params as params
import random
import numpy as np

from kapre.composed import get_melspectrogram_layer


checkpoint_path='/nas/home/cborrelli/tripletloss_bot/checkpoints/vggish/vggish_model.ckpt'


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=-1, keepdims=True), K.epsilon()))


def get_base_model(input_shape):
    img_input = Input(shape=input_shape)

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1')(img_input)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)

    # Block fc
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1_1')(x)
    x = Dense(4096, activation='relu', name='fc1_2')(x)
    # TODO: decide if add the last dense layer
    #x = Dense(params.EMBEDDING_SIZE, activation='relu', name='fc2')(x)

    model = Model(img_input, x, name='vggish')


    return model


def get_head_model(embedding_shape):
    embedding_l = Input(embedding_shape, name='embed_ref')
    embedding_r = Input(embedding_shape, name='embed_dif')
    lambda_layer = Lambda(euclidean_distance)([embedding_l, embedding_r])
    model = Model([embedding_l, embedding_r], lambda_layer, name='distance')
    return model


def create_siamese_network(input_shape,
                           checkpoint_path='/nas/home/cborrelli/tripletloss_bot/checkpoints/vggish/vggish_model.ckpt'):
    """
    Create the siamese model structure using the supplied base and head model.
    """
    input_ref = Input(input_shape, name="input_ref")  # reference track
    input_dif = Input(input_shape, name="input_dif")  # different track

    base_model = get_base_model(input_shape)

    # Initialize base model with VGGish weights
    reader = tf.compat.v1.train.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()

    tensor_layers_list = []
    for key in var_to_shape_map:
        tensor_layers_list.append('/'.join(key.split('/')[:-1]))

    for index, t in enumerate(tensor_layers_list):
        weights_key = t + '/weights'
        bias_key = t + '/biases'
        weights = reader.get_tensor(weights_key)
        biases = reader.get_tensor(bias_key)

        keras_layer_name = t.split('/')[-1]
        # TODO: decide if initialize the last dense layer
        #if keras_layer_name == 'logits':
        if keras_layer_name == 'logits' or keras_layer_name == 'fc2':
            continue

        base_model.get_layer(keras_layer_name).set_weights([weights, biases])

    processed_ref = base_model(input_ref)
    processed_dif = base_model(input_dif)

    head_model = get_head_model(base_model.output_shape[-1])
    head = head_model([processed_ref, processed_dif])

    siamese_model = Model([input_ref, input_dif], head)
    return siamese_model


def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean((1 - y_true) * K.square(y_pred) + (y_true) * K.square(K.maximum(margin - y_pred, 0)))
