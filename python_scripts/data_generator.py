import logmelspectr_params as params
from tensorflow.keras.utils import Sequence
import tensorflow as tf
import pandas as pd
import numpy as np
import itertools
import os

def frame(data, window_length, hop_length):
    """Convert array into a sequence of successive possibly overlapping frames.
    An n-dimensional array of shape (num_samples, ...) is converted into an
    (n+1)-D array of shape (num_frames, window_length, ...), where each frame
    starts hop_length points after the preceding one.
    This is accomplished using stride_tricks, so the original data is not
    copied.  However, there is no zero-padding, so any incomplete frames at the
    end are not included.
    Args:
    data: np.array of dimension N >= 1.
    window_length: Number of samples in each frame.
    hop_length: Advance (in samples) between each window.
    Returns:
    (N+1)-D np.array with as many rows as there are complete frames that can be
    extracted.
    """
    num_samples = data.shape[0]
    num_frames = 1 + int(np.floor((num_samples - window_length) / hop_length))
    shape = (num_frames, window_length) + data.shape[1:]
    strides = (data.strides[0] * hop_length,) + data.strides
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


class TrainDataGenerator(Sequence):
    'Generates data for Keras'

    def __init__(self, dataframe, feature_path, batch_size=32, dim=(96, 64), n_channels=1,
                 shuffle=True, classes_list=['-', 'A01', 'A02', 'A03', 'A04', 'A05', 'A06'],
                 num_batch_epoch=100):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.dataframe = dataframe
        self.classes_list = classes_list
        self.n_channels = n_channels
        self.len = num_batch_epoch
        self.feature_path = feature_path

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.len

    def __getitem__(self, batch_index):
        'Generate one batch of data'
        negative_couples_classes = np.array(list(itertools.combinations(self.classes_list, r=2)))
        positive_couples_classes = np.array(list(zip(self.classes_list, self.classes_list)))

        negative_selected_pairs = negative_couples_classes[np.random.choice(negative_couples_classes.shape[0],
                                                                            self.batch_size // 2, replace=True), :]
        positive_selected_pairs = positive_couples_classes[np.random.choice(positive_couples_classes.shape[0],
                                                                            self.batch_size // 2, replace=True), :]

        selected_pairs = np.concatenate((positive_selected_pairs, negative_selected_pairs), axis=0)

        y = np.concatenate((np.zeros((self.batch_size // 2)), np.ones((self.batch_size // 2))), axis=0)

        features_sample_rate = 1.0 / params.STFT_HOP_LENGTH_SECONDS

        example_window_length = int(round(
            params.EXAMPLE_WINDOW_SECONDS * features_sample_rate))
        example_hop_length = int(round(
            params.EXAMPLE_HOP_SECONDS * features_sample_rate))

        X_0 = np.empty((self.batch_size, *self.dim, self.n_channels))
        X_1 = np.empty((self.batch_size, *self.dim, self.n_channels))

        for sample_batch_index, pairs in enumerate(selected_pairs):

            sample = np.empty((2, *self.dim, self.n_channels))
            for a, alg in enumerate(pairs):
                row = self.dataframe[self.dataframe.system_id == alg].sample(n=1)
                log_mel = np.load(os.path.join(self.feature_path, row['audio_filename'].values[0] + '.npy'))
                log_mel = log_mel.transpose()

                if log_mel.shape[0] < self.dim[0]:
                    pad_len = self.dim[0] - log_mel.shape[0] + 1
                    log_mel = np.pad(log_mel, ((0, pad_len), (0, 0)))

                log_mel = frame(log_mel, example_window_length, example_hop_length)

                selected_frame = np.random.randint(low=0, high=log_mel.shape[0], size=1)

                selected_log_mel = log_mel[selected_frame, :, :]
                selected_log_mel = selected_log_mel[0, :, :, np.newaxis]

                sample[a] = selected_log_mel

            X_0[sample_batch_index] = sample[0]
            X_1[sample_batch_index] = sample[1]

        return [X_0, X_1], y


class TestDataGenerator(Sequence):
    'Generates data for Keras'

    def __init__(self, dataframe, feature_path, batch_size=32, dim=(96, 64), n_channels=1,
                 shuffle=True, classes_pair=['-', '-'],
                 num_batch_epoch=100):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.dataframe = dataframe
        self.n_channels = n_channels
        self.len = num_batch_epoch
        self.feature_path = feature_path
        self.classes_pair = classes_pair

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.len

    def __getitem__(self, batch_index):
        'Generate one batch of data'
        # If i am specifying only one element it means I want to use the data generator for testing
        # only one class
        selected_pairs = [self.classes_pair] * self.batch_size
        features_sample_rate = 1.0 / params.STFT_HOP_LENGTH_SECONDS
        example_window_length = int(round(
            params.EXAMPLE_WINDOW_SECONDS * features_sample_rate))
        example_hop_length = int(round(
            params.EXAMPLE_HOP_SECONDS * features_sample_rate))

        X_0 = np.empty((self.batch_size, *self.dim, self.n_channels))
        X_1 = np.empty((self.batch_size, *self.dim, self.n_channels))

        if self.classes_pair[0] == self.classes_pair[1]:
            y = np.zeros((self.batch_size))
        else:
            y = np.ones((self.batch_size))

        for sample_batch_index, pairs in enumerate(selected_pairs):

            sample = np.empty((2, *self.dim, self.n_channels))
            for a, alg in enumerate(pairs):
                row = self.dataframe[self.dataframe.system_id == alg].sample(n=1)
                log_mel = np.load(os.path.join(self.feature_path, row['audio_filename'].values[0] + '.npy'))
                log_mel = log_mel.transpose()

                if log_mel.shape[0] < self.dim[0]:
                    pad_len = self.dim[0] - log_mel.shape[0] + 1
                    log_mel = np.pad(log_mel, ((0, pad_len), (0, 0)))

                log_mel = frame(log_mel, example_window_length, example_hop_length)

                selected_frame = np.random.randint(low=0, high=log_mel.shape[0], size=1)

                selected_log_mel = log_mel[selected_frame, :, :]
                selected_log_mel = selected_log_mel[0, :, :, np.newaxis]

                sample[a] = selected_log_mel

            X_0[sample_batch_index] = sample[0]
            X_1[sample_batch_index] = sample[1]

        return [X_0, X_1], y


class VGGishDataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, dataframe, feature_path, batch_size=32, dim=(96, 64), n_channels=1,
                 shuffle=True, classification_type='binary',
                 classes_list=('-', 'A01', 'A02', 'A03', 'A04', 'A05', 'A06'),
                 traindev_eval=True):
        '''Initialization
        - use classes list for specifing the selected classes for training and testing
        '''
        self.dim = dim
        self.batch_size = batch_size
        self.dataframe = dataframe
        self.shuffle = shuffle
        self.traindev_eval = traindev_eval
        if self.shuffle:
            self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)
        self.classes_list = classes_list
        self.dataframe = self.dataframe[(dataframe['system_id'].isin(self.classes_list))]
        self.n_channels = n_channels
        self.feature_path = feature_path
        self.classification_type = classification_type
        if self.traindev_eval:
            self.multiclass_dict = {'-' : 0.0, 'A01' : 1.0, 'A02' : 2.0, 'A03' : 3.0, 'A04' : 4.0, 'A05' : 5.0, 'A06' : 6.0}
        else:
            self.multiclass_dict = {'-': 0.0, 'A07': 1.0, 'A08': 2.0, 'A09': 3.0, 'A10': 4.0,
                                    'A11': 5.0, 'A12': 6.0,'A13': 7.0, 'A14': 8.0, 'A15': 9.0,
                                    'A16': 10.0, 'A17': 11.0, 'A18': 12.0, 'A19': 13.0}

        self.binary_dict = {'bonafide' : 0.0, 'spoof' : 1.0}

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.dataframe) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            self.dataframe = self.dataframe.sample(frac=1, axis=1).reset_index(drop=True)

    def __getitem__(self, batch_index):
        'Generate one batch of data'
        start_index = batch_index * self.batch_size
        selected_rows = self.dataframe.iloc[start_index : start_index + self.batch_size].copy()
        selected_rows = selected_rows.reset_index()
        features_sample_rate = 1.0 / params.STFT_HOP_LENGTH_SECONDS
        example_window_length = int(round(
            params.EXAMPLE_WINDOW_SECONDS * features_sample_rate))
        example_hop_length = int(round(
            params.EXAMPLE_HOP_SECONDS * features_sample_rate))

        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        if self.classification_type == 'binary':
            y = selected_rows['label'].values
            y = np.array([self.binary_dict[a] for a in y], dtype=np.float)
            y = tf.keras.utils.to_categorical(y, num_classes=2)
        if self.classification_type == 'multiclass':
            y = selected_rows['system_id'].values
            y = np.array([self.multiclass_dict[a] for a in y], dtype=np.float)
            y = tf.keras.utils.to_categorical(y, num_classes=len(self.multiclass_dict))

        for index, row in selected_rows.iterrows():
            log_mel = np.load(os.path.join(self.feature_path, row['audio_filename'] + '.npy'))
            log_mel = log_mel.transpose()

            if log_mel.shape[0] < self.dim[0]:
                pad_len = self.dim[0] - log_mel.shape[0] + 1
                log_mel = np.pad(log_mel, ((0, pad_len), (0, 0)))

            log_mel = frame(log_mel, example_window_length, example_hop_length)

            selected_frame = np.random.randint(low=0, high=log_mel.shape[0], size=1)

            selected_log_mel = log_mel[selected_frame, :, :]
            selected_log_mel = selected_log_mel[0, :, :, np.newaxis]

            X[index] = selected_log_mel

        return X, y