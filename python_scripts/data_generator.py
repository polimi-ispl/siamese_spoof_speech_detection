import logmelspectr_params as params
from tensorflow.keras.utils import Sequence
import tensorflow as tf
import pandas as pd
import numpy as np
import itertools
import os
import soundfile as sf

def read_audio(audio_filename):
    audio, sr = sf.read(audio_filename, dtype='int16')
    assert audio.dtype == np.int16, 'Bad sample type: %r' % audio.dtype
    samples = audio / 32768.0  # Convert to [-1.0, +1.0]

    # Stereo to mono
    if len(samples.shape) > 1:
        samples = samples.mean(axis=1)

    return samples, sr

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