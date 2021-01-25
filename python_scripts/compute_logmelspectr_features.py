from scipy import signal
import numpy as np
import soundfile as sf
import os
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import argparse
import resampy
import librosa
import logmelspectr_params as params
import matplotlib.pyplot as plt


default_logmelspectr_dest_folder = "/nas/home/cborrelli/tripletloss_bot/features/logmelspectr"
default_stft_dest_folder = "/nas/home/cborrelli/tripletloss_bot/features/stft"


def read_audio(audio_filename):
    audio, sr = sf.read(audio_filename, dtype='int16')
    assert audio.dtype == np.int16, 'Bad sample type: %r' % audio.dtype
    samples = audio / 32768.0  # Convert to [-1.0, +1.0]

    # Stereo to mono
    if len(samples.shape) > 1:
        samples = samples.mean(axis=1)

    # Resample to the rate assumed by VGGish.
    if sr != params.SAMPLE_RATE:
        samples = resampy.resample(samples, sr, params.SAMPLE_RATE)
        sr = params.SAMPLE_RATE

    return samples, sr

def frame_feature(data, window_length, hop_length):
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



def compute_spectrogram(arg, logmelspectr_dest_root, stft_dest_root, audio_folder):
    """ Compute the spectrogram of a signal
    using the function scipy.signal.spectrogram
    """
    # Read audio
    index = arg[0]
    row = arg[1]
    audio_filename = row["audio_filename"] + '.flac'
    audio, rate = read_audio(audio_filename=os.path.join(audio_folder, audio_filename))

    audio_sample_rate = params.SAMPLE_RATE
    log_offset = params.LOG_OFFSET
    window_length_secs = params.STFT_WINDOW_LENGTH_SECONDS
    hop_length_secs = params.STFT_HOP_LENGTH_SECONDS
    num_mel_bins = params.NUM_MEL_BINS
    lower_edge_hertz = params.MEL_MIN_HZ
    upper_edge_hertz = params.MEL_MAX_HZ

    window_length_samples = int(round(audio_sample_rate * window_length_secs))
    hop_length_samples = int(round(audio_sample_rate * hop_length_secs))
    fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))
    X = librosa.stft(
        audio,
        window='hann',
        n_fft=fft_length,
        hop_length=hop_length_samples,
        win_length=window_length_samples)

    S = np.abs(X)
    mel_f = librosa.filters.mel(
        sr=audio_sample_rate,
        n_fft=fft_length,
        n_mels=num_mel_bins,
        fmin=lower_edge_hertz,
        fmax=upper_edge_hertz,
        htk=True,
        norm=None
    )
    # per uniformare con la parte in data generator salvo la versione trasposta N_bins x N_sample
    S_mel = np.dot(S.T, mel_f.T).T
    S_log_mel = np.log(S_mel + log_offset)

    logmelspectr_out_name = os.path.join(logmelspectr_dest_root, row['audio_filename']+'.npy')
    stft_out_name = os.path.join(stft_dest_root, row['audio_filename']+'.npy')

    np.save(logmelspectr_out_name, S_log_mel)
    np.save(stft_out_name, X)
    return


def compute_features(audio_folder, txt_path, logmelspectr_dest_root, stft_dest_root, data_subset):
    # Open dataset df
    df = pd.read_csv(txt_path, sep=" ", header=None)
    df.columns = ["speaker_id", "audio_filename", "null", "system_id", "label"]
    df = df.drop(columns="null")

    # Prepare parallel execution
    args_list = list(df.iterrows())
    dest_subset_folder = '{}'.format(data_subset)
    logmelspectr_dest_subset_root = os.path.join(logmelspectr_dest_root, dest_subset_folder)
    stft_dest_subset_root = os.path.join(stft_dest_root, dest_subset_folder)

    if not os.path.exists(logmelspectr_dest_subset_root):
        os.makedirs(logmelspectr_dest_subset_root)
    if not os.path.exists(stft_dest_subset_root):
        os.makedirs(stft_dest_subset_root)
    print("Save in {}".format(logmelspectr_dest_subset_root))


    compute_features_partial = partial(compute_spectrogram, logmelspectr_dest_root=logmelspectr_dest_subset_root,
                        stft_dest_root=stft_dest_subset_root,
                        audio_folder=audio_folder)
    # Run parallel execution
    pool = Pool(cpu_count() // 2)
    _ = list(tqdm(pool.imap(compute_features_partial, args_list), total=len(args_list)))
    return


if __name__ == '__main__':
    # parse input arguments
    os.nice(2)
    parser = argparse.ArgumentParser()
    parser.add_argument('--logmelspectr_dest_folder', type=str, required=False, default=default_logmelspectr_dest_folder)
    parser.add_argument('--stft_dest_folder', type=str, required=False, default=default_stft_dest_folder)
    parser.add_argument('--data_subset', type=str, required=True)


    args = parser.parse_args()
    logmelspectr_dest_folder = args.logmelspectr_dest_folder
    stft_dest_folder = args.stft_dest_folder
    data_subset = args.data_subset

    audio_folder = '/nas/public/dataset/asvspoof2019/LA/ASVspoof2019_LA_{}/flac'.format(data_subset)

    if data_subset != 'train':
        txt_path = '/nas/public/dataset/asvspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{}.trl.txt'.format(
            data_subset)
    else:
        txt_path = '/nas/public/dataset/asvspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{}.trn.txt'.format(
            data_subset)


    compute_features(audio_folder, txt_path, logmelspectr_dest_folder,  stft_dest_folder, data_subset)
