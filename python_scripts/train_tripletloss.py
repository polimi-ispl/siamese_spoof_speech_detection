from data_generator import *
from model import *
import logmelspectr_params as params
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import time

train_classes_list = ['-', 'A01', 'A02', 'A03', 'A04', 'A05', 'A06']
train_classes_dictionary={'-': 0,
                    'A01': 1,
                    'A02': 2,
                    'A03': 3,
                    'A04': 4,
                    'A05': 5,
                    'A06': 6}
dev_classes_list = train_classes_list
dev_classes_dictionary = train_classes_dictionary

def load_features(df, feature_path, classes_dictionary, dim=(96, 64), n_channels=1):
    features_sample_rate = 1.0 / params.STFT_HOP_LENGTH_SECONDS

    example_window_length = int(round(
        params.EXAMPLE_WINDOW_SECONDS * features_sample_rate))
    example_hop_length = int(round(
        params.EXAMPLE_HOP_SECONDS * features_sample_rate))

    n_samples = len(df)
    X = np.empty((n_samples, *dim, n_channels))
    y = np.empty((n_samples), dtype=int)
    print("Loading features")
    for index, row in tqdm(df.iterrows(), total=n_samples):

        log_mel = np.load(os.path.join(feature_path, row['audio_filename'] + '.npy'))
        log_mel = log_mel.transpose()

        if log_mel.shape[0] < dim[0]:
            pad_len = dim[0] - log_mel.shape[0] + 1
            log_mel = np.pad(log_mel, ((0, pad_len), (0, 0)))

        log_mel = frame(log_mel, example_window_length, example_hop_length)

        selected_frame = np.random.randint(low=0, high=log_mel.shape[0], size=1)

        selected_log_mel = log_mel[selected_frame, :, :]
        selected_log_mel = selected_log_mel[0, :, :, np.newaxis]

        X[index, :] = selected_log_mel
        y[index] = classes_dictionary[row['system_id']]
    print('Features loaded')
    return X, y


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


if __name__=='__main__':
    model = get_vggish_tripletloss(input_shape=(96, 64, 1))
    model.summary()

    train_txt_path = "/nas/public/dataset/asvspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
    train_feat_root = "/nas/home/cborrelli/tripletloss_bot/features/logmelspectr/train"
    df_train = pd.read_csv(train_txt_path, sep=" ", header=None)
    df_train.columns = ["speaker_id", "audio_filename", "null", "system_id", "label"]
    df_train = df_train.drop(columns="null")

    dev_txt_path = "/nas/public/dataset/asvspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
    dev_feat_root = "/nas/home/cborrelli/tripletloss_bot/features/logmelspectr/dev"
    df_dev = pd.read_csv(dev_txt_path, sep=" ", header=None)
    df_dev.columns = ["speaker_id", "audio_filename", "null", "system_id", "label"]
    df_dev = df_dev.drop(columns="null")

    eval_txt_path = "/nas/public/dataset/asvspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
    eval_feat_root = "/nas/home/cborrelli/tripletloss_bot/features/logmelspectr/eval"
    df_eval = pd.read_csv(eval_txt_path, sep=" ", header=None)
    df_eval.columns = ["speaker_id", "audio_filename", "null", "system_id", "label"]
    df_eval = df_eval.drop(columns="null")

    X_train, y_train = load_features(df=df_train, feature_path=train_feat_root, classes_dictionary=train_classes_dictionary)
    X_dev, y_dev = load_features(df=df_dev, feature_path=dev_feat_root, classes_dictionary=dev_classes_dictionary)

    model_checkpoint_filename = '/nas/home/cborrelli/tripletloss_bot/checkpoints/tripletloss'
    my_callbacks = [
         tf.keras.callbacks.EarlyStopping(patience=2),
         tf.keras.callbacks.ModelCheckpoint(filepath=model_checkpoint_filename),
    ]
    ## add callback
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss=tfa.losses.TripletSemiHardLoss(margin=1.0), optimizer=opt)
    #
    history = model.fit(x=X_train, y=y_train, validation_data=(X_dev, y_dev), epochs=5, callbacks=my_callbacks)
    #
    #history_name = '/nas/home/cborrelli/tripletloss_bot/history/model_classes_{}_epochs_{}_steps_per_epoch_{}.npy'.format('_'.join(train_classes_list),
    #                                                                                                                       epochs,
    #                                                                                                                       steps_per_epoch)
    # # Save history
    #
    # np.save(history_name, history.history)