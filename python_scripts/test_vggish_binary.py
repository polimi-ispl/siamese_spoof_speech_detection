import tensorflow as tf
from tensorflow.keras.models import *
import matplotlib.pyplot as plt
from asvspoof_params import *
from data_generator import *
from model import *
import pandas as pd
import sklearn.model_selection
import numpy as np
import tqdm
import itertools
import os

batch_size = 5

results_path = '/nas/home/cborrelli/tripletloss_bot/results'

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
print(gpus)

if __name__=='__main__':
    train_classes_list = ['-', 'A01', 'A02', 'A03', 'A04', 'A05', 'A06']
    dev_classes_list = ['-', 'A01', 'A02', 'A03', 'A04', 'A05', 'A06']
    eval_classes_list = ['-', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A13', 'A14',
                         'A15', 'A16', 'A17', 'A18', 'A19']

    binary_dict = {'-': 0, 'A01': 1, 'A02': 1, 'A03': 1, 'A04': 1, 'A05': 1, 'A06': 1, 'A07': 1, 'A08': 1, 'A09': 1,
                   'A10': 1, 'A11': 1, 'A12': 1, 'A13': 1, 'A14': 1, 'A15': 1, 'A16': 1, 'A17': 1, 'A18': 1, 'A19': 1}

    # Plot histories
    # history_root = "/nas/home/cborrelli/tripletloss_bot/history/vggish"
    # history_filename = '/nas/home/cborrelli/tripletloss_bot/history/vggish/model_classification_{}_classes_{}.npy'.format(
    #     classification_type, '_'.join(classes_list))
    # history = np.load(os.path.join(history_root, history_filename), allow_pickle=True)
    # history = history.item()
    #
    # plt.figure(figsize=(15, 10))
    # plt.grid(True)
    # plt.plot(history['loss'])
    # plt.plot(history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper right')
    # plt.show()
    #
    # plt.figure(figsize=(15, 10))
    # plt.grid(True)
    # plt.plot(history['weighted_accuracy'])
    # plt.plot(history['val_weighted_accuracy'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper right')
    # plt.show()

    # Compute metrics
    df_train = pd.read_csv(train_txt_path, sep=" ", header=None)
    df_train.columns = ["speaker_id", "audio_filename", "null", "system_id", "label"]
    df_train = df_train.drop(columns="null")

    df_dev = pd.read_csv(dev_txt_path, sep=" ", header=None)
    df_dev.columns = ["speaker_id", "audio_filename", "null", "system_id", "label"]
    df_dev = df_dev.drop(columns="null")

    df_eval = pd.read_csv(eval_txt_path, sep=" ", header=None)
    df_eval.columns = ["speaker_id", "audio_filename", "null", "system_id", "label"]
    df_eval = df_eval.drop(columns="null")

    model_path = '/nas/home/cborrelli/tripletloss_bot/checkpoints/myvggish/model_classification_{}_classes_{}'.format(
    'binary', '_'.join(train_classes_list))

    #model_path = '/nas/home/cborrelli/tripletloss_bot/checkpoints/myvggish/model_classification_{}_classes_{}'.format(
    #'binary', '_'.join(eval_classes_list))

    model = load_model(model_path)

    partial_results = []

    for c in tqdm.tqdm(train_classes_list, total=len(train_classes_list)):

        train_generator = VGGishDataGenerator(dataframe=df_train, feature_path=train_feat_root,
                                              shuffle=False,
                                              batch_size=batch_size, classes_list=[c])
        dev_generator = VGGishDataGenerator(dataframe=df_dev, feature_path=dev_feat_root,
                                        batch_size=batch_size, shuffle=False, classes_list=[c])

        train_predicted = model.predict(train_generator)
        dev_predicted = model.predict(dev_generator)

        train_predicted = np.argmax(train_predicted, axis=1)
        test_predicted = np.argmax(dev_predicted, axis=1)

        train_rr = [[binary_dict[c], p, c, 'train'] for p in train_predicted]
        dev_rr = [[binary_dict[c], p, c, 'dev'] for p in dev_predicted]

        partial_results.extend(train_rr)
        partial_results.extend(dev_rr)


    for c in tqdm.tqdm(eval_classes_list, total=len(eval_classes_list)):
        eval_generator = VGGishDataGenerator(dataframe=df_eval, feature_path=eval_feat_root,
                                             batch_size=batch_size, shuffle=False, classes_list=[c])
        eval_predicted = model.predict(eval_generator)
        eval_predicted = np.argmax(eval_predicted, axis=1)
        eval_rr = [[binary_dict[c], p, c, 'eval'] for p in eval_predicted]
        partial_results.extend(eval_rr)


    columns = ['label_true', 'label_pred', 'class', 'dataset']

    results = pd.DataFrame(columns=columns, data=partial_results)

    results.to_pickle(os.path.join(results_path, 'results_classification_{}_classes_{}'.format('binary',
                                                                                               '_'.join(train_classes_list)) ))


