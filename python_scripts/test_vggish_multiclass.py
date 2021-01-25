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
    # Training parameters
    classification_type = 'multiclass'
    # In binary case this affects only the name
    traindev_eval = True

    train_classes_list = ['-', 'A01', 'A02', 'A03', 'A04', 'A05', 'A06']
    dev_classes_list = ['-', 'A01', 'A02', 'A03', 'A04', 'A05', 'A06']
    eval_classes_list = ['-', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A13', 'A14',
                         'A15', 'A16', 'A17', 'A18', 'A19']

    binary_dict = {'-': 0, 'A01': 1, 'A02': 1, 'A03': 1, 'A04': 1, 'A05': 1, 'A06': 1, 'A07': 1, 'A08': 1, 'A09': 1,
                   'A10': 1, 'A11': 1, 'A12': 1, 'A13': 1, 'A14': 1, 'A15': 1, 'A16': 1, 'A17': 1, 'A18': 1, 'A19': 1}

    multiclass_dict = {'-': 0, 'A01':1, 'A02':2, 'A03':3, 'A04':4, 'A05':5, 'A06':6, 'A07': 1, 'A08': 2, 'A09': 3, 'A10': 4, 'A11': 5, 'A12': 6,
                       'A13': 7, 'A14': 8, 'A15': 9, 'A16': 10, 'A17': 11, 'A18': 12, 'A19': 13}
    if traindev_eval:
        classes_list = train_classes_list
    else:
        classes_list = eval_classes_list

    # Plot histories
    history_root = "/nas/home/cborrelli/tripletloss_bot/history/vggish"
    history_filename = '/nas/home/cborrelli/tripletloss_bot/history/vggish/model_classification_{}_classes_{}.npy'.format(
        classification_type, '_'.join(classes_list))
    history = np.load(os.path.join(history_root, history_filename), allow_pickle=True)
    history = history.item()

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
    classification_type, '_'.join(classes_list))

    model = load_model(model_path)

    partial_results = []

    for c in tqdm.tqdm(classes_list, total=len(classes_list)):
        if traindev_eval :
            train_generator = VGGishDataGenerator(dataframe=df_train,
                                                  feature_path=train_feat_root,
                                                  shuffle=False,
                                                  batch_size=batch_size,
                                                  classes_list=[c],
                                                  traindev_eval=traindev_eval)
            test_generator = VGGishDataGenerator(dataframe=df_dev,
                                                 feature_path=dev_feat_root,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 classes_list=[c],
                                                 traindev_eval=traindev_eval)
            dataset_train = 'train'
            dataset_test = 'dev'
        else:
            df_eval_train, df_eval_test = sklearn.model_selection.train_test_split(df_eval, test_size=0.2,
                                                                                   random_state=2)
            df_eval_train.reset_index(inplace=True, drop=True)
            df_eval_test.reset_index(inplace=True, drop=True)

            train_generator = VGGishDataGenerator(dataframe=df_eval_train,
                                                  feature_path=eval_feat_root,
                                                  batch_size=batch_size,
                                                  classes_list=[c],
                                                  classification_type=classification_type,
                                                  traindev_eval=traindev_eval,
                                                  shuffle=False)
            test_generator = VGGishDataGenerator(dataframe=df_eval_test,
                                                feature_path=eval_feat_root,
                                                batch_size=batch_size,
                                                classes_list=[c],
                                                classification_type=classification_type,
                                                traindev_eval=traindev_eval,
                                                 shuffle=False)
            dataset_train = 'eval'
            dataset_test = 'eval'

        train_predicted = model.predict(train_generator)
        test_predicted = model.predict(test_generator)

        train_predicted = np.argmax(train_predicted, axis=1)
        test_predicted = np.argmax(test_predicted, axis=1)

        train_rr = [[multiclass_dict[c], p, dataset_train] for p in train_predicted]
        dev_rr = [[multiclass_dict[c], p, dataset_test ] for p in test_predicted]

        partial_results.extend(train_rr)
        partial_results.extend(dev_rr)


    columns = ['class_true', 'class_pred', 'dataset']

    results = pd.DataFrame(columns=columns, data=partial_results)
    results.to_pickle(os.path.join(results_path, 'results_classification_{}_classes_{}'.format(classification_type,
                                                                                               '_'.join(classes_list)) ))


