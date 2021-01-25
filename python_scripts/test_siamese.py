import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.constraints import unit_norm
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import Sequence
import tensorflow.keras.backend as K
from tqdm import tqdm
from .asvspoof_params import *
from .data_generator import *
from .model import *
import pandas as pd
import numpy as np
import itertools
import os

results_path = '/nas/home/cborrelli/tripletloss_bot/results'
model_path = '/nas/home/cborrelli/tripletloss_bot/checkpoints/siamese'

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
    df_train = pd.read_csv(train_txt_path, sep=" ", header=None)
    df_train.columns = ["speaker_id", "audio_filename", "null", "system_id", "label"]
    df_train = df_train.drop(columns="null")

    df_dev = pd.read_csv(dev_txt_path, sep=" ", header=None)
    df_dev.columns = ["speaker_id", "audio_filename", "null", "system_id", "label"]
    df_dev = df_dev.drop(columns="null")

    df_eval = pd.read_csv(eval_txt_path, sep=" ", header=None)
    df_eval.columns = ["speaker_id", "audio_filename", "null", "system_id", "label"]
    df_eval = df_eval.drop(columns="null")

    model = load_model(model_path,
                       custom_objects={'contrastive_loss': contrastive_loss,
                                       'euclidean_distance':euclidean_distance})

    dev_classes_list = ['-', 'A01', 'A02', 'A03', 'A04', 'A05', 'A06']
    eval_classes_list = ['-', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19']

    diff_couples_classes_dev = np.array(list(itertools.combinations(dev_classes_list, r=2)))
    same_couples_classes_dev = np.array(list(zip(dev_classes_list, dev_classes_list)))
    couples_classes_dev = np.concatenate((diff_couples_classes_dev, same_couples_classes_dev))

    diff_couples_classes_eval = np.array(list(itertools.combinations(eval_classes_list, r=2)))
    same_couples_classes_eval = np.array(list(zip(eval_classes_list, eval_classes_list)))
    couples_classes_eval = np.concatenate((diff_couples_classes_eval, same_couples_classes_eval))


    partial_results = []
    print('Compute results for dev')
    for c in tqdm(couples_classes_dev, total=len(couples_classes_dev)):
        test_generator = TestDataGenerator(dataframe=df_dev, feature_path=dev_feat_root, classes_pair=c)
        predicted_dist = model.predict(test_generator)
        rr = [[dist[0], c[0], c[1]] for dist in predicted_dist]
        partial_results.extend(rr)

    columns = ['distance', 'class_1', 'class_2']
    dev_results = pd.DataFrame(columns=columns, data=partial_results)
    result_dev_filename = os.path.join(results_path, 'dev_results.pkl')
    dev_results.to_pickle(result_dev_filename)

    partial_results = []
    print('Compute results for eval')
    for c in tqdm(couples_classes_eval, total=len(couples_classes_eval)):
        test_generator = TestDataGenerator(dataframe=df_eval, feature_path=eval_feat_root, classes_pair=c)
        predicted_dist = model.predict(test_generator)
        rr = [[dist[0], c[0], c[1]] for dist in predicted_dist]
        partial_results.extend(rr)

    eval_results = pd.DataFrame(columns=columns, data=partial_results)
    result_eval_filename = os.path.join(results_path, 'eval_results.pkl')
    eval_results.to_pickle(result_eval_filename)