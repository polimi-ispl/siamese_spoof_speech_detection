from data_generator import *
from model import *
import tensorflow as tf
import pandas as pd
import os
import numpy as np
import time

epochs = 1000
steps_per_epoch = 200

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


if __name__=='__main__':
    model = create_siamese_network(input_shape=(96, 64, 1))
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

    train_classes_list = ['-', 'A01', 'A02', 'A03', 'A04', 'A05', 'A06']
    #train_classes_list = ['-', 'A01', 'A02', 'A03']
    train_generator = TrainDataGenerator(dataframe=df_train, feature_path=train_feat_root,
                                         classes_list=train_classes_list, num_batch_epoch=steps_per_epoch)
    val_generator = TrainDataGenerator(dataframe=df_dev, feature_path=dev_feat_root, classes_list=train_classes_list,
                                       num_batch_epoch=steps_per_epoch)

    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2),
        tf.keras.callbacks.ModelCheckpoint(filepath='/nas/home/cborrelli/tripletloss_bot/checkpoints/siamese'),
    ]
    ## add callback
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss=contrastive_loss, optimizer=opt)

    history = model.fit(train_generator, validation_data=val_generator,
                        epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=my_callbacks)

    history_name = '/nas/home/cborrelli/tripletloss_bot/history/model_classes_{}_epochs_{}_steps_per_epoch_{}.npy'.format('_'.join(train_classes_list),
                                                                                                                          epochs,
                                                                                                                          steps_per_epoch)
    # Save history

    np.save(history_name, history.history)