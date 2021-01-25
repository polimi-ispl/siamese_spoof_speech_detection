from data_generator import *
from model import *
import tensorflow as tf
import pandas as pd
import os
import numpy as np
import sklearn.model_selection

epochs = 40

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
    # Specify classification type
    classification_type = 'multiclass'
    traindev_eval = True

    train_batch_size = 128
    eval_batch_size = 1000

    if traindev_eval:

        train_classes_list = ['-', 'A01', 'A02', 'A03', 'A04', 'A05', 'A06']
        dev_classes_list = ['-', 'A01', 'A02', 'A03', 'A04', 'A05', 'A06']
        classes_list = train_classes_list

        if classification_type == 'multiclass':
            n_classes = len(train_classes_list)
        else:
            n_classes = 2

        model = get_vggish(input_shape=(96, 64, 1), out_dim=n_classes)
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

        train_generator = VGGishDataGenerator(dataframe=df_train, feature_path=train_feat_root, batch_size=train_batch_size,
                                          classification_type=classification_type,
                                          classes_list=train_classes_list)
        dev_generator = VGGishDataGenerator(dataframe=df_dev, feature_path=dev_feat_root, batch_size=eval_batch_size,
                                        classification_type=classification_type,
                                        classes_list=dev_classes_list, shuffle=False)
    else:
        eval_classes_list = ['-', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A13', 'A14',
                             'A15', 'A16', 'A17', 'A18', 'A19']
        classes_list = eval_classes_list

        if classification_type == 'multiclass':
            n_classes = len(eval_classes_list)
        else:
            n_classes = 2

        model = get_vggish(input_shape=(96, 64, 1), out_dim=n_classes)
        model.summary()

        eval_txt_path = "/nas/public/dataset/asvspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
        eval_feat_root = "/nas/home/cborrelli/tripletloss_bot/features/logmelspectr/eval"
        df_eval = pd.read_csv(eval_txt_path, sep=" ", header=None)
        df_eval.columns = ["speaker_id", "audio_filename", "null", "system_id", "label"]
        df_eval = df_eval.drop(columns="null")

        df_eval_train, df_eval_test = sklearn.model_selection.train_test_split(df_eval, test_size=0.2, random_state=2)
        df_eval_train.reset_index(inplace=True, drop=True)
        df_eval_test.reset_index(inplace=True, drop=True)

        train_generator = VGGishDataGenerator(dataframe=df_eval_train,
                                              feature_path=eval_feat_root,
                                              batch_size=train_batch_size,
                                              classes_list=eval_classes_list,
                                              classification_type=classification_type,
                                              traindev_eval=traindev_eval)
        dev_generator = VGGishDataGenerator(dataframe=df_eval_test,
                                            feature_path=eval_feat_root,
                                            batch_size=eval_batch_size,
                                            classes_list=eval_classes_list,
                                            classification_type=classification_type,
                                            traindev_eval=traindev_eval)


    checkpoint_path = '/nas/home/cborrelli/tripletloss_bot/checkpoints/myvggish/model_classification_{}_classes_{}'.format(
        classification_type, '_'.join(classes_list))
    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10),
        tf.keras.callbacks.ReduceLROnPlateau(patience=5),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss'),
    ]
    ## add callback
    opt = tf.keras.optimizers.Adam()
    if classification_type == 'binary':
        loss = tf.keras.losses.BinaryCrossentropy()
        metrics = [
            tf.keras.metrics.TruePositives(name='tp'),
            tf.keras.metrics.FalsePositives(name='fp'),
            tf.keras.metrics.TrueNegatives(name='tn'),
            tf.keras.metrics.FalseNegatives(name='fn'),
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            # tf.keras.metrics.Precision(name='precision'),
            # tf.keras.metrics.Recall(name='recall'),
            # tf.keras.metrics.AUC(name='auc'),
        ]
    if classification_type == 'multiclass':
        loss = tf.keras.losses.CategoricalCrossentropy()
        metrics = [
            tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
        ]

    model.compile(loss=loss, optimizer=opt, metrics=metrics, weighted_metrics=['accuracy'])
    history = model.fit(train_generator, validation_data=dev_generator,
                        epochs=epochs, callbacks=my_callbacks)
                        #class_weight=class_weight)
    history_name = '/nas/home/cborrelli/tripletloss_bot/history/vggish/model_classification_{}_classes_{}.npy'.format(
        classification_type, '_'.join(classes_list))
    # Save history
    np.save(history_name, history.history)