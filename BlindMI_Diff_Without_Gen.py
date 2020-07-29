from BlindMIUtil import *
from dataLoader import *
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.cluster import KMeans
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
DATA_NAME = sys.argv[1] if len(sys.argv) > 1 else "CIFAR"
TARGET_MODEL_GENRE = sys.argv[2] if len(sys.argv) > 2 else "ResNet50"
TARGET_WEIGHTS_PATH = "weights/Target/{}_{}.hdf5".format(DATA_NAME, TARGET_MODEL_GENRE)

(x_train_tar, y_train_tar), (x_test_tar, y_test_tar), m_true = globals()['load_' + DATA_NAME]('TargetModel')
Target_Model = load_model(TARGET_WEIGHTS_PATH)


def KMeans_Divide(mix):
    '''
    Using K-Means method to roughly divide the data into training set or not.
    :param mix: the data to be divided
    :return: the roughly results
    '''
    kmeans = KMeans(n_clusters=2).fit(mix)
    mix_1 = mix[kmeans.labels_.astype(bool)]
    mix_2 = mix[kmeans.labels_.astype(bool) == False]
    m_pred = kmeans.labels_ if np.mean(mix_1.numpy().max(axis=1))>\
                               np.mean(mix_2.numpy().max(axis=1)) else np.where(kmeans.labels_ == 1, 0, 1)
    return m_pred


def threshold_Divide(mix, ratio):
    '''
    Choose a threshold according to a percentile and use it to roughly divide the data into training set or not.
    :param mix: the data to be divided
    :param ratio: the ratio to roughly choose the threshold from the maximum value of confidence scores.
    :return: the roughly results
    '''
    threshold = np.percentile(mix.max(axis=1), ratio*100, interpolation='lower')
    m_pred = np.where(mix.max(axis=1)>threshold, 1, 0)
    return m_pred


def BlindMI_Diff_Single(x_, m_true, target_model):
    '''
    Attck the target with BLINDMI-DIFF-W/O, BLINDMI-DIFF without gernerated non-member.
    Roughly choose the non-member by threshold method.
    If the data has been shuffled, please directly remove the process of shuffling.
    :param target_model: the model that will be attacked
    :param x_: the data that target model may used for training
    :param m_true: one of 0 and 1, which represents each of x_ has been trained or not.
    :return:  Tensor arrays of results
    '''
    y_pred = target_model.predict(x_)
    mix = np.sort(y_pred, axis=1)[:, ::-1][:, :3]
    non_Mem = tf.convert_to_tensor(mix[threshold_Divide(mix, 1000/x_.shape[0]).astype(bool)==False])
    data = tf.data.Dataset.from_tensor_slices((mix, m_true)).shuffle(buffer_size=x_.shape[0]).batch(
        1000).prefetch(tf.data.experimental.AUTOTUNE)
    m_pred, m_true = [], []
    for (mix_batch, m_true_batch) in data:
        m_pred_batch = np.ones(mix_batch.shape[0])
        Flag = True
        while Flag:
            m_in_loop = m_pred_batch.copy()
            dis_ori = mmd_loss(non_Mem, mix_batch[m_in_loop.astype(bool)], weight=1)
            Flag = False
            for index, item in enumerate(mix_batch):
                if m_in_loop[index] == 1:
                    m_in_loop[index] = 0
                    mix_1 = mix_batch[m_in_loop.astype(bool)]
                    mix_2 = tf.concat([non_Mem, [item]], axis=0)
                    dis_new = mmd_loss(mix_2, mix_1, weight=1)
                    m_in_loop[index] = 1
                    #print("dis_new:{}, dis_ori:{}".format(dis_new, dis_ori))
                    if dis_new > dis_ori:
                        Flag = True
                        m_pred_batch[index] = 0

        m_pred.append(m_pred_batch)
        m_true.append(m_true_batch)
    return np.concatenate(m_true, axis=0), np.concatenate(m_pred, axis=0)


def BlindMI_Diff_Bi(x_, m_true, target_model):
    '''
    Attck the target with BLINDMI-DIFF-W/O, BLINDMI-DIFF without gernerated non-member.
    Roughly divide the data into member and non-member by threshold method.
    If the data has been shuffled, please directly remove the process of shuffling.
    :param target_model: the model that will be attacked
    :param x_: the data that target model may used for training
    :param m_true: one of 0 and 1, which represents each of x_ has been trained or not.
    :return:  Tensor arrays of results
    '''
    y_pred = target_model.predict(x_)
    mix = np.sort(y_pred, axis=1)[:, ::-1][:, :3]
    m_pred = threshold_Divide(mix, 0.5)
    data = tf.data.Dataset.from_tensor_slices((mix, m_true, m_pred)).shuffle(buffer_size=x_.shape[0]).batch(
        1000).prefetch(tf.data.experimental.AUTOTUNE)
    m_pred, m_true = [], []
    for (mix_batch, m_true_batch, m_pred_batch) in data:
        m_pred_batch = m_pred_batch.numpy()
        Flag = True
        while Flag:
            dis_ori = mmd_loss(mix_batch[m_pred_batch.astype(bool)==False], mix_batch[m_pred_batch.astype(bool)],
                               weight=1)
            Flag = False
            for index, item in tqdm(enumerate(mix_batch)):
                if m_pred_batch[index] == 0:
                    m_pred_batch[index] = 1
                    mix_1 = mix_batch[m_pred_batch.astype(bool)]
                    mix_2 = mix_batch[m_pred_batch.astype(bool)==False]
                    dis_new = mmd_loss(mix_2, mix_1, weight=1)
                    if dis_new < dis_ori:
                        m_pred_batch[index] = 0
                    else:
                        Flag = True
                        dis_ori = tf.identity(dis_new)
            for index, item in tqdm(enumerate(mix_batch)):
                if m_pred_batch[index] == 1:
                    m_pred_batch[index] = 0
                    mix_1 = mix_batch[m_pred_batch.astype(bool)]
                    mix_2 = mix_batch[m_pred_batch.astype(bool)==False]
                    dis_new = mmd_loss(mix_2, mix_1, weight=1)
                    if dis_new < dis_ori:
                        m_pred_batch[index] = 1
                    else:
                        Flag = True
                        dis_ori = tf.identity(dis_new)
        print("Loop finished")
        m_pred.append(m_pred_batch)
        m_true.append(m_true_batch)
    return np.concatenate(m_true, axis=0), np.concatenate(m_pred, axis=0)


m_pred, m_true = BlindMI_Diff_Threshold(np.r_[x_train_tar, x_test_tar], m_true, Target_Model)

evaluate_attack(m_true, m_pred)