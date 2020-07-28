from BlindMIUtil import *
from dataLoader import *
import tensorflow as tf
from tensorflow.keras.models import load_model
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
DATA_NAME = sys.argv[1] if len(sys.argv) > 1 else "CIFAR"
TARGET_MODEL_GENRE = sys.argv[2] if len(sys.argv) > 2 else "ResNet50"
TARGET_WEIGHTS_PATH = "weights/Target/{}_{}.hdf5".format(DATA_NAME, TARGET_MODEL_GENRE)

(x_train_tar, y_train_tar), (x_test_tar, y_test_tar), \
m_true = globals()['load_' + DATA_NAME]('TargetModel')

Target_Model = load_model(TARGET_WEIGHTS_PATH)

def BlindMI_Diff_W(x_, y_true, m_true, target_model, non_Mem_Generator=sobel):
    ''' Attck the target with BLINDMI-DIFF-W, BLINDMI-DIFF with gernerated non-member.
    If the data has been shuffled, please directly remove the process of shuffling.
    :param target_model: the model that will be attacked
    :param x_: the data that target model may used for training
    :param y_true: the label of x_
    :param m_true: one of 0 and 1, which represents each of x_ has been trained or not.
    :param non_Mem_Generator: the method to generate the non-member data. The default non-member generator
    is Sobel.
    :return:  Tensor arrays of results
    '''

    #Prepare the data by probing the target model
    y_pred = target_model.predict(x_)
    mix = np.c_[y_pred[y_true.astype(bool)], np.sort(y_pred, axis=1)[:, ::-1][:, :2]]

    nonMem_pred = target_model.predict(non_Mem_Generator(x_))
    nonMem = np.c_[nonMem_pred[y_true.astype(bool)], np.sort(nonMem_pred, axis=1)[:, ::-1][:, :2]]
    data = tf.data.Dataset.from_tensor_slices((nonMem, mix, m_true)).shuffle(buffer_size=x_.shape[0])\
        .batch(1000).prefetch(tf.data.experimental.AUTOTUNE)

    #Using differential comparison to divide the data
    m_pred, m_true = [], []
    mix_shuffled = []
    nonMem_shuffled = []
    for (nonMem_batch, mix_batch, m_true_batch) in data:
        m_pred_batch = np.ones(nonMem_batch.shape[0])
        m_pred_epoch = np.ones(nonMem_batch.shape[0])
        nonMemInMix = True
        while nonMemInMix:
            nonMemInMix = False
            mix_epoch_new = mix_batch[m_pred_epoch.astype(bool)]
            dis_ori = mmd_loss(nonMem_batch, mix_epoch_new, weight=1)
            for index, item in tqdm(enumerate(mix_batch)):
                if m_pred_batch[index] == 1:
                    nonMem_batch_new = tf.concat([nonMem_batch, [mix_batch[index]]], axis=0)
                    mix_batch_new = tf.concat([mix_batch[:index], mix_batch[index+1:]], axis=0)
                    m_pred_without = np.r_[m_pred_batch[:index], m_pred_batch[index+1:]]
                    mix_batch_new = mix_batch_new[m_pred_without.astype(bool, copy=True)]
                    dis_new = mmd_loss(nonMem_batch_new, mix_batch_new, weight=1)
                    if dis_new >= dis_ori:
                        nonMemInMix = True
                        m_pred_epoch[index] = 0
            m_pred_batch = m_pred_epoch.copy()

        mix_shuffled.append(mix_batch)
        nonMem_shuffled.append(nonMem_batch)
        m_pred.append(m_pred_batch)
        m_true.append(m_true_batch)
    return np.concatenate(m_true, axis=0), np.concatenate(m_pred, axis=0), \
           np.concatenate(mix_shuffled, axis=0), np.concatenate(nonMem_shuffled, axis=0)


m_true, m_pred, mix, nonMem = BlindMI_Diff_W(np.r_[x_train_tar, x_test_tar],
                                             np.r_[y_train_tar, y_test_tar],
                                             m_true, Target_Model)

evaluate_attack(m_true, m_pred)
