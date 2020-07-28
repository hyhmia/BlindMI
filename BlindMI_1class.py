from BlindMIUtil import *
from dataLoader import *
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn import svm
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
DATA_NAME = sys.argv[1] if len(sys.argv) > 1 else "CIFAR"
TARGET_MODEL_GENRE = sys.argv[2] if len(sys.argv) > 2 else "ResNet50"
TARGET_WEIGHTS_PATH = "weights/Target/{}_{}.hdf5".format(DATA_NAME, TARGET_MODEL_GENRE)


def BlindMI_1class(x_, y_true, target_model):
    '''
    One-class SVM version with generated non-member as training set and predict whether the data has
    been trained or not.If the data has been shuffled, please directly remove the process of shuffling.
    :param x_:The data to be classified as trained or untrained
    :param y_true: The label of data
    :param target_model: The MI Model to probe
    :return: the predicted results
    '''
    y_pred = Target_Model.predict(np.r_[x_train_tar, x_test_tar])
    mix = np.sort(y_pred, axis=1)[:, ::-1][:, :3]

    nonMem_pred = target_model.predict(sobel(x_))
    nonMem = np.sort(nonMem_pred, axis=1)[:, ::-1][:, :3]

    cls = svm.OneClassSVM(nu=0.9, kernel='sigmoid', gamma='scale')
    cls.fit(nonMem)
    m_pred = [i if i == 1 else 0 for i in cls.predict(mix)]
    return m_pred


(x_train_tar, y_train_tar), (x_test_tar, y_test_tar), m_true = globals()['load_' + DATA_NAME]('TargetModel')
Target_Model = load_model(TARGET_WEIGHTS_PATH)

m_pred = BlindMI_1class(np.r_[x_train_tar, x_test_tar], np.r_[y_train_tar, y_test_tar], Target_Model)
evaluate_attack(m_true, m_pred)