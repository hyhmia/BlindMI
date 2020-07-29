from dataLoader import *
import tensorflow as tf
from tensorflow.keras.models import load_model
from BlindMIUtil import evaluate_attack
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
DATA_NAME = sys.argv[1] if len(sys.argv) > 1 else "CIFAR"
TARGET_MODEL_GENRE = sys.argv[2] if len(sys.argv) > 2 else "ResNet50"
TARGET_WEIGHTS_PATH = "weights/Target/{}_{}.hdf5".format(DATA_NAME, TARGET_MODEL_GENRE)

(x_train_tar, y_train_tar), (x_test_tar, y_test_tar), m_true = globals()['load_' + DATA_NAME]('TargetModel')
Target_Model = load_model(TARGET_WEIGHTS_PATH)

def top1_threshold_attack(x_, target):
    nonM_generated = np.random.uniform(0, 255, (1000, )+x_.shape[1:])
    threshold = np.percentile(target.predict(nonM_generated).max(axis=1), 90, interpolation='lower')
    m_pred = np.where(target.predict(x_).max(axis=1)>threshold, 1, 0)

    return m_pred


m_pred = top1_threshold_attack(np.r_[x_train_tar, x_test_tar], Target_Model)
evaluate_attack(m_pred, m_true)