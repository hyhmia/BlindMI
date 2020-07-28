from dataLoader import *
import tensorflow as tf
from tensorflow.keras.models import load_model
from BlindMIUtil import evaluate_attack
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
DATA_NAME = sys.argv[1] if len(sys.argv) > 1 else "CIFAR"
TARGET_MODEL_GENRE = sys.argv[2] if len(sys.argv) > 2 else "ResNet50"
NN_ATTACK_WEIGHTS_PATH = "weights/NN_Attack/NN_Attack_{}_{}.hdf5".format(DATA_NAME, TARGET_MODEL_GENRE)
TARGET_WEIGHTS_PATH = "weights/Target/{}_{}.hdf5".format(DATA_NAME, TARGET_MODEL_GENRE)

(x_train_tar, y_train_tar), (x_test_tar, y_test_tar), m_true = globals()['load_' + DATA_NAME]('TargetModel')

Target_Model = load_model(TARGET_WEIGHTS_PATH)


def Label_Only_Attack(x_, y_true):
    y_pred = Target_Model.predict_classes(x_)
    y_true = y_true.argmax(axis=1)
    m_pred = np.where(np.equal(y_pred, y_true), 1, 0)
    return m_pred

m_pred = Label_Only_Attack(np.r_[x_train_tar, x_test_tar], np.r_[y_train_tar, y_test_tar])
evaluate_attack(m_true, m_pred)
