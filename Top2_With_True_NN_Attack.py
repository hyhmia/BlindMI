from dataLoader import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import metrics
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dropout, Dense, Activation
from tensorflow.keras.callbacks import ModelCheckpoint
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
DATA_NAME = sys.argv[1] if len(sys.argv) > 1 else "CIFAR"
TARGET_MODEL_GENRE = sys.argv[2] if len(sys.argv) > 2 else "ResNet50"
SHADOW_MODEL_GENRE = sys.argv[3] if len(sys.argv) > 3 else "VGG16"
EPOCHS = 40
BATCH_SIZE = 64
NUM_CLASSES = 1
LEARNING_RATE = 5e-5
NN_ATTACK_WEIGHTS_PATH = "weights/NN_Attack/BlackBox/Top2_True_NN_Attack_{}_{}.hdf5".format(DATA_NAME, SHADOW_MODEL_GENRE)
TARGET_WEIGHTS_PATH = "weights/Target/{}_{}.hdf5".format(DATA_NAME, TARGET_MODEL_GENRE)
SHADOW_WEIGHTS_PATH = "weights/BlackShadow/{}_{}.hdf5".format(DATA_NAME, SHADOW_MODEL_GENRE)

(x_train_sha, y_train_sha), (x_test_sha, y_test_sha), m_train = globals()['load_' + DATA_NAME]('ShadowModel')
Shadow_Model = load_model(SHADOW_WEIGHTS_PATH)
con_Score_Sha = Shadow_Model.predict(np.r_[x_train_sha, x_test_sha])
c_train = np.c_[con_Score_Sha[np.r_[y_train_sha, y_test_sha].astype(bool)], np.sort(con_Score_Sha, axis=1)[:, ::-1][:, :2]]

(x_train_tar, y_train_tar), (x_test_tar, y_test_tar), m_test = globals()['load_' + DATA_NAME]('TargetModel')
Target_Model = load_model(TARGET_WEIGHTS_PATH)
con_Score_Tar = Target_Model.predict(np.r_[x_train_tar, x_test_tar])
c_test = np.c_[con_Score_Tar[np.r_[y_train_tar, y_test_tar].astype(bool)], np.sort(con_Score_Tar, axis=1)[:, ::-1][:, :2]]

def create_attack_model(input_dim, num_classes=NUM_CLASSES):
    model = tf.keras.Sequential([
        Dense(512, input_dim=input_dim, activation='relu'),
        Dropout(0.2),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dense(num_classes),
        Activation('sigmoid')
    ])
    model.summary()
    return model

def train(model, x_train, y_train):
    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=LEARNING_RATE),
                  metrics=[metrics.BinaryAccuracy(), metrics.Precision(), metrics.Recall()])
    checkpoint = ModelCheckpoint(NN_ATTACK_WEIGHTS_PATH, monitor='precision', verbose=1, save_best_only=True,
                                 mode='max')
    model.fit(x_train, y_train,
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              callbacks=[checkpoint])

def evaluate(x_test, y_test):
    model = keras.models.load_model(NN_ATTACK_WEIGHTS_PATH)
    loss, accuracy, precision, recall = model.evaluate(x_test, y_test, verbose=1)
    F1_Score = 2 * (precision * recall) / (precision + recall)
    print('loss:%.4f accuracy:%.4f precision:%.4f recall:%.4f F1_Score:%.4f'
          % (loss, accuracy, precision, recall, F1_Score))


attackModel = create_attack_model(c_train.shape[1])
train(attackModel, c_train, m_train)
evaluate(c_test, m_test)