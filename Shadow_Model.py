from dataLoader import *
from ModelUtil import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import keras
from tensorflow.keras import metrics
import configparser
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
config = configparser.ConfigParser()
config.read('shadow_blackBox_config.ini')
DATA_NAME = sys.argv[1] if len(sys.argv) > 1 else "CIFAR"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "VGG16"
EPOCHS = int(config['{}_{}'.format(DATA_NAME, MODEL)]['EPOCHS'])
SAVED_FOLDER = config['{}_{}'.format(DATA_NAME, MODEL)]['SAVED_FOLDER']
BATCH_SIZE = 64
LEARNING_RATE = float(config['{}_{}'.format(DATA_NAME, MODEL)]['LEARNING_RATE'])
WEIGHTS_PATH = "weights/{}/{}_{}.hdf5".format(SAVED_FOLDER, DATA_NAME, MODEL)
(x_train, y_train), (x_test, y_test), _ = globals()['load_' + DATA_NAME]('ShadowModel')


def train(model, x_train, y_train, x_test, y_test):
    """
    Train the target model and save the weight of the model
    :param model: the model that will be trained
    :param x_train: the image as numpy format
    :param y_train: the label for x_train
    :param weights_path: path to save the model file
    :return: None
    """
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=LEARNING_RATE),
                  metrics=[metrics.CategoricalAccuracy(), metrics.Precision(), metrics.Recall()])

    model.fit(x_train,
              y_train,
              validation_data=(x_test, y_test),
              batch_size=BATCH_SIZE,
              epochs=EPOCHS)
    model.save(WEIGHTS_PATH)



def evaluate(x_test, y_test):
    model = keras.models.load_model(WEIGHTS_PATH)
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=5e-5),
                  metrics=[metrics.CategoricalAccuracy(), metrics.Precision(), metrics.Recall()])
    loss, accuracy, precision, recall = model.evaluate(x_test, y_test, verbose=1)
    F1_Score = 2 * (precision * recall) / (precision + recall)
    print('loss:%.4f accuracy:%.4f precision:%.4f recall:%.4f F1_Score:%.4f'
          % (loss, accuracy, precision, recall, F1_Score))


TargetModel = globals()['create_{}_model'.format(MODEL)](x_train.shape[1:], y_train.shape[1])

train(TargetModel, x_train, y_train, x_test, y_test)

evaluate(x_train, y_train)
evaluate(x_test, y_test)