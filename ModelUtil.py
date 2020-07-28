from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Activation,Conv2D, MaxPooling2D,Flatten
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, ResNet101, VGG16, VGG19, DenseNet121


def create_ResNet50_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        ResNet50(include_top=False,
                 weights='imagenet',
                 input_shape=input_shape),
        GlobalAveragePooling2D(),
        Dense(num_classes),
        Activation("softmax")
    ])
    model.summary()
    return model


def create_ResNet101_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        ResNet101(include_top=False,
                 weights='imagenet',
                 input_shape=input_shape),
        GlobalAveragePooling2D(),
        Dense(num_classes),
        Activation("softmax")
    ])
    model.summary()
    return model


def create_VGG16_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        VGG16(include_top=False,
                 weights='imagenet',
                 input_shape=input_shape),
        GlobalAveragePooling2D(),
        Dense(num_classes),
        Activation("softmax")
    ])
    model.summary()
    return model


def create_VGG19_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        VGG19(include_top=False,
                 weights='imagenet',
                 input_shape=input_shape),
        GlobalAveragePooling2D(),
        Dense(num_classes),
        Activation("softmax")
    ])
    model.summary()
    return model


def create_DenseNet121_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        DenseNet121(include_top=False,
                 weights='imagenet',
                 input_shape=input_shape),
        GlobalAveragePooling2D(),
        Dense(num_classes),
        Activation("softmax")
    ])
    model.summary()
    return model


def create_CNN_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes),
        Activation('softmax')
    ])
    model.summary()
    return model


def create_Dense_3_layer_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        Dense(512, activation='relu', input_shape=input_shape),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(num_classes),
        Activation('softmax')
    ])
    model.summary()
    return model


def create_Dense_4_layer_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        Dense(1024, activation='relu', input_shape=input_shape),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(num_classes),
        Activation('softmax')
    ])
    model.summary()
    return model


def create_Dense_5_layer_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        Dense(2048, activation='relu', input_shape=input_shape),
        Dense(1024, activation='relu'),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(num_classes),
        Activation('softmax')
    ])
    model.summary()
    return model


def create_Dense_6_layer_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        Dense(4096, activation='relu', input_shape=input_shape),
        Dense(2048, activation='relu'),
        Dense(1024, activation='relu'),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(num_classes),
        Activation('softmax')
    ])
    model.summary()
    return model


def create_Dense_7_layer_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        Dense(8192, activation='relu', input_shape=input_shape),
        Dense(4096, activation='relu'),
        Dense(2048, activation='relu'),
        Dense(1024, activation='relu'),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(num_classes),
        Activation('softmax')
    ])
    model.summary()
    return model
