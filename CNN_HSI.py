import tensorflow as tf
import numpy as np
from keras import models
from keras import layers
from keras import optimizers
from keras.utils import to_categorical

n_input_bands = 103
n_classes = 9
train_pixels = (np.load('train_data_syn.npy') / 8000 - 0.5) * 2
train_pixels = train_pixels.reshape(-1, 103, 1)
train_label = to_categorical(np.load('train_label_syn.npy') - 1)
eval_pixels = (np.load('eval_data.npy') / 8000 - 0.5) * 2
eval_pixels = eval_pixels.reshape(-1, 103, 1)
eval_label = to_categorical(np.load('eval_label.npy') - 1)
batch_size = 32


# def build_model(n_filters1=500, n_filters2=100, filter_size=3, pool_size=2, n_fc_units1=200, n_fc_units2=84):
def build_model(n_filters=20, filter_size=11, pool_size=3, n_fc_units=100):
    model = models.Sequential()
    # model.add(layers.Conv1D(n_filters1, filter_size, activation='relu', input_shape=(n_input_bands, 1)))
    # model.add(layers.MaxPool1D(pool_size=pool_size))
    # model.add(layers.Conv1D(n_filters2, filter_size, activation='relu'))
    # model.add(layers.MaxPool1D(pool_size=pool_size))
    # model.add(layers.Flatten())
    # model.add(layers.Dense(n_fc_units1, activation='relu'))
    # model.add(layers.Dense(n_fc_units2, activation='relu'))

    model.add(layers.Conv1D(n_filters, filter_size, activation='tanh', input_shape=(n_input_bands, 1)))
    model.add(layers.MaxPool1D(pool_size=pool_size))
    model.add(layers.Flatten())
    model.add(layers.Dense(n_fc_units, activation='tanh'))

    model.add(layers.Dense(n_classes, activation='softmax'))
    Adam = optimizers.Adam(decay=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=Adam, metrics=['accuracy'])
    return model


def preprocess_input(x):
    x /= 8000.0
    x -= 0.5
    x *= 2


def train(train_pixels, train_label):
    # train_generator = datagen.flow(x=train_pixels, y=train_label, batch_size=batch_size)
    model = build_model()
    model.fit(train_pixels, train_label, batch_size=batch_size, epochs=100)
    model.save("models/model.h5")
    return model


def evaluate(model):
    # eval_generator = datagen.flow(x=eval_pixels, y=eval_label, batch_size=batch_size)
    return model.evaluate(eval_pixels, eval_label, batch_size=batch_size)


if __name__ == '__main__':
    model = train(train_pixels, train_label)
    score = evaluate(model)
    print(score)
    del model



