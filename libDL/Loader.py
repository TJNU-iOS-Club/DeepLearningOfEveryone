from keras.datasets import mnist
from keras.utils import to_categorical
from keras import backend as K

img_rows, img_cols = 28, 28

def loading():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    y_test, y_train = to_categorical(y_test, 10), to_categorical(y_train, 10)

    return input_shape, (X_train, y_train), (X_test, y_test)
