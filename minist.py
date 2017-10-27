import numpy as np
from libDL import Loader, Saver
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten

# 加载训练数据
input_shape, (X_train, y_train), (X_test, y_test) = Loader.loading()

# 构建神经网络
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=input_shape, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='RMSprop',
                  metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, epochs=1, batch_size=32, 
            validation_data=(X_test, y_test), verbose=1)

# 保存预测数据
session_data = np.load('./predict/session-10-27.npy')
session_predict = model.predict(session_data)
Saver.save(session_predict)

