from keras import layers
from keras import models
from preprocess import one_hot_train_labels, x_train
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽通知信息、警告信息和报错信息

model = models.Sequential()
model.add(layers.Conv2D(4, (10, 10), padding='same', activation='relu', input_shape=(20, 20, 1)))
model.add(layers.MaxPooling2D((3, 3), strides=2))
model.add(layers.Conv2D(4, (5, 5), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((3, 3), strides=2))
model.add(layers.Conv2D(8, (3, 3), padding='same', activation='relu'))
model.add(layers.Conv2D(8, (3, 3), padding='same', activation='relu'))
model.add(layers.Conv2D(8, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((3, 3), strides=2))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add((layers.Dense(6, activation='softmax')))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, one_hot_train_labels, epochs=20, batch_size=50)