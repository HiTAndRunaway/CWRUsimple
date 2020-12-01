import numpy as np
import matplotlib.pyplot as plt
from keras import layers
from keras import models


def yuchuli(data, label):
    a = np.loadtxt(data)
    a = a[0:120000]
    a = a.reshape(300, 400)
    np.random.shuffle(a)
    train = a[:200, :]
    test = a[200:, :]
    label_test = np.array([label for i in range(0, 100)])
    label_train = np.array([label for i in range(0, 200)])
    return train, test, label_train, label_test


def stackkk(a, b, c, d, e, f, g, h):
    aa = np.vstack((a, e))
    bb = np.vstack((b, f))
    cc = np.hstack((c, g))
    dd = np.hstack((d, h))
    return aa, bb, cc, dd


x_tra0, x_tes0, y_tra0, y_tes0 = yuchuli('E:\\allDataSet\\cwru\\tempotetory data\\X098_DE_time.txt', 0)
x_tra1, x_tes1, y_tra1, y_tes1 = yuchuli('E:\\allDataSet\\cwru\\tempotetory data\\X106_DE_time.txt', 1)
x_tra2, x_tes2, y_tra2, y_tes2 = yuchuli('E:\\allDataSet\\cwru\\tempotetory data\\X119_DE_time.txt', 2)
x_tra3, x_tes3, y_tra3, y_tes3 = yuchuli('E:\\allDataSet\\cwru\\tempotetory data\\X145_DE_time.txt', 3)
x_tra4, x_tes4, y_tra4, y_tes4 = yuchuli('E:\\allDataSet\\cwru\\tempotetory data\\X131_DE_time.txt', 4)
x_tra5, x_tes5, y_tra5, y_tes5 = yuchuli('E:\\allDataSet\\cwru\\tempotetory data\\X158_DE_time.txt', 5)

tr1, te1, yr1, ye1 = stackkk(x_tra0, x_tes0, y_tra0, y_tes0, x_tra1, x_tes1, y_tra1, y_tes1)
tr2, te2, yr2, ye2 = stackkk(tr1, te1, yr1, ye1, x_tra2, x_tes2, y_tra2, y_tes2)
tr3, te3, yr3, ye3 = stackkk(tr2, te2, yr2, ye2, x_tra3, x_tes3, y_tra3, y_tes3)
tr4, te4, yr4, ye4 = stackkk(tr3, te3, yr3, ye3, x_tra4, x_tes4, y_tra4, y_tes4)
tr5, te5, yr5, ye5 = stackkk(tr4, te4, yr4, ye4, x_tra5, x_tes5, y_tra5, y_tes5)
print('测试标签:', ye5, '\n', '训练标签：', yr5)

y_train = yr5
y_test = ye5

# 转化为二维矩阵

x_train = tr5.reshape(1200, 20, 20, 1)
x_test = te5.reshape(600, 20, 20, 1)

state = np.random.get_state()
np.random.shuffle(x_train)
np.random.set_state(state)
np.random.shuffle(y_train)


def to_one_hot(labels, dimension=6):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1
    return results


one_hot_train_labels = to_one_hot(y_train)
one_hot_test_labels = to_one_hot(y_test)

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
# 绘制训练过程中的损失曲线和精度曲线


acc = history.history['accuracy']
loss = history.history['loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.title('Training  accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.title('Training  loss')
plt.legend()
plt.show()

print(history.history.keys())
print(model.evaluate(x_test, one_hot_test_labels))
