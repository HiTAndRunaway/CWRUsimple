import matplotlib.pyplot as plt

from model import history, model
from preprocess import x_test, one_hot_test_labels

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