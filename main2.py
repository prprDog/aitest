from tensorflow import keras
import numpy as np
from matplotlib.pylab import plt

x = np.random.random((10000, 1))
times = 1
print(x)
y = x ** times
x_train = x[:5000].astype('float32')
y_train = y[:5000].astype('float32')

x_val = x[5000:].astype('float32')
y_val = y[5000:].astype('float32')

x_test = np.random.uniform(low=1, high=100, size=(10000, 1))
y_test = (x_test ** times).astype('float32')

models = keras.Sequential([
    keras.layers.Dense(units=16, activation='relu'),
    keras.layers.Dense(units=16, activation='relu'),
    keras.layers.Dense(units=1),
])
models.compile(optimizer='rmsprop', loss='mse', metrics=["mae"])

history = models.fit(x_train, y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))
result = models.evaluate(x_test, y_test)
print(result)
predict = models.predict(x_test)
print(f"predict:{predict}")

history_dic = history.history
loss_values = history_dic['loss']
print(f'loss_values:{loss_values}')
val_loss_values = history_dic['val_loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, "bo", label="Training loss")
plt.plot(epochs, val_loss_values, "b", label="Vaildation loss")
plt.title("Training and vaildation loss")
plt.xlabel("Epchos")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.plot(x_test, y_test, "bo")
plt.legend()
plt.show()

