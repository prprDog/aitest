from tensorflow import keras
import numpy as np
from matplotlib.pylab import plt

(train_data, train_label), (test_data, test_label) = keras.datasets.imdb.load_data(num_words=10000)
print(train_label)


def vectorize_sequences(sequence: list, dimension=10000):
    results = np.zeros((len(sequence), dimension))
    for i, sequence in enumerate(sequence):
        for j in sequence:
            results[i, j] = 1
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_label).astype("float32")
y_test = np.asarray(test_label).astype("float32")
print(f'length:{len(x_train)}, {x_train}')

model = keras.Sequential([
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train, partial_y_train, epochs=6, batch_size=512, validation_data=(x_val, y_val))

result = model.evaluate(x_test, y_test)
print(f"predict:{result}")

print(
    f"predict:{model.predict(x_test)}"
)
print(f"real:{y_test}")
# history_dic = history.history
# loss_values = history_dic['loss']
# print(f'loss_values:{loss_values}')
# val_loss_values = history_dic['val_loss']
# epochs = range(1, len(loss_values) + 1)
# plt.plot(epochs, loss_values, "bo", label="Training loss")
# plt.plot(epochs, val_loss_values, "b", label="Vaildation loss")
# plt.title("Training and vaildation loss")
# plt.xlabel("Epchos")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()
