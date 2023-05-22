from tensorflow import keras
import numpy as np
from matplotlib.pylab import plt
import os
import random

input_dirs = "D:\\Data\\AI\\images"
target_dirs = "D:\\Data\\AI\\annotations\\trimaps"
input_img_paths = sorted([
    os.path.join(input_dirs, fname)
    for fname in os.listdir(input_dirs)
    if fname.endswith('.jpg')
])

target_paths = sorted([
    os.path.join(target_dirs, fname)
    for fname in os.listdir(target_dirs)
    if fname.endswith(".png") and not fname.startswith('.')
])


#
# plt.axis('off')
# plt.imshow(keras.utils.load_img(input_img_paths[7]))
# plt.show()


def display_target(target_array):
    normalized_array = (target_array.astype('uint8') - 1) * 127
    plt.axis('off')
    plt.imshow(normalized_array)
    plt.show()


img = keras.utils.img_to_array(keras.utils.load_img(target_paths[7], color_mode='grayscale'))
display_target(img)

img_size = (200, 200)
num_imgs = len(input_img_paths)

random.seed(2)
random.shuffle(input_img_paths)
random.seed(2)
random.shuffle(target_paths)


def path_to_input_img(path: str):
    img = keras.utils.img_to_array(
        keras.utils.load_img(path, target_size=img_size)
    )
    return img


def path_to_target_img(path: str):
    img = keras.utils.img_to_array(
        keras.utils.load_img(path, target_size=img_size, color_mode='grayscale')
    )
    img = img.astype('uint8') - 1
    return img


input_imgs = np.zeros((num_imgs,) + img_size + (3,), dtype='float32')
targets = np.zeros((num_imgs,) + img_size + (1,), dtype='uint8')
for i in range(num_imgs):
    input_imgs[i] = path_to_input_img(input_img_paths[i])
    targets[i] = path_to_target_img(target_paths[i])

num_val_sample = 1000
train_input_imgs = input_imgs[:-num_val_sample]
train_targets = targets[:-num_val_sample]
val_input_imgs = input_imgs[-num_val_sample:]
val_targets = targets[-num_val_sample:]


def get_model(img_size, num_class):
    inputs = keras.Input(shape=img_size + (3,))
    x = keras.layers.Rescaling(1 / 255.)(inputs)
    x = keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = keras.layers.Conv2D(128, 3, padding="same", activation="relu", strides=2)(x)
    x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = keras.layers.Conv2D(256, 3, padding="same", activation="relu", strides=2)(x)
    x = keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = keras.layers.Conv2DTranspose(256, 3, padding="same", activation="relu")(x)
    x = keras.layers.Conv2DTranspose(256, 1, padding="same", activation="relu", strides=2)(x)
    x = keras.layers.Conv2DTranspose(128, 1, padding="same", activation="relu")(x)
    x = keras.layers.Conv2DTranspose(128, 1, padding="same", activation="relu", strides=2)(x)
    x = keras.layers.Conv2DTranspose(64, 1, padding="same", activation="relu")(x)
    x = keras.layers.Conv2DTranspose(64, 1, padding="same", activation="relu", strides=2)(x)
    out_puts = keras.layers.Conv2D(num_class, 3, activation="softmax", padding="same")(x)
    model = keras.Model(inputs, out_puts)
    return model


model = get_model(img_size, num_class=3)
model.summary()

model.compile(loss=keras.losses.sparse_categorical_crossentropy)
callbacks = [
    keras.callbacks.ModelCheckpoint('oxford_segmentation.keras', save_best_only=True)
]
history = model.fit(
    train_input_imgs, train_targets, batch_size=64, epochs=50, callbacks=callbacks,
    validation_data=(val_input_imgs, val_targets)
)

loss = history.history['loss']
epochs = range(1, len(history.history['loss'] + 1))
val_loss = history.history['val_loss']
plt.figure()
plt.plot(epochs, loss, "bo", label="Training Loss")
plt.plot(epochs, val_loss, "b", label="Validation Loss")
plt.legend()
plt.show()
