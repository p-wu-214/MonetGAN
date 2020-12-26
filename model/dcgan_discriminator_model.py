import tensorflow as tf

from tensorflow.keras import layers


class DCGanDiscriminatorModel:

    def make_discriminator_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(16, (5, 5), strides=(2, 2), padding='same',
                                input_shape=[256, 256, 1]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        return model
