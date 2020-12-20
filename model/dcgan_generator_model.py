import tensorflow as tf

from tensorflow.keras import layers

class DCGanGeneratorModel:

    def make_generator_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(units=16*16*256, use_bias=False, input_shape=(100,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((16, 16, 256)))
        assert model.output_shape == (None, 16, 16, 256)

        model.add(layers.Conv2DTranspose(128, kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 16, 16, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(64, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 32, 32, 64)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(32, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 64, 64, 32)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(16, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 128, 128, 16)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(1, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 256, 256, 1)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        return model