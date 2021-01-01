import tensorflow as tf

from tensorflow.keras import layers

class DCGanGeneratorModel:

    def make_generator_model(self):
        model = tf.keras.Sequential()

        model.add(layers.Conv2DTranspose(3, kernel_size=3, strides=1, padding='same',
                                         use_bias=False, input_shape=(256, 256, 3)))
        assert model.output_shape == (None, 256, 256, 3)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2D(18, kernel_size=4, strides=2, padding='same',
                                         use_bias=False))
        print(model.output_shape)
        assert model.output_shape == (None, 128, 128, 18)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2D(36, kernel_size=4, strides=2, padding='same',
                                         use_bias=False))
        assert model.output_shape == (None, 64, 64, 36)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2D(72, kernel_size=4, strides=2, padding='same', use_bias=False))
        assert model.output_shape == (None, 32, 32, 72)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(36, kernel_size=4, strides=2, padding='same', use_bias=False))
        assert model.output_shape == (None, 64, 64, 36)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(18, kernel_size=4, strides=2, padding='same', use_bias=False))
        assert model.output_shape == (None, 128, 128, 18)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', use_bias=False))
        assert model.output_shape == (None, 256, 256, 3)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        return model