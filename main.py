import tensorflow as tf
import matplotlib.pyplot as plt

import tqdm

from dataset.monet_dataset import MonetDataset

from model.dcgan_generator_model import DCGanGeneratorModel

from model.dcgan_discriminator_model import DCGanDiscriminatorModel

from tensorflow.keras.utils import plot_model

EPOCHS = 1

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
noise_dim = 100
BATCH_SIZE = 8

AUTOTUNE = tf.data.experimental.AUTOTUNE
directory = './data'

# From logits = not softmaxed (not in range of 0-1) (not probabilities)
CROSS_ENTROPY = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def transform(example):
    feature_map = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, feature_map)
    image = tf.io.decode_jpeg(example['image'])
    image = tf.image.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

    image = (image - 127.5) / 127.5

    return image

def display_image(num, monet_images, photo_images):
    example_monet = next(iter(monet_images))[:num-1]
    example_photo = next(iter(photo_images))[:num-1]

    for i in range(example_monet.shape[0]):
        plt.subplot(num-1, 2, i * 2 + 1)
        plt.imshow(tf.cast(example_monet[i], tf.uint8))

        plt.subplot(num-1, 2, i * 2 + 2)
        plt.imshow(tf.cast(example_photo[i], tf.uint8))
    plt.show()


def generator_loss_fn(disc_pred):
    return CROSS_ENTROPY(tf.ones_like(disc_pred), disc_pred)

def discriminator_loss_fn(generated, real):
    generated_loss = CROSS_ENTROPY(tf.zeros_like(generated), generated)
    real_loss = CROSS_ENTROPY(tf.ones(real), real)

    return generated_loss + real_loss

def __print_model(model, filename):
    plot_model(model, to_file='{filename}.png'.format(filename=filename), show_shapes=True, show_layer_names=True)

def main():

    monet_images_filenames = tf.io.gfile.glob(str(directory + '/monet_tfrec/*.tfrec'))
    photo_images_filenames = tf.io.gfile.glob(str(directory + '/photo_tfrec/*.tfrec'))

    monet_ds = MonetDataset(monet_images_filenames, transform, BATCH_SIZE, AUTOTUNE)
    photo_ds = MonetDataset(photo_images_filenames, transform, BATCH_SIZE, AUTOTUNE)

    monet = monet_ds.get()
    photo = photo_ds.get()

    dcg = DCGanGeneratorModel()
    dcd = DCGanDiscriminatorModel()

    generator = dcg.make_generator_model()
    discriminator = dcd.make_discriminator_model()

    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    __print_model(generator, 'generator')
    __print_model(discriminator, 'discriminator')

    # for epoch in range(EPOCHS):
    #     for mon, pho in tqdm(zip(monet, photo)):
    #         with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    #             generated_monet = generator(pho, training=True)
    #
    #             monet_pred = discriminator(mon, training=True)
    #             generated_monet_pred = discriminator(generated_monet, training=True)
    #
    #             generator_loss = generator_loss_fn(generated_monet_pred)
    #             discriminator_loss = discriminator_loss_fn(generated_monet_pred, monet_pred)
    #         gradients_of_generator = gen_tape.gradient(generator_loss, generator.trainable_variables)
    #         gradients_of_discriminator = disc_tape.gradient(discriminator_loss, discriminator.trainable_variables)
    #
    #         generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    #         discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))







if __name__ == '__main__':
    main()