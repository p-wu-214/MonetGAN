import tensorflow as tf
import matplotlib.pyplot as plt

from dataset.monet_dataset import MonetDataset

from model.dcgan_generator_model import DCGanGeneratorModel

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

BATCH_SIZE = 8

AUTOTUNE = tf.data.experimental.AUTOTUNE
directory = './data'

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

def main():

    monet_images_filenames = tf.io.gfile.glob(str(directory + '/monet_tfrec/*.tfrec'))
    photo_images_filenames = tf.io.gfile.glob(str(directory + '/photo_tfrec/*.tfrec'))

    monet_ds = MonetDataset(monet_images_filenames, transform, BATCH_SIZE, AUTOTUNE)
    photo_ds = MonetDataset(photo_images_filenames, transform, BATCH_SIZE, AUTOTUNE)

    monet = monet_ds.get()
    photo = photo_ds.get()

    dcg = DCGanGeneratorModel()

    model = dcg.make_generator_model()
    noise = tf.random.normal([1, 100])
    generated_example = model(noise, training=False)

    plt.imshow(generated_example[0, :, :, 0])
    plt.show()



if __name__ == '__main__':
    main()