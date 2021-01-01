import tensorflow as tf

class MonetDataset:
    def __init__(self, pattern, transform, batch_size, auto_tune):
        self.pattern = pattern
        self.transform = transform
        self.batch_size = batch_size
        self.auto_tune = auto_tune

    def get(self):
        tfrec = self.__load()
        tfrec = tfrec.batch(batch_size=self.batch_size)
        tfrec = tfrec.prefetch(self.auto_tune)

        tfrec_iter = iter(tfrec)

        return tfrec_iter

    def __load(self):
        tfrec = tf.data.TFRecordDataset(self.pattern)
        tfrec = tfrec.map(self.transform, num_parallel_calls=1)

        return tfrec
