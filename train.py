import os

import tensorflow as tf

from generator import SigmoidClassifierDataGenerator

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

train_image_path = r'.'
input_shape = (28, 28, 1)
batch_size = 64
lr = 0.01
momentum = 0.9
validation_split = 0.2
epochs = 500


def get_model(num_classes):
    global input_shape
    model_input = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=3,
        kernel_initializer='he_uniform',
        padding='same',
        activation='relu')(model_input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        kernel_initializer='he_uniform',
        padding='same',
        activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=3,
        kernel_initializer='he_uniform',
        padding='same',
        activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=3,
        kernel_initializer='he_uniform',
        padding='same',
        activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=3,
        kernel_initializer='he_uniform',
        padding='same',
        activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=3,
        kernel_initializer='he_uniform',
        padding='same',
        activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(
        filters=num_classes,
        kernel_size=1,
        activation='sigmoid')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    return tf.keras.models.Model(model_input, x)


def main():
    global train_image_path, input_shape, batch_size, lr, momentum, validation_split, epochs
    generator = SigmoidClassifierDataGenerator(train_image_path, input_shape, batch_size, validation_split)
    model = get_model(generator.get_num_classes())
    model.compile(
        optimizer=tf.keras.optimizers.SGD(lr=lr, momentum=momentum, nesterov=True),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    model.summary()
    model.fit(
        x=generator.flow(subset='training'),
        validation_data=generator.flow(subset='validation'),
        epochs=epochs,
        callbacks=tf.keras.callbacks.ModelCheckpoint(
            filepath='checkpoints/epoch_{epoch}_loss_{loss:.4f}_val_loss_{val_loss:.4f}_recall_{recall:.4f}_val_recall_{val_recall:.4f}.h5',
            monitor='val_recall',
            save_best_only=True,
            mode='max'))


if __name__ == '__main__':
    main()
