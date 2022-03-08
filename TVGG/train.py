import argparse
import os
import numpy as np
import tensorflow as tf

import settings

from tensorflow.keras.layers import Dense
from data_generator import data_gen


def load_vgg_tiny() -> tf.keras.Model:
    x = tf.keras.layers.Input(shape=settings.IMAGE_SIZE)
    y = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu")(x)
    y = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu")(y)
    y = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(y)
    y = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(y)
    y = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(y)
    y = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(y)
    y = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(y)
    y = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(y)
    y = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")(y)
    y = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(y)
    y = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(y)
    y = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(y)
    y = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(y)
    y = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(y)
    y = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(y)
    y = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(y)
    y = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(y)
    y = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(y)

    y = tf.keras.layers.GlobalAveragePooling2D()(y)
    y = Dense(1, activation='linear', name='predictions')(y)
    m = tf.keras.Model(inputs=x, outputs=y)

    if settings.FULL_TRAINING:
        print("USE FULL TRAINING")
        for layer in m.layers[:]:
            layer.trainable = True

    return m


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str, default="workspace/dataset")
    parser.add_argument("--model_path", required=True, type=str, default="workspace/model.h5")
    parser.add_argument("--model_arch", required=True, type=str)
    parser.add_argument("--batch_size", required=True, type=int)
    args = parser.parse_args()

    settings.BATCH_SIZE = args.batch_size

    train_dataset = []
    folders = [(f.path, int(f.name.replace('class_', ''))) for f in os.scandir(args.dataset) if f.is_dir()]
    for (folder, distance_z) in folders:
        train_dataset.extend([(f.path, distance_z) for f in os.scandir(folder) if f.name != "mire.png"])

    if settings.MODULO_ACTIVE:
        train_dataset = [e for e in train_dataset if e[1] % settings.MODULO_N == 0]

    np.random.shuffle(train_dataset)
    test_idx = int(settings.TEST_SPLIT * len(train_dataset))
    test_dataset = train_dataset[:test_idx]

    size = settings.IMAGE_SIZE[0]

    test_file = open(f'workspace/test_images_{size}_tvgg.csv', 'w')
    for path, distance in test_dataset:
        test_file.write(f"{path}, {distance}\n")

    test_file.close()

    train_dataset = train_dataset[test_idx:]
    split_idx = int(settings.VALIDATION_SPLIT * len(train_dataset))

    train_dataset_array = train_dataset[split_idx:]
    val_dataset_array = train_dataset[:split_idx]

    print(f'dataset: {len(train_dataset)}')
    print(f'train dataset: {len(train_dataset_array)}')
    print(f'valuation dataset: {len(val_dataset_array)}')
    print(f'model arch: {args.model_arch}')
    print(f'batch size: {settings.BATCH_SIZE}')

    model = load_vgg_tiny()
    train_generator = data_gen(train_dataset_array)
    val_generator = data_gen(val_dataset_array)
    print(model.summary())
    model.compile(tf.keras.optimizers.Adam(lr=settings.LEARNING_RATE),
                  loss=tf.keras.losses.log_cosh,
                  metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError()])

    steps_per_epoch = len(train_dataset_array) // settings.N_BATCH
    validation_steps = len(val_dataset_array) // settings.N_BATCH

    checkpoint = tf.keras.callbacks.ModelCheckpoint(args.model_path, verbose=1, save_best_only=True,
                                                    monitor="val_loss")
    early_stopper = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, verbose=1,
                                                     restore_best_weights=True)
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3, verbose=1)

    callbacks = [checkpoint, early_stopper, lr_reducer]

    history = model.fit(
        x=train_generator,
        epochs=settings.EPOCH,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1)
