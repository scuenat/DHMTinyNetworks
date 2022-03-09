import argparse
import sys
import os
from typing import Optional

import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Dense

import settings
import TViT.tvit as tvit

from TSwinT.swin import SwinTransformer
from data_generator import data_gen


def load_tvgg() -> tf.keras.Model:
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


def load_tvit_model() -> tf.keras.Model:
    base_model = tvit.tinyViT(
        image_size=size,
        include_top=False,
        patch_size=16
    )

    x = base_model.output
    output_layer = Dense(1, activation='linear', name='prediction')(x)
    m = tf.keras.Model(inputs=base_model.input, outputs=output_layer)
    m.compile(
        loss=tf.keras.losses.log_cosh,
        metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError()],
        optimizer=tf.keras.optimizers.Adam(lr=settings.LEARNING_RATE)
    )
    return m


def load_tswint_model() -> tf.keras.Model:
    base_model = SwinTransformer(
            'tswint',
            pretrained=False,
            include_top=False,
        )
    x = base_model.output
    output_layer = Dense(1, activation='linear', name='prediction')(x)
    m = tf.keras.Model(inputs=base_model.input, outputs=output_layer)

    if settings.FULL_TRAINING:
        print("USE FULL TRAINING")
        for layer in m.layers[:]:
            layer.trainable = True

    return m


def load_model(model_type: str) -> Optional[tf.keras.Model]:
    if model_type == "tswint":
        return load_tswint_model()
    if model_type == "tvgg":
        return load_tvgg()
    if model_type == "tvit":
        return load_tvit_model()
    return None


if __name__ == "__main__":
    sys.path.append("..")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str, default="workspace/dataset")
    parser.add_argument("--batch_size", required=True, type=int)
    parser.add_argument("--num_roi", required=True, type=int)
    parser.add_argument("--model_type", required=True, type=str)
    args = parser.parse_args()

    settings.BATCH_SIZE = args.batch_size

    train_dataset = []
    folders = [(f.path, int(f.name.replace('class_', ''))) for f in os.scandir(args.dataset) if f.is_dir()]
    for (folder, distance_z) in folders:
        train_dataset.extend([(f.path, distance_z) for f in os.scandir(folder) if f.name != "mire.png"])

    np.random.shuffle(train_dataset)
    test_idx = int(settings.TEST_SPLIT * len(train_dataset))
    test_dataset = train_dataset[:test_idx]

    size = settings.IMAGE_SIZE[0]

    workspace_path = "workspace"
    is_workspace_exists = os.path.exists(workspace_path)
    if not is_workspace_exists:
        os.makedirs(workspace_path)

    test_file = open(f'{workspace_path}/test_images_{size}_{args.model_type}.csv', 'w')
    for path, distance in test_dataset:
        test_file.write(f"{path}, {distance}\n")
    test_file.close()

    train_dataset = train_dataset[test_idx:]
    split_idx = int(settings.VALIDATION_SPLIT * len(train_dataset))
    train_dataset_array = train_dataset[split_idx:]
    val_dataset_array = train_dataset[:split_idx]

    pre_processing = tf.keras.applications.vgg16.preprocess_input
    if args.model_type == "tvit":
        pre_processing = tvit.preprocess_inputs
    if args.model_type == "tswint":
        pre_processing = tf.keras.applications.densenet.preprocess_input

    train_generator = data_gen(train_dataset_array, args.batch_size, args.num_roi, pre_processing)
    val_generator = data_gen(val_dataset_array, args.batch_size, args.num_roi, pre_processing)
 
    model = load_model(args.model_type)
    if model is None:
        print("model type not found...")
        exit()

    model_path = f'{workspace_path}/{args.model_type}.h5'
    print(f"Training starting for {args.model_type}")
    print(f"batch size = {args.batch_size}")
    print(f"num roi = {args.num_roi}")
    print(f"model path = {model_path}")
    print(f"dataset = {args.dataset}")

    print(model.summary())

    model.compile(tf.keras.optimizers.Adam(lr=settings.LEARNING_RATE),
                  loss=tf.keras.losses.log_cosh,
                  metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError()])

    steps_per_epoch = len(train_dataset_array) // args.batch_size
    validation_steps = len(val_dataset_array) // args.batch_size

    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_path, verbose=1, save_best_only=True,
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

