import argparse
import sys
import os
import tensorflow as tf
import numpy as np

import settings

from tensorflow.keras.layers import Dense
from swin import SwinTransformer
from data_generator import data_gen


def load_model() -> tf.keras.Model:
    base_model = SwinTransformer(
            'swin_supertiny_128',
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


if __name__ == "__main__":
    sys.path.append("..")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str, default="workspace/dataset")
    parser.add_argument("--model_path", required=True, type=str, default="workspace/model.h5")
    parser.add_argument("--batch_size", required=True, type=int)
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

    test_file = open(f'workspace/test_images_{size}_tswint.csv', 'w')
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
    print(f'batch size: {settings.BATCH_SIZE}')

    train_generator = data_gen(train_dataset_array, settings.BATCH_SIZE, tf.keras.applications.densenet.preprocess_input)
    val_generator = data_gen(val_dataset_array, settings.BATCH_SIZE, tf.keras.applications.densenet.preprocess_input)
 
    model = load_model()
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

