import tensorflow as tf
import numpy as np

import settings
import vit_small

from data_generator import data_gen
from tensorflow.keras.layers import Dense
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image-size", default=512, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
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

    test_file = open(f'../workspace/test_images_{size}_{args.model_arch}_scratch.csv', 'w')
    for path, distance in test_dataset:
        test_file.write(f"{path}, {distance}\n")

    test_file.close()

    train_dataset = train_dataset[test_idx:]
    split_idx = int(settings.VALIDATION_SPLIT * len(train_dataset))

    train_dataset_array = train_dataset[split_idx:]
    val_dataset_array = train_dataset[:split_idx]

    print(f'train dataset: {len(train_dataset_array)}')
    print(f'valuation dataset: {len(val_dataset_array)}')
    print(f'test dataset: {len(test_dataset)}')
    print(f'model arch: {args.model_arch}')
    print(f'batch size: {settings.BATCH_SIZE}')
    print(f'patch size: {args.patch_size}')

    train_generator = data_gen(train_dataset_array, vit_small.preprocess_inputs, settings.BATCH_SIZE)
    val_generator = data_gen(val_dataset_array, vit_small.preprocess_inputs, settings.BATCH_SIZE)

    model = vit_small.tinyViT(
        image_size=size,
        include_top=False,
        patch_size=args.patch_size
    )

    x = model.output
    output_layer = Dense(1, activation='linear', name='prediction')(x)
    model = tf.keras.Model(inputs=model.input, outputs=output_layer)
    model.compile(
        loss=tf.keras.losses.log_cosh,
        metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError()],
        optimizer=tf.keras.optimizers.Adam(lr=settings.LEARNING_RATE)
    )

    steps_per_epoch = len(train_dataset_array) // settings.N_BATCH
    validation_steps = len(val_dataset_array) // settings.N_BATCH

    early_stop = tf.keras.callbacks.EarlyStopping(patience=10),
    mcp = tf.keras.callbacks.ModelCheckpoint(filepath=f'../workspace/{args.model}.h5', save_best_only=True,
                                             monitor='val_loss', verbose=1)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='auto',
        min_delta=0.0001, cooldown=0, min_lr=0)

    print(model.summary())

    model.fit(
        x=train_generator,
        epochs=settings.EPOCH,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=[mcp, early_stop, reduce_lr],
        verbose=1)
