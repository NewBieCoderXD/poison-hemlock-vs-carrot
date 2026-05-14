import keras
from keras import layers, callbacks
from tensorflow import data as tf_data
import tensorflow as tf
import pathlib
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

image_size = (None, None)
batch_size = 32 

class_names = ["Conium_maculatum_L.", "Daucus_carota_L."]
data_dir = pathlib.Path("downloads")


# Helper to get label from path (assuming downloads/class_name/image.jpg)
def process_path(file_path):
    # Extract class name from path
    parts = tf.strings.split(file_path, "/")
    class_name = parts[-2]

    # Convert string class name to integer (0 or 1)
    label = tf.argmax(class_name == class_names)

    # Load and Resize
    img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(img, channels=3)

    # resize_with_pad is great: it keeps aspect ratio and adds black bars
    img = tf.image.resize_with_pad(img, 256, 256)
    img = tf.image.convert_image_dtype(img, tf.float32)

    return img, label


# 1. Get all file paths
list_ds = tf.data.Dataset.list_files(str(data_dir / "*/*"), shuffle=True, seed=1337)

# 2. Define split sizes
dataset_size = tf.data.experimental.cardinality(list_ds).numpy()
val_size = int(dataset_size * 0.2)
train_size = dataset_size - val_size

# 3. Create the subsets
train_ds = list_ds.take(train_size)
val_ds = list_ds.skip(train_size)

# 4. Map the loading function (process_path from the previous step)
train_ds = train_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

# 5. Batch them (using padding for dynamic sizes)
train_ds = train_ds.padded_batch(batch_size=batch_size).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.padded_batch(batch_size=batch_size).prefetch(tf.data.AUTOTUNE)

data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]


def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images


inputs = keras.Input(shape=image_size)
x = data_augmentation(inputs)
x = layers.Rescaling(1.0 / 255)(x)

train_ds = train_ds.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf_data.AUTOTUNE,
)
# Prefetching samples in GPU memory helps maximize GPU utilization.
train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
val_ds = val_ds.prefetch(tf_data.AUTOTUNE)


def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(64, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        units = 1
    else:
        units = num_classes

    x = layers.Dropout(0.25)(x)
    # We specify activation=None so as to return logits
    outputs = layers.Dense(units, activation='sigmoid')(x)
    return keras.Model(inputs, outputs)


model = make_model(input_shape=image_size + (3,), num_classes=2)
keras.utils.plot_model(model, show_shapes=True)
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=2000,
    decay_rate=0.9
)
optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)
model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
checkpoint = callbacks.ModelCheckpoint(
    filepath='best_model.keras',   # Path where the model will be saved
    monitor='val_accuracy',        # Metric to monitor (use 'accuracy' for training accuracy)
    mode='max',                    # 'max' because higher accuracy is better
    save_best_only=True,           # Only save when a new 'best' is reached
    verbose=1                      # Log when a new best model is saved
)

print("Starting training on GPU...")
model.fit(train_ds, validation_data=val_ds, epochs=50, callbacks=[checkpoint],)
