"""Preprocessing utilities shared by the notebook, training, retraining, and the API."""
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

# Applied during training only
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomBrightness(0.2, value_range=(0, 255)),
    tf.keras.layers.RandomContrast(0.2),
], name='augmentation')


def load_dataset(directory, shuffle=False, augment=False, batch_size=BATCH_SIZE, class_names=None):
    """Load a folder-per-class image directory into a ready-to-train tf.data pipeline.
    Pass class_names to guarantee the same label order as the original model."""
    ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=shuffle,
        class_names=class_names,
        seed=42,
    )
    names = ds.class_names
    if augment:
        ds = ds.map(lambda x, y: (augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE), names


def compute_class_weights(directory):
    """Inverse-frequency class weights to handle imbalance."""
    directory = Path(directory)
    classes = sorted(p.name for p in directory.iterdir() if p.is_dir())
    counts = np.array([len([f for f in (directory / c).glob('*') if f.is_file()]) for c in classes])
    total = counts.sum()
    return {i: total / (len(classes) * c) for i, c in enumerate(counts)}


def preprocess_image(image):
    """Single image (path, bytes-like file object, or PIL image) -> model-ready batch of shape (1,224,224,3).
    Used by the prediction API so inference matches training exactly."""
    from PIL import Image
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = image.convert('RGB').resize(IMG_SIZE)
    arr = np.asarray(image, dtype=np.float32)
    return preprocess_input(np.expand_dims(arr, axis=0))
