"""Model definitions. Rescaling is inside the model so the API and retraining only need to resize."""
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from .preprocessing import IMG_SIZE


def _compile(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')],
    )
    return model


def build_baseline(num_classes=6, lr=1e-3):
    """Vanilla CNN with no regularization. Used only for comparison."""
    model = models.Sequential([
        layers.Input(shape=IMG_SIZE + (3,)),
        layers.Rescaling(1. / 255),
        layers.Conv2D(32, 3, activation='relu'), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'), layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax'),
    ], name='baseline_cnn')
    return _compile(model, lr)


def build_custom(num_classes=6, lr=1e-3, dropout=0.3, l2=1e-5, filters=(32, 64, 128, 256)):
  reg = regularizers.l2(l2)
  inputs = layers.Input(shape=IMG_SIZE + (3,))
  x = layers.Rescaling(1. / 255)(inputs)
  for f in filters:
    for _ in range(2):
      x = layers.Conv2D(f, 3, padding='same', use_bias=False, kernel_regularizer=reg)(x)
      x = layers.BatchNormalization()(x)
      x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.1)(x)
  x = layers.GlobalAveragePooling2D()(x)
  x = layers.Dense(256, activation='relu', kernel_regularizer=reg)(x)
  x = layers.Dropout(dropout)(x)
  outputs = layers.Dense(num_classes, activation='softmax')(x)
  return _compile(models.Model(inputs, outputs, name='custom_cnn'), lr)
