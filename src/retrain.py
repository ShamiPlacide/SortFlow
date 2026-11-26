import shutil
from pathlib import Path
import tensorflow as tf
from . import db, prediction
from .preprocessing import load_dataset

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR, TEST_DIR = ROOT / 'data' / 'train', ROOT / 'data' / 'test'
UPLOADS = ROOT / 'data' / 'uploads'
VERSIONS = ROOT / 'models' / 'versions'
IMG_EXT = {'.jpg', '.jpeg', '.png'}

def ensure_upload_dirs(names):
    for n in names:
        (UPLOADS / n).mkdir(parents=True, exist_ok=True)

def _has_images(d):
    return d.exists() and any(p.suffix.lower() in IMG_EXT for p in d.rglob('*'))

def retrain(epochs=3, lr=1e-4):
    model, names = prediction.load_model()
    ensure_upload_dirs(names)
    run_id = db.start_run(db.pending_count())
    try:
        parts = []
        if _has_images(UPLOADS):
            parts.append(load_dataset(UPLOADS, shuffle=True, augment=True, class_names=names)[0])
        if _has_images(TRAIN_DIR):   # small sample of the original data to avoid forgetting
            parts.append(load_dataset(TRAIN_DIR, shuffle=True, augment=True, class_names=names)[0])
        if not parts:
            raise RuntimeError('No training data found')
        ds = parts[0].unbatch()
        for p in parts[1:]:
            ds = ds.concatenate(p.unbatch())
        ds = ds.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),   # low LR: fine-tune, don't overwrite
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall')],
        )
        model.fit(ds, epochs=epochs, verbose=0)

        acc = None
        if _has_images(TEST_DIR):
            test_ds, _ = load_dataset(TEST_DIR, class_names=names)
            res = dict(zip(model.metrics_names, model.evaluate(test_ds, verbose=0)))
            acc = float(res['accuracy'])

        VERSIONS.mkdir(parents=True, exist_ok=True)
        version = len(list(VERSIONS.glob('*.h5'))) + 1
        model.save(VERSIONS / f'waste_cnn_v{version}.h5')
        shutil.copy(VERSIONS / f'waste_cnn_v{version}.h5', prediction.MODEL_PATH)
        prediction.load_model(force=True)
        db.finish_run(run_id, 'success', acc, version, 'ok')
    except Exception as e:
        db.finish_run(run_id, 'failed', msg=str(e))
        raise
