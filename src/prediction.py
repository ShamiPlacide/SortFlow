import json
from pathlib import Path
import tensorflow as tf
from .preprocessing import preprocess_image

MODELS_DIR = Path(__file__).resolve().parent.parent / 'models'
MODEL_PATH = MODELS_DIR / 'waste_cnn.h5'
CLASSES_PATH = MODELS_DIR / 'class_names.json'
_model, _names = None, None

def load_model(force=False):
    global _model, _names
    if _model is None or force:
        _model = tf.keras.models.load_model(MODEL_PATH)
        _names = json.loads(CLASSES_PATH.read_text())
    return _model, _names

def predict(image):
    model, names = load_model()
    probs = model.predict(preprocess_image(image), verbose=0)[0]
    i = int(probs.argmax())
    return {'class': names[i], 'confidence': float(probs[i]),
            'probabilities': {n: float(p) for n, p in zip(names, probs)}}
