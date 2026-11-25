import io, os, shutil, time, uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from data.test.src import retrain as rt, prediction, db

START = time.time()
THRESHOLD = int(os.getenv('RETRAIN_THRESHOLD', 50))
state = {'retraining': False}
app = FastAPI(title='SortFlow API')

@app.on_event('startup')
def startup():
    db.init()
    try:
        _, names = prediction.load_model()
        rt.ensure_upload_dirs(names)
    except Exception as e:
        print('Model not loaded:', e)

def _run_retrain():
    if state['retraining']:
        return
    state['retraining'] = True
    try:
        rt.retrain()
    finally:
        state['retraining'] = False

@app.get('/health')
def health():
    return {'status': 'ok', 'uptime_seconds': round(time.time() - START),
            'model_loaded': prediction._model is not None, 'retraining': state['retraining']}

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    t = time.time()
    try:
        result = prediction.predict(io.BytesIO(await file.read()))
    except Exception as e:
        raise HTTPException(400, f'Could not predict: {e}')
    db.log_prediction(result['class'], result['confidence'], (time.time() - t) * 1000)
    return result

@app.post('/upload')
async def upload(bg: BackgroundTasks, label: str = Form(...), files: list[UploadFile] = File(...)):
    _, names = prediction.load_model()
    if label not in names:
        raise HTTPException(400, f'label must be one of {names}')
    saved = 0
    for f in files:
        ext = Path(f.filename).suffix.lower()
        if ext not in rt.IMG_EXT:
            continue
        dest = rt.UPLOADS / label / f'{uuid.uuid4().hex}{ext}'
        with open(dest, 'wb') as out:
            shutil.copyfileobj(f.file, out)
        db.add_upload(f.filename, label, dest)
        saved += 1
    auto = db.pending_count() >= THRESHOLD and not state['retraining']
    if auto:
        bg.add_task(_run_retrain)
    return {'saved': saved, 'pending': db.pending_count(),
            'threshold': THRESHOLD, 'auto_retrain_started': auto}

@app.post('/retrain')
def retrain(bg: BackgroundTasks):
    if state['retraining']:
        raise HTTPException(409, 'Retraining already running')
    bg.add_task(_run_retrain)
    return {'message': 'Retraining started'}

@app.get('/status')
def status():
    return {'retraining': state['retraining'], 'pending': db.pending_count(),
            'threshold': THRESHOLD, 'runs': db.runs()}

@app.get('/stats')
def stats():
    return {'uploads_per_class': db.uploads_per_class(), 'predictions': db.prediction_stats()}
