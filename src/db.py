import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / 'data' / 'app.db'
now = lambda: datetime.utcnow().isoformat(timespec='seconds')

def _c():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH); c.row_factory = sqlite3.Row
    return c

def q(sql, args=()):
    with _c() as c:
        return [dict(r) for r in c.execute(sql, args).fetchall()]

def x(sql, args=()):
    with _c() as c:
        return c.execute(sql, args).lastrowid

def init():
    with _c() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS uploads(id INTEGER PRIMARY KEY, filename TEXT, label TEXT,
            path TEXT, uploaded_at TEXT, used_in_run INTEGER);
        CREATE TABLE IF NOT EXISTS predictions(id INTEGER PRIMARY KEY, ts TEXT,
            predicted TEXT, confidence REAL, latency_ms REAL);
        CREATE TABLE IF NOT EXISTS runs(id INTEGER PRIMARY KEY, started_at TEXT, finished_at TEXT,
            status TEXT, n_new INTEGER, accuracy REAL, version INTEGER, message TEXT);
        """)

def add_upload(filename, label, path):
    return x("INSERT INTO uploads(filename,label,path,uploaded_at) VALUES(?,?,?,?)",
             (filename, label, str(path), now()))

def pending_count():
    return q("SELECT COUNT(*) n FROM uploads WHERE used_in_run IS NULL")[0]['n']

def uploads_per_class():
    return q("SELECT label, COUNT(*) n FROM uploads GROUP BY label")

def log_prediction(pred, conf, ms):
    x("INSERT INTO predictions(ts,predicted,confidence,latency_ms) VALUES(?,?,?,?)",
      (now(), pred, conf, ms))

def prediction_stats():
    return {'total': q("SELECT COUNT(*) n FROM predictions")[0]['n'],
            'avg_latency_ms': q("SELECT AVG(latency_ms) a FROM predictions")[0]['a'],
            'per_class': q("SELECT predicted, COUNT(*) n FROM predictions GROUP BY predicted"),
            'confidences': [r['confidence'] for r in q("SELECT confidence FROM predictions ORDER BY id DESC LIMIT 500")]}

def start_run(n_new):
    return x("INSERT INTO runs(started_at,status,n_new) VALUES(?,?,?)", (now(), 'running', n_new))

def finish_run(run_id, status, acc=None, version=None, msg=''):
    x("UPDATE runs SET finished_at=?,status=?,accuracy=?,version=?,message=? WHERE id=?",
      (now(), status, acc, version, msg, run_id))
    if status == 'success':
        x("UPDATE uploads SET used_in_run=? WHERE used_in_run IS NULL", (run_id,))

def runs():
    return q("SELECT * FROM runs ORDER BY id DESC LIMIT 20")
