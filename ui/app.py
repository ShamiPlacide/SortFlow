import os
import pandas as pd
import requests
import streamlit as st

API = os.getenv('API_URL', 'http://localhost:8000')
CLASSES = ['cardboard', 'glass', 'metal', 'organic', 'paper', 'plastic']
TRAIN_COUNTS = {'cardboard': 814, 'glass': 1814, 'metal': 706,
                'organic': 889, 'paper': 964, 'plastic': 782}

st.set_page_config(page_title='SortFlow', layout='wide')
st.title('SortFlow: Waste Classifier')
page = st.sidebar.radio('Page', ['Predict', 'Insights', 'Upload & Retrain', 'Uptime'])

def get(path):
    try:
        return requests.get(f'{API}{path}', timeout=10).json()
    except Exception:
        return None

if page == 'Predict':
    f = st.file_uploader('Upload a waste photo', type=['jpg', 'jpeg', 'png'])
    if f:
        st.image(f, width=300)
        if st.button('Classify'):
            r = requests.post(f'{API}/predict', files={'file': (f.name, f.getvalue(), f.type)})
            if r.ok:
                d = r.json()
                st.success(f"{d['class']} ({d['confidence']:.1%})")
                st.bar_chart(pd.Series(d['probabilities']))
            else:
                st.error(r.text)

elif page == 'Insights':
    st.subheader('Training data: images per class')
    st.bar_chart(pd.Series(TRAIN_COUNTS))
    st.caption('Glass dominates (about 2.6x the smallest class), so accuracy alone is misleading and we use per-class F1 and class weights.')
    # Add your other notebook plots by saving them to ui/assets/ and using st.image('ui/assets/brightness.png')
    s = get('/stats')
    if s:
        st.subheader('Live predictions by class')
        pc = s['predictions']['per_class']
        if pc:
            st.bar_chart(pd.DataFrame(pc).set_index('predicted')['n'])
        conf = s['predictions']['confidences']
        if conf:
            st.subheader('Confidence distribution')
            st.bar_chart(pd.cut(pd.Series(conf), bins=10).value_counts().sort_index().rename(str))

elif page == 'Upload & Retrain':
    st.subheader('1. Upload labelled images')
    label = st.selectbox('Class of these images', CLASSES)
    files = st.file_uploader('Images', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
    if files and st.button('Upload'):
        r = requests.post(f'{API}/upload', data={'label': label},
                          files=[('files', (f.name, f.getvalue(), f.type)) for f in files])
        st.json(r.json() if r.ok else r.text)
    st.subheader('2. Retrain')
    s = get('/status')
    if s:
        st.write(f"Pending new images: **{s['pending']}** / auto-trigger at {s['threshold']}")
        if s['retraining']:
            st.warning('Retraining in progress. Refresh to update.')
        if st.button('Trigger retraining', disabled=s['retraining']):
            r = requests.post(f'{API}/retrain')
            st.json(r.json() if r.ok else r.text)
        st.button('Refresh status')
        if s['runs']:
            st.dataframe(pd.DataFrame(s['runs']))

elif page == 'Uptime':
    h = get('/health')
    if h:
        c1, c2, c3 = st.columns(3)
        c1.metric('Status', h['status'])
        m, sec = divmod(h['uptime_seconds'], 60)
        c2.metric('Uptime', f'{m // 60}h {m % 60}m {sec}s')
        c3.metric('Model loaded', str(h['model_loaded']))
        st.write('Retraining now:', h['retraining'])
        s = get('/stats')
        if s and s['predictions']['avg_latency_ms']:
            st.metric('Avg prediction latency', f"{s['predictions']['avg_latency_ms']:.0f} ms")
    else:
        st.error('API unreachable')
    st.button('Refresh')
