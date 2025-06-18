import streamlit as st
import torch
import soundfile as sf
import numpy as np
import pandas as pd
from utils import BiLSTMModel, extract_features
from features import extract_frame_features

HOP_LENGTH = 512
MODEL_PATH = 'speech_seg_model.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = BiLSTMModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Streamlit UI
st.title("Автоматическая разметка аудио (RT)")

uploaded_file = st.file_uploader("Загрузите WAV-файл", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file, format='audio/wav')
    y, sr = sf.read(uploaded_file)
    if y.ndim > 1:
        y = y[:, 0]
    feats = extract_frame_features(y, sr)

    with torch.no_grad():
        X_tensor = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)
        lengths = torch.tensor([X_tensor.shape[1]])
        output = model(X_tensor, lengths)
        pred = torch.argmax(output, dim=-1).cpu().numpy()[0]

    segs = []
    start = None
    for i, m in enumerate(pred):
        if m == 1 and start is None:
            start = i
        elif m == 0 and start is not None:
            segs.append((start, i - 1))
            start = None
    if start is not None:
        segs.append((start, len(pred) - 1))

    if not segs:
        rs, re = None, None
    else:
        ls, le = segs[-1]
        rs = int(ls * HOP_LENGTH / sr * 1000)
        re = int(le * HOP_LENGTH / sr * 1000)

    st.write(f"**RT_start:** {rs} мс")
    st.write(f"**RT_end:** {re} мс")

    df = pd.DataFrame([{
        'participant_ID': '',
        'StimSite': '',
        'Stimulus': '',
        'session': '',
        'Response': '',
        'Response_transcription': '',
        'Error_type': '',
        'Comment': '',
        'filename': uploaded_file.name,
        'pain': '',
        'RT_start': rs,
        'RT_end': re,
        'rater': '',
        'audio_file': ''.join(filter(str.isdigit, uploaded_file.name))
    }])

    st.dataframe(df)

    @st.cache_data
    def convert_df(df):
        return df.to_excel(index=False, engine='openpyxl')

    excel = convert_df(df)
    st.download_button("Скачать как Excel", excel, file_name="annotation.xlsx")