import streamlit as st
import torch
import os
import soundfile as sf
import numpy as np
import pandas as pd
import io
import zipfile
from features import extract_frame_features
from utils import BiLSTMModel

st.title("Авторазметка аудио")

# === Настройки ===
HOP_LENGTH = 512
device = torch.device("cpu")
MODEL_PATH = "speech_seg_model.pt"

# === Загрузка модели ===
@st.cache_resource
def load_model():
    model = BiLSTMModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model.to(device)

model = load_model()

# === Обработка одиночного аудиофайла ===
def process_audio(file, participant_id, session):
    # Чтение файла
    y, sr = sf.read(file)
    if y.ndim > 1:
        y = y[:, 0]

    feats = extract_frame_features(y, sr)
    X_tensor = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)
    lengths = torch.tensor([X_tensor.shape[1]])
    
    with torch.no_grad():
        output = model(X_tensor, lengths)
        pred = torch.argmax(output, dim=-1).cpu().numpy()[0]

    # Выделяем сегменты речи
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

    # Строим DataFrame с нужными колонками
    audio_number = os.path.splitext(file.name)[0]
    df = pd.DataFrame([{
        "participant_ID": participant_id,
        "StimSite": "",
        "Stimulus": "",
        "session": session,
        "Response": "",
        "Response_transcription": "",
        "Error_type": "",
        "Comment": "",
        "filename": file.name,
        "pain": "",
        "RT_start": rs,
        "RT_end": re,
        "rater": rater,
        "audio_file": audio_number
    }])

    return df

# === Интерфейс Streamlit ===
uploaded_files = st.file_uploader("Выберите .wav файлы", type=["wav"], accept_multiple_files=True)

if uploaded_files:
    participant_id = st.text_input("ID участника (например, 03)", "")
    session = st.selectbox("Сессия", ["day1", "day2", "other"])
    rater = st.text_input("👩‍🔬 Имя экспериментатора", "")
    excel_buffers = []
    if participant_id and session:
        for file in uploaded_files:
            with st.spinner(f"Обработка {file.name}..."):
                df = process_audio(file, participant_id, session)

                # Временный буфер
                output = io.BytesIO()
                df.to_excel(output, index=False)
                output.seek(0)
                
                #excel_buffers.append((file.name.replace(".wav", ".xlsx"), output))
                excel_buffers.append((f'p{participant_id}_results.xlsx', output))
                # Вывод таблицы
                st.write(f"📄 Результат для {file.name}")
                st.dataframe(df)

                # # Кнопка скачивания Excel
                # output = io.BytesIO()
                # with pd.ExcelWriter(output, engine='openpyxl') as writer:
                #     df.to_excel(writer, index=False)
                # output.seek(0)

                # st.download_button(
                #     label=f"⬇ Скачать Excel для {file.name}",
                #     data=output,
                #     file_name=f"{file.name.replace('.wav', '')}_rt.xlsx",
                #     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                # )
                # Создание архива
        if excel_buffers:
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w") as zipf:
                        for filename, buffer in excel_buffers:
                            zipf.writestr(filename, buffer.read())
                    zip_buffer.seek(0)

                    st.download_button(
                        label="Скачать все в ZIP",
                        data=zip_buffer,
                        file_name="all_annotations.zip",
                        mime="application/zip"
                    )       
    else:
        st.warning("Пожалуйста, укажите ID участника и выберите сессию.")
