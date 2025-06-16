import streamlit as st
import imageio
import tempfile
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Mapping label index ke huruf A–Y (tanpa J dan Z)
label_map = [chr(i) for i in range(65, 91) if i not in [74, 90]]  # 25 huruf

# Load model
model = load_model("model/sign_language_model.h5")

st.set_page_config(page_title="Penerjemah Bahasa Isyarat", layout="centered")
st.title("🤟 Penerjemah Bahasa Isyarat (Huruf A–Y)")
st.write("Unggah video gesture huruf tangan dan sistem akan menerjemahkannya ke teks.")

uploaded_file = st.file_uploader("Upload video (.mp4)", type=["mp4", "mov"])

if uploaded_file is not None:
    # Simpan video ke file sementara
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    st.video(tfile.name)

    st.info("🔍 Menyusun terjemahan...")
    reader = imageio.get_reader(tfile.name)
    hasil_teks = []

    try:
        # Ambil 1 frame setiap 15 (agar tidak berlebihan)
        for i, frame in enumerate(reader):
            if i % 15 == 0:
                img = Image.fromarray(frame).convert("L")  # grayscale
                img = img.resize((28, 28))
                img_array = np.array(img).reshape(1, 28, 28, 1) / 255.0
                pred = model.predict(img_array)
                label_index = np.argmax(pred)
                hasil_teks.append(label_map[label_index])
        reader.close()

        hasil = ''.join(hasil_teks)
        st.success(f"📝 Hasil Terjemahan: `{hasil}`")

    except Exception as e:
        st.error("Gagal membaca frame video atau prediksi gesture: " + str(e))
